from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer
import torch

class CollegeResultsDataset(torch.utils.data.Dataset):
    def __init__(self, data, colleges_list, stopwords):
        self.data = []
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.stopwords = stopwords
        self.embedding_dim = 16
        self.num_embeddings_list = [2, 3, 5, 5, 40, 5, 5, 2, 4, len(colleges_list)]
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for post_id, post in data.items():
            for college in post['results']:
                self.data.append({
                    'post': post,
                    'college': college,
                    'college_id': colleges_list.index(college['school_name'])
                })

    def __len__(self):
        return len(self.data)
    
    def embed_text(self, text, max_length):
        inputs = self.tokenizer.encode_plus(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        return torch.tensor(self.model(**inputs)['last_hidden_state'][0][0], dtype=torch.float32)

    def __getitem__(self, i):
        item = self.data[i]
        post = item['post']
        college = item['college']

        numerical_inputs = torch.tensor([
            int(post['ethnicity']),
            int(post['gender']),
            int(post['income_bracket']),
            int(post['gpa']),
            int(post['apib_number']),
            int(post['apib_scores']),
            int(post['standardized_test_scores']),
            int(college['in_state']),
            int(college['round']),
            item['college_id']
        ], dtype=torch.float)

        new_numerical_inputs = torch.empty((10,16))

        for i, n in enumerate(self.num_embeddings_list):
            new_numerical_inputs[i] = torch.nn.Embedding(n, self.embedding_dim)(numerical_inputs[i].long())
        numerical_inputs = torch.flatten(new_numerical_inputs)

        major = self.remove_stopwords(post['major'])
        residence = self.remove_stopwords(post['residence'])
        extracurriculars = self.remove_stopwords('\n'.join(post['extracurriculars'] + post['awards']))

        major_embedding = self.embed_text(major, 20)
        residence_embedding = self.embed_text(residence, 10)
        ecs_embedding = self.embed_text(extracurriculars, 512)

        return {
            'numerical_inputs': numerical_inputs.detach(),
            'major_embedding': major_embedding.detach(),
            'residence_embedding': residence_embedding.detach(),
            'ecs_embedding': ecs_embedding.detach(),
            'target': torch.tensor(college['accepted'], dtype=torch.float32).detach()
        }

    def remove_stopwords(self, text):
        filtered_text = [w for w in text.split() if w.lower() not in self.stopwords]
        return " ".join(filtered_text)

class ResultRegressor(torch.nn.Module):
    def __init__(self):
        super(ResultRegressor, self).__init__()

        total_embedding_dim = 160

        self.bc1 = torch.nn.Linear(total_embedding_dim, 500)
        self.bc2 = torch.nn.Linear(768, 50)
        self.bc3 = torch.nn.Linear(768, 50)
        self.bc4 = torch.nn.Linear(768, 300)

        self.fc1 = torch.nn.Linear(900, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc4 = torch.nn.Linear(64, 1)

        self.layernorm1 = torch.nn.LayerNorm(total_embedding_dim)
        self.layernorm2 = torch.nn.LayerNorm(512)
        self.layernorm3 = torch.nn.LayerNorm(256)
        self.layernorm4 = torch.nn.LayerNorm(64)

        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.GELU()

    def forward(self, batch):
        numerical_inputs = self.layernorm1(batch['numerical_inputs'])
        major = self.dropout(self.activation(self.bc2(batch['major_embedding'])))
        residence = self.dropout(self.activation(self.bc3(batch['residence_embedding'])))
        ecs = self.dropout(self.activation(self.bc4(batch['ecs_embedding'])))

        numerical_inputs = self.bc1(numerical_inputs)
        numerical_inputs = self.activation(numerical_inputs)
        numerical_inputs = self.dropout(numerical_inputs)

        combined_input = torch.cat([major, residence, ecs, numerical_inputs], dim=1)

        x = self.fc1(combined_input)
        x = self.layernorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.layernorm3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.layernorm4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return torch.sigmoid(x.squeeze(dim=1))