from transformers import DistilBertModel, DistilBertTokenizer
import torch

class CombinedCollegeResultsDataset(torch.utils.data.Dataset):
    def __init__(self, data, colleges_list, stopwords):
        self.data = []
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.stopwords = stopwords
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        for post_id, post in data.items():
            if post['results']:
                major = self.remove_stopwords(post['major'])
                residence = self.remove_stopwords(post['residence'])
                extracurriculars = self.remove_stopwords('\n'.join([ec[:50] for ec in post['extracurriculars']]))
                awards = self.remove_stopwords('\n'.join([award[:15] for award in post['awards']]))

                major_embedding = self.embed_text(major, 10)
                residence_embedding = self.embed_text(residence, 5)
                extracurriculars_embedding = self.embed_text(extracurriculars, 500)
                awards_embedding = self.embed_text(awards, 75)

            for college in post['results']:
                numerical_inputs = post['numerical'] + [int(college['in_state']), 
                                    int(college['round']), 
                                    colleges_list.index(college['school_name']),]

                self.data.append({
                    'numerical_inputs': torch.tensor(numerical_inputs, dtype=torch.float32).detach(),
                    'major_embedding': major_embedding.detach(),
                    'residence_embedding': residence_embedding.detach(),
                    'extracurriculars_embedding': extracurriculars_embedding.detach(),
                    'awards_embedding': awards_embedding.detach(),
                    'target': torch.tensor(college['accepted'], dtype=torch.float32).detach()
                })
            print(f"Loaded {post_id}")

    def __len__(self):
        return len(self.data)
    
    def embed_text(self, text, max_length):
        inputs = self.tokenizer.encode_plus(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        return torch.tensor(self.model(**inputs)['last_hidden_state'][0][0], dtype=torch.float32)

    def __getitem__(self, i):
        return self.data[i]

    def remove_stopwords(self, text):
        filtered_text = [w for w in text.split() if w.lower() not in self.stopwords]
        return " ".join(filtered_text)

class CombinedResultRegressor(torch.nn.Module):
    def __init__(self):
        super(CombinedResultRegressor, self).__init__()
        self.bc1 = torch.nn.Linear(34, 50)
        self.bc2 = torch.nn.Linear(768, 20)
        self.bc3 = torch.nn.Linear(768, 10)
        self.bc4 = torch.nn.Linear(768, 500)
        self.bc5 = torch.nn.Linear(768, 150)

        self.fc1 = torch.nn.Linear(730, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc4 = torch.nn.Linear(64, 1)

        self.layernorm0 = torch.nn.BatchNorm1d(50)
        self.layernorm1 = torch.nn.BatchNorm1d(730)
        self.layernorm2 = torch.nn.BatchNorm1d(1024)
        self.layernorm3 = torch.nn.BatchNorm1d(256)
        self.layernorm4 = torch.nn.BatchNorm1d(64)

        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.GELU()

    def forward(self, batch):
        numerical_inputs = self.layernorm0(self.dropout(self.activation(self.bc1(batch['numerical_inputs']))))
        
        major = self.dropout(self.activation(self.bc2(batch['major_embedding'])))
        residence = self.dropout(self.activation(self.bc3(batch['residence_embedding'])))
        ecs = self.dropout(self.activation(self.bc4(batch['extracurriculars_embedding'])))
        awards = self.dropout(self.activation(self.bc5(batch['awards_embedding'])))

        combined_input = self.layernorm1(torch.cat([numerical_inputs, major, residence, ecs, awards], dim=1))

        x = self.dropout(self.activation(self.layernorm2(self.fc1(combined_input))))
        x = self.dropout(self.activation(self.layernorm3(self.fc2(x))))
        x = self.dropout(self.activation(self.layernorm4(self.fc3(x))))
        x = self.fc4(x)

        return torch.sigmoid(x.squeeze(dim=1))