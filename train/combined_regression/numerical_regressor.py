import torch
from transformers import DistilBertTokenizer

class NumericCollegeResultsDataset(torch.utils.data.Dataset):
    def vectorize_text(self, text, max_length=10):
        return self.tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True)['input_ids']

    def __init__(self, data, colleges_list):
        self.data = []
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        for post_id, post in data.items():
            for college in post['results']:
                major = self.vectorize_text(post['major'], 10)
                residence = self.vectorize_text(post['residence'], 5)
                
                ecs = []
                for ec in post['ecs'][0:10]:
                    ecs += self.vectorize_text(ec, 50)
                ecs += [0 for _ in range(500 - len(ecs))]

                awards = []
                for award in post['awards'][0:5]:
                    awards += self.vectorize_text(award, 15)
                awards += [0 for _ in range(75 - len(awards))]
                
                numerical_inputs = major + residence + ecs + awards + post['numeric'] + [int(college['in_state']), 
                                                      int(college['round']), 
                                                      colleges_list.index(college['school_name']),]
                
                self.data.append({
                    'numeric_inputs': torch.tensor(numerical_inputs, dtype=torch.float32).detach(),
                    'target': torch.tensor(college['accepted'], dtype=torch.float32).detach()
                })
            print(f"Loaded {post_id}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
class NumericResultRegressor(torch.nn.Module):
    def __init__(self):
        super(NumericResultRegressor, self).__init__()
        self.fc1 = torch.nn.Linear(624, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 64)
        self.fc5 = torch.nn.Linear(64, 1)

        self.layernorm1 = torch.nn.LayerNorm(1024)
        self.layernorm2 = torch.nn.LayerNorm(512)
        self.layernorm3 = torch.nn.LayerNorm(256)
        self.layernorm4 = torch.nn.LayerNorm(64)

        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.GELU()

    def forward(self, batch):
        x = self.fc1(batch['numeric_inputs'])
        x = self.layernorm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.layernorm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.layernorm3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.layernorm4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.dropout(x)
        return torch.sigmoid(x.squeeze(dim=1))