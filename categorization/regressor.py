import torch
from torch.utils.data import Dataset
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig

class CollegeDataset(Dataset):
    def __init__(self, data, colleges_list, stopwords):
        self.data = []
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.stopwords = stopwords
        
        for post_id, post in data.items():
            for college in post['results']:
                self.data.append({
                    'post': post,
                    'college': college,
                    'college_id': colleges_list.index(college['school_name'])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
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
        ], dtype=torch.long)

        major = self.remove_stopwords(post['major'])
        residence = self.remove_stopwords(post['residence'])
        extracurriculars = self.remove_stopwords('\n'.join(post['extracurriculars'] + post['awards']))

        major_encoded = self.tokenizer.encode_plus(major, max_length=20, padding='max_length', truncation=True)
        residence_encoded = self.tokenizer.encode_plus(residence, max_length=10, padding='max_length', truncation=True)
        extracurriculars_encoded = self.tokenizer.encode_plus(extracurriculars, max_length=512, padding='max_length', truncation=True)

        return {
            'numerical_inputs': numerical_inputs,
            'major_ids': torch.tensor(major_encoded['input_ids'], dtype=torch.long),
            'major_mask': torch.tensor(major_encoded['attention_mask'], dtype=torch.long),
            'residence_ids': torch.tensor(residence_encoded['input_ids'], dtype=torch.long),
            'residence_mask': torch.tensor(residence_encoded['attention_mask'], dtype=torch.long),
            'extracurriculars_ids': torch.tensor(extracurriculars_encoded['input_ids'], dtype=torch.long),
            'extracurriculars_mask': torch.tensor(extracurriculars_encoded['attention_mask'], dtype=torch.long),
            'target': torch.tensor(college['accepted'], dtype=torch.float32)
        }

    def remove_stopwords(self, text):
        filtered_text = [w for w in text.split() if w.lower() not in self.stopwords]
        return " ".join(filtered_text)

class ResultRegressor(torch.nn.Module):
    def __init__(self):
        super(ResultRegressor, self).__init__()
        self.text1 = DistilBertModel.from_pretrained("distilbert-base-uncased", 
                                                     ignore_mismatched_sizes=True,
                                                     config=DistilBertConfig(max_position_embeddings=20))
        self.text2 = DistilBertModel.from_pretrained("distilbert-base-uncased", 
                                                     ignore_mismatched_sizes=True,
                                                     config=DistilBertConfig(max_position_embeddings=10))
        self.text3 = DistilBertModel.from_pretrained("distilbert-base-uncased", 
                                                     ignore_mismatched_sizes=True,
                                                     config=DistilBertConfig(max_position_embeddings=512))
        
        self.pc1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.pc2 = torch.nn.Linear(768, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.pc3 = torch.nn.Linear(768, 128)
        self.bn3 = torch.nn.BatchNorm1d(128)
        
        self.fc1 = torch.nn.Linear(266, 32)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.fc2 = torch.nn.Linear(32, 1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, batch):
        major_pooler = self.text1(batch['major_ids'], batch['major_mask'])[0][:,0]
        residence_pooler = self.text2(batch['residence_ids'], batch['residence_mask'])[0][:,0]
        extracurriculars_pooler = self.text3(batch['extracurriculars_ids'], batch['extracurriculars_mask'])[0][:,0]
        
        numerical_inputs = torch.cat([
            batch['numerical_inputs'],
            self.relu(self.bn1(self.pc1(major_pooler))),
            self.relu(self.bn2(self.pc2(residence_pooler))),
            self.relu(self.bn3(self.pc3(extracurriculars_pooler)))
        ], dim=1)

        x = self.bn4(self.fc1(numerical_inputs))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.sigmoid(x).squeeze()