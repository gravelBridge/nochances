import torch
from transformers import DistilBertTokenizer
import difflib

mappings = {
    0: ['engineering', 'applied'],
    1: ['computer science', ' cs', 'eecs', 'information technology', 'comp sci'],
    2: ['business', 'marketing', 'finance'],
    3: ['economics','political science', 'pol sci', 'econ'],
    4: ['biology', 'bio', 'neurosci'],
    5: ['math', 'data', 'stat'],
    6: ['physics', 'chemistry', 'chem', 'astro', 'science', 'environ'],
    7: ['education'],
    8: ['communications', 'journalism'],
    9: ['history', 'humanities', 'psych', 'english', 'studies', 'philosophy'],
    10: ['art', 'architecture', 'design', 'drama', 'theatre', 'music'],
    11: ['linguistics', 'spanish', 'classics', 'latin', 'french', 'language'],
}

def map_major(major):
    for key, value in mappings.items():
        for item in value:
            if item in " " + major.lower():
                return key

class TokenNumericCollegeResultsDataset(torch.utils.data.Dataset):
    def vectorize_text(self, text, max_length=10):
        return self.tokenizer.encode_plus(text, max_length=max_length, padding='max_length', truncation=True)['input_ids']

    def __init__(self, data, college_data):
        self.data = []
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        for post_id, post in data.items():
            for college in post['results']:
                residence = self.vectorize_text(post['residence'], 5)
                
                ecs = []
                for ec in post['ecs'][0:10]:
                    ecs += self.vectorize_text(ec, 15)
                ecs += [0 for _ in range(150 - len(ecs))]

                awards = []
                for award in post['awards'][0:5]:
                    awards += self.vectorize_text(award, 10)
                awards += [0 for _ in range(50 - len(awards))]

                activities_inputs = ecs + awards
                
                try:
                    closest_college = difflib.get_close_matches(college['school_name'], college_data['Name'], n=1, cutoff=0.8)[0]
                    college_information = college_data.loc[college_data['Name'] == closest_college].iloc[0]
                except IndexError:
                    try:
                        closest_college = difflib.get_close_matches(college['school_name'][:-30], college_data['Name'], n=1, cutoff=0.8)[0]
                        college_information = college_data.loc[college_data['Name'] == closest_college].iloc[0]
                    except IndexError:
                        try:
                            closest_college = difflib.get_close_matches(college['school_name'][:-20], college_data['Name'], n=1, cutoff=0.8)[0]
                            college_information = college_data.loc[college_data['Name'] == closest_college].iloc[0]
                        except IndexError:
                            print(college['school_name'], closest_college)
                            continue
                
                major_id = map_major(post['major']) or 1
                major_frequency = college_information['combined'][major_id]

                post['numeric'][4] = post['numeric'][4] ** (1/3)
                numerical_inputs = residence + post['numeric'] + [int(college['in_state']), 
                                                                  float(college_information['Applicants total']/college_information['Admissions total']),
                                                                  float(college_information['SAT Critical Reading 75th percentile score']),
                                                                  float(college_information['SAT Math 75th percentile score']),
                                                                  int(college['round']),
                                                                  major_frequency] + college_information['combined'] + activities_inputs
                
                self.data.append({
                    'inputs': torch.tensor(numerical_inputs, dtype=torch.float32).detach().nan_to_num(500),
                    'target': torch.tensor(college['accepted'], dtype=torch.float32).detach()
                })
            print(f"Loaded {post_id}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
class CombinedDelayedRegressor(torch.nn.Module):
    def __init__(self):
        super(CombinedDelayedRegressor, self).__init__()
        self.fc1 = torch.nn.Linear(250, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.Linear(128, 64)
        self.fc6 = torch.nn.Linear(64, 32)
        self.fc7 = torch.nn.Linear(32, 16)
        self.fc8 = torch.nn.Linear(16, 1)

        self.batchnorm = torch.nn.BatchNorm1d(250)

        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.GELU()

    def forward(self, input):
        x = self.dropout(self.activation(self.fc1(self.batchnorm(input))))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.dropout(self.activation(self.fc3(x)))
        x = self.dropout(self.activation(self.fc4(x)))
        x = self.dropout(self.activation(self.fc5(x)))
        x = self.dropout(self.activation(self.fc6(x)))
        x = self.dropout(self.activation(self.fc7(x)))
        x = self.fc8(x)

        return x.squeeze(dim=1)