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
                    ecs += self.vectorize_text(ec, 50)
                ecs += [0 for _ in range(500 - len(ecs))]

                awards = []
                for award in post['awards'][0:5]:
                    awards += self.vectorize_text(award, 15)
                awards += [0 for _ in range(75 - len(awards))]

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

                post['numeric'][4] = post['numeric'][4] ** (1/2)
                numerical_inputs = residence + post['numeric'] + [int(college['in_state']), 
                                                                  int(college_information['Control of institution'] == 'Public'),
                                                                  float(college_information['Applicants total']/college_information['Admissions total']),
                                                                  float(college_information['SAT Critical Reading 75th percentile score']),
                                                                  float(college_information['SAT Math 75th percentile score']),
                                                                  int(college['round']),
                                                                  major_frequency] + college_information['combined']
                
                self.data.append({
                    'activities_inputs': torch.tensor(activities_inputs, dtype=torch.float32).detach(),
                    'numeric_inputs': torch.tensor(numerical_inputs, dtype=torch.float32).detach().nan_to_num(500),
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
        self.fcA1 = torch.nn.Linear(575, 256)
        self.fcA2 = torch.nn.Linear(256, 128)
        self.fcA3 = torch.nn.Linear(128, 64)
        self.fcA4 = torch.nn.Linear(64, 32)

        self.fcB1 = torch.nn.Linear(55, 256)
        self.fcB2 = torch.nn.Linear(256, 128)
        self.fcB3 = torch.nn.Linear(128, 64)
        self.fcB4 = torch.nn.Linear(64, 32)

        self.fcC1 = torch.nn.Linear(64, 64)
        self.fcC2 = torch.nn.Linear(64, 32)
        self.fcC3 = torch.nn.Linear(32, 1)

        self.layernormA = torch.nn.BatchNorm1d(575)
        self.layernormB = torch.nn.BatchNorm1d(55)

        self.dropout = torch.nn.Dropout(0.5)
        self.activation = torch.nn.GELU()

    def forward(self, batch):
        a = self.dropout(self.activation(self.fcA1(self.layernormA(batch['activities_inputs']))))
        a = self.dropout(self.activation(self.fcA2(a)))
        a = self.dropout(self.activation(self.fcA3(a)))
        a = self.dropout(self.activation(self.fcA4(a)))

        b = self.dropout(self.activation(self.fcB1(self.layernormB(batch['numeric_inputs']))))
        b = self.dropout(self.activation(self.fcB2(b)))
        b = self.dropout(self.activation(self.fcB3(b)))
        b = self.dropout(self.activation(self.fcB4(b)))

        c = self.dropout(self.activation(self.fcC1(torch.concat([a, b], dim=1))))
        c = self.dropout(self.activation(self.fcC2(c)))
        c = self.fcC3(c)
        return c.squeeze(dim=1)