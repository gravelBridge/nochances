import json
import random
from collections import defaultdict

def generate_synthetic_data(num_samples=1000):
    synthetic_data = []

    for _ in range(num_samples):
        sample = defaultdict(dict)

        # Generate basic_info
        sample['basic_info'] = {
            'ethnicity': random.choices([0, 1])[0],
            'gender': random.choices([0, 1, 2])[0],
            'income_bracket': random.choices([0, 1, 2, 3, 4])[0],
            'type_school': random.choices([0, 1, 2, 3, 4])[0],
            'app_round': random.choices([0, 1])[0],
            'gpa': random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.3, 0.1, 0.08, 0.02])[0],
            'ap_ib_courses': max(0, int(random.gauss(3, 2))),
            'ap_ib_scores': random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.3, 0.1, 0.08, 0.02])[0],
            'test_score': random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.3, 0.1, 0.08, 0.02])[0],
            'location': random.choices([0, 1, 2])[0],
            'state_status': random.choices([0, 1])[0],
            'legacy': random.choices([0, 1])[0],
            'intended_major': random.randint(1, 10),
            'major_alignment': random.choices([1, 2, 3, 4, 5])[0],
            'first_gen': random.choices([0, 1])[0],
            'languages': max(1, int(random.gauss(1.5, 0.5))),
            'special_talents': random.choices([0, 1, 2, 3, 4], weights=[0.5, 0.3, 0.1, 0.08, 0.02])[0],
            'hooks': random.choices([0, 1, 2, 3], weights=[0.8, 0.1, 0.08, 0.02])[0],
            'accept_rate': 3,
        }

        sample['ecs'] = {
            'nat_int': 0,
            'reg': 0,
            'local': 0,
            'volunteering': 0,
            'ent': 0,
            'intern': 0,
            'add': 0,
            'res': 0,
            'sports': 0,
            'work_exp': 0,
            'leadership': 0,
            'community_impact': 0,
            'ec_years': 0,
        }

        sample['awards'] = {
            'int': 0,
            'nat': 0,
            'state': 0,
            'local': 0,
            'other': 0,
        }

        synthetic_data.append(sample)

    return synthetic_data

def append_to_json(file_path, new_data):
    try:
        with open(file_path, 'r+') as file:
            file.seek(0, 2)  # Move to the end of the file
            for entry in new_data:
                json.dump(entry, file)
                file.write('\n')
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating a new file.")
        with open(file_path, 'w') as file:
            for entry in new_data:
                json.dump(entry, file)
                file.write('\n')

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data(100)
    append_to_json('categorization/categorized.json', synthetic_data)
    print("100 synthetic data points have been appended to categorized.json")