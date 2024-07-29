import json
import random
from collections import defaultdict

def generate_synthetic_data(num_samples=1000):
    synthetic_data = []

    for _ in range(num_samples):
        sample = defaultdict(dict)

        # Generate basic_info
        sample['basic_info'] = {
            'ethnicity': random.choices([0, 1], weights=[0.3, 0.7])[0],
            'gender': random.choices([0, 1, 2], weights=[0.45, 0.45, 0.1])[0],
            'income_bracket': random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0],
            'type_school': random.choices([0, 1, 2, 3, 4], weights=[0.05, 0.1, 0.05, 0.05, 0.75])[0],
            'app_round': random.choices([0, 1], weights=[0.3, 0.7])[0],
            'gpa': random.choices([0, 1, 2, 3, 4], weights=[0.05, 0.15, 0.4, 0.3, 0.1])[0],
            'ap_ib_courses': max(0, int(random.gauss(3, 2))),
            'ap_ib_scores': random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0],
            'test_score': random.choices([0, 1, 2, 3, 4], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0],
            'location': random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0],
            'state_status': random.choices([0, 1], weights=[0.7, 0.3])[0],
            'legacy': random.choices([0, 1], weights=[0.1, 0.9])[0],
            'intended_major': random.randint(1, 10),
            'major_alignment': random.choices([1, 2, 3, 4, 5], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0],
            'first_gen': random.choices([0, 1], weights=[0.7, 0.3])[0],
            'languages': max(1, int(random.gauss(1.5, 0.5))),
            'special_talents': random.choices([0, 1, 2, 3, 4], weights=[0.6, 0.2, 0.1, 0.07, 0.03])[0],
            'hooks': random.choices([0, 1, 2, 3], weights=[0.7, 0.2, 0.07, 0.03])[0],
            'accept_rate': random.choices([2, 3], weights=[0.3, 0.7])[0],
        }

        # Generate ecs (unchanged)
        sample['ecs'] = {
            'nat_int': max(0, int(random.gauss(0.5, 0.5))),
            'reg': max(0, int(random.gauss(1, 1))),
            'local': max(0, int(random.gauss(2, 1.5))),
            'volunteering': max(0, int(random.gauss(1, 1))),
            'ent': max(0, int(random.gauss(0.2, 0.4))),
            'intern': max(0, int(random.gauss(0.3, 0.5))),
            'add': max(0, int(random.gauss(1, 1))),
            'res': max(0, int(random.gauss(0.3, 0.5))),
            'sports': max(0, int(random.gauss(0.7, 0.8))),
            'work_exp': max(0, int(random.gauss(0.5, 0.7))),
            'leadership': max(0, int(random.gauss(1, 1))),
            'community_impact': random.choices([0, 1, 2, 3, 4], weights=[0.3, 0.3, 0.2, 0.15, 0.05])[0],
            'ec_years': max(1, min(4, int(random.gauss(2.5, 1)))),
        }

        # Generate awards (unchanged)
        sample['awards'] = {
            'int': max(0, int(random.gauss(0.1, 0.3))),
            'nat': max(0, int(random.gauss(0.3, 0.5))),
            'state': max(0, int(random.gauss(0.5, 0.7))),
            'local': max(0, int(random.gauss(1, 1))),
            'other': max(0, int(random.gauss(1.5, 1))),
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
    synthetic_data = generate_synthetic_data(1000)
    append_to_json('categorization/categorized.json', synthetic_data)
    print("1000 synthetic data points have been appended to categorized.json")