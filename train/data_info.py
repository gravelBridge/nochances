import json
from collections import Counter


def analyze_json_structure(file_path):
    field_positions = {}
    accept_rate_values = Counter()
    total_entries = 0

    with open(file_path, "r") as file:
        for line_number, line in enumerate(file, 1):
            try:
                entry = json.loads(line)
                total_entries += 1

                if "skip" in entry and entry["skip"] == True:
                    continue

                flat_entry = []
                for category, values in entry.items():
                    for key, value in values.items():
                        flat_entry.append(value)
                        if key not in field_positions:
                            field_positions[key] = len(flat_entry) - 1

                if "accept_rate" in field_positions:
                    accept_rate = flat_entry[field_positions["accept_rate"]]
                    if isinstance(accept_rate, str) and accept_rate.isdigit():
                        accept_rate = int(accept_rate)
                    accept_rate_values[accept_rate] += 1

            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {line_number}")

    print(f"Total entries processed: {total_entries}")
    print("\nField positions in flattened entry:")
    for field, position in sorted(field_positions.items(), key=lambda x: x[1]):
        print(f"{field}: index {position}")

    print("\nAccept Rate Distribution:")
    for rate, count in sorted(accept_rate_values.items()):
        print(f"Category {rate}: {count} samples")


if __name__ == "__main__":
    file_path = "categorization/categorized.json"
    analyze_json_structure(file_path)
