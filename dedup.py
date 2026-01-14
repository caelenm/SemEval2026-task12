import json
import sys
from pathlib import Path

def deduplicate_scores(input_path):
    # Load raw data
    path = Path(input_path)
    if not path.exists():
        print(f"Error: {path} not found.")
        return

    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Flatten list (handle nested lists from previous append errors)
    flattened_data = []
    
    def flatten(item):
        if isinstance(item, list):
            for sub_item in item:
                flatten(sub_item)
        else:
            flattened_data.append(item)

    flatten(raw_data)

    # Deduplicate using dictionary (last occurrence wins)
    unique_scores = {
        item['outfile_name']: item 
        for item in flattened_data 
        if 'outfile_name' in item
    }

    # Convert back to list and sort 
    cleaned_data = list(unique_scores.values())
    
    # Save back to file
    with path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4)
    
    print(f"Cleaned {len(raw_data)} raw entries (including nested) into {len(cleaned_data)} unique scores.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dedup.py <json_file>")
    else:
        deduplicate_scores(sys.argv[1])
