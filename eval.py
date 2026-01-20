import json
from pathlib import Path

# Load questions once (efficient)
questions_path = Path("/var/home/user/Code/SemEval2026-task12/semeval2026-task12-dataset/dev_data/questions.jsonl")
questions = []
try:
    with questions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
except FileNotFoundError:
    print(f"Questions file not found at {questions_path}")
    exit()

def partial_match(result, golden_answer):
    return any(r in golden_answer for r in result)

def full_match(result, golden_answer):
    return sorted(result) == sorted(golden_answer)

# Setup directory and list to hold all scores
outfiles_dir = Path("/var/home/user/Code/SemEval2026-task12/semeval2026-task12-dataset/dev_data/outfiles")
all_scores = []

if not outfiles_dir.exists():
    print("Directory not found:", outfiles_dir)
else:
    # Iterate over every .json file in the directory
    for file_path in outfiles_dir.glob("*.json"):
        print(f"Processing: {file_path.name}")
        
        try:
            with file_path.open("r", encoding="utf-8") as f:
                result_data = json.load(f)
            
            points = 0
            # Run logic for this specific file
            for i in questions:
                id = i["uuid"]
                
                # Safety check: ensure the question ID exists in this result file
                if id not in result_data:
                    print(f"Warning: ID {id} missing in {file_path.name}")
                    continue

                golden_answer = [x.strip() for x in i["golden_answer"].split(",")]
                result = [x.strip() for x in result_data[id].split(",")]

                if full_match(result, golden_answer):
                    points += 1
                elif partial_match(result, golden_answer):
                    points += 0.5
                
            # Calculate metrics
            score_entry = {
                "outfile_name": file_path.stem,
                "score": points,
                "total": len(questions),
                "percent_correct": (points / len(questions)) if questions else 0
            }
            all_scores.append(score_entry)
            print(f"Score: {points}/{len(questions)}")
            print("-" * 20)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

# Save all scores to a single file
current_scores = json.load(open("scores.json")) if Path("scores.json").exists() else []
json.dump(current_scores + [all_scores], open("scores.json", "w"), indent=4)

print("Processing complete. Results saved to scores.json")
