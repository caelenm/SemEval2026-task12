# main.py
import subprocess
import sys
from dedup import deduplicate_scores

#ensure you run $ ollama serve

def run_script(path):
    result = subprocess.run([sys.executable, path], check=False)
    if result.returncode != 0:
        raise SystemExit(f"{path} exited with code {result.returncode}")

def main():
    run_script("task_12_betterPrompt.py")
    run_script("eval.py")
    deduplicate_scores("/var/home/user/Code/SemEval2026-task12/semeval2026-task12-dataset/sample_data/scores.json")


if __name__ == "__main__":
    main()
