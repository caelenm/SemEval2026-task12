import subprocess
import json

def callOllama(size, text):
    if size == "small":
        model = "qwen3-embedding:0.6b"
    else:
        model = "qwen3-embedding:8b"

    cmd = ["ollama", "run", model,"--", text]  

    embedding = []  # Default

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout.strip())

        if isinstance(data, list):
            embedding = data
        elif isinstance(data, dict):
            embedding = data.get('embedding') or data.get('response', [])

    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode}):")
        print("Stdout:", e.stdout.strip() if e.stdout else "None")
        print("Stderr:", e.stderr.strip() if e.stderr else "None")
        print("\nTips:\n- ollama pull <model>\n- ollama serve\n- ollama list")
        return embedding  # Empty on error
    except json.JSONDecodeError:
        print("Raw output (not JSON):", result.stdout.strip())
        return embedding  # Empty on error

    # print("Embedding len:", len(embedding))  # Optional uncomment for debug
    return embedding