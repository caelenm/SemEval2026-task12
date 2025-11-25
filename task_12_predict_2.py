import json
import sys
import os
import time
import subprocess

def manage_dependencies():
    """
    Uninstalls heavy local ML libraries and installs the Google Generative AI SDK.
    """
    print("--- Starting Dependency Management ---")

    # 1. Uninstall heavy libraries (torch, transformers)
    # The '-y' flag automatically says "yes" to confirmation prompts.
    print("\n[1/2] Uninstalling 'torch' and 'transformers'...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y", "torch", "transformers"
        ])
        print("Successfully uninstalled heavy libraries.")
    except subprocess.CalledProcessError:
        print("Warning: Could not uninstall packages. They might not be installed.")

    # 2. Install Google Generative AI
    print("\n[2/2] Installing 'google-generativeai'...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "google-generativeai"
        ])
        print("Successfully installed 'google-generativeai'.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package: {e}")
        sys.exit(1)

    print("\n--- Setup Complete ---")
    print("You can now run the optimized script.")

manage_dependencies()

import google.generativeai as genai
from google.api_core import retry

# --- LLM Setup ---

# 1. Get your FREE API key here: https://aistudio.google.com/app/apikey
# You can hardcode it below for testing, but using an env variable is safer.
# API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
API_KEY = 'AbC123'
genai.configure(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

try:
    model = genai.GenerativeModel(MODEL_NAME)
    print(f"API Model '{MODEL_NAME}' initialized.")
except Exception as e:
    print(f"Error setting up Google API: {e}")
    model = None


def generate_llm_prediction(question_data, topic_docs, model_instance):
    """
    Constructs the prompt, calls the API, and parses the prediction.
    """
    if model_instance is None:
        return 'A'

    # 1. Prepare the context
    docs_content = "\n".join([doc['content'] for doc in topic_docs['docs']])

    # Construct the prompt
    prompt = f"""
    You are an expert analyst. Analyze the following documents and answer the question.

    Contextual Documents:
    ---
    {docs_content}
    ---

    Question: "{question_data['target_event']}"

    Options:
    A: {question_data['option_A']}
    B: {question_data['option_B']}
    C: {question_data['option_C']}
    D: {question_data['option_D']}

    Based on the documents, select the letter (A, B, C, or D) that best explains the cause.
    Output strictly ONLY the single letter. Do not output any other text.
    """

    # 2. Call the API
    try:
        # Generate content with a small delay to respect Free Tier Rate Limits (approx 15 RPM)
        # If you are on a paid tier, you can remove this sleep.
        time.sleep(4)

        response = model_instance.generate_content(prompt)
        raw_output = response.text.strip().upper()

        # 3. Parse the prediction
        # Since we instructed the model to be strict, we just look for the first valid char
        for char in raw_output:
            if char in ['A', 'B', 'C', 'D']:
                return char

        print(f"Warning: API output '{raw_output}' unclear. Defaulting to 'A'.")
        return 'A'

    except Exception as e:
        print(f"Error during API call: {e}. Defaulting to 'A'.")
        # If you hit a 429 (Rate Limit), you might want to increase the time.sleep() above
        return 'A'


# --- Main Logic ---

def iterate_over_dataset(questions, docs, model_name=""):
    preds = {}
    total_questions = len(questions)

    print(f"Starting processing of {total_questions} questions...")

    for i, question_line in enumerate(questions):
        try:
            question = json.loads(question_line)
        except json.JSONDecodeError as e:
            print(f"Skipping line {i} due to JSON decoding error: {e}")
            continue

        topic_id = question['topic_id']
        uuid = question['uuid']

        topic_docs = ''
        for curr_topic in docs:
            if (curr_topic['topic_id'] == topic_id):
                topic_docs = curr_topic
                break

        if topic_docs == '':
            print(f"Did not find docs for topic_id {topic_id}")
            preds[uuid] = 'A'
            continue

        # --- Prediction Logic ---
        if (model_name == "llm_predictor" and model is not None):
            preds[uuid] = generate_llm_prediction(question, topic_docs, model)
        else:
            preds[uuid] = 'A'

        # Simple progress indicator
        print(f"[{i + 1}/{total_questions}] UUID: {uuid} | Pred: {preds[uuid]}")

    return preds


# --- File Loading ---

dataset_file = "questions.jsonl"
docs_file = "docs.json"
output_file = "out.json"

# Check if files exist
if not os.path.exists(dataset_file) or not os.path.exists(docs_file):
    print(f"Error: Ensure '{dataset_file}' and '{docs_file}' are in the directory.")
    sys.exit(1)

try:
    with open(dataset_file, encoding='utf-8') as f:
        questions_lines = f.readlines()
    with open(docs_file, 'r', encoding='utf-8') as f:
        docs = json.load(f)

except Exception as e:
    print(f"Error reading files: {e}")
    sys.exit(1)

# *** RUN ***
predictions = iterate_over_dataset(questions_lines, docs, model_name="llm_predictor")

print(f"Writing {len(predictions)} predictions to {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, indent=4)

print("Script finished.")