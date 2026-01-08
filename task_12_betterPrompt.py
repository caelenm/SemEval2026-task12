import json
import sys
import os
import re
import time
import subprocess
from pathlib import Path
from model import Model
from getDocsLocal import getRelevantDocs
from chunker2 import chunker
import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

#===============
#Important variables
#==================
COT = 1 #binary variable
k=3 # number of similar documents
size = "large" #size of qwen embedding model, small =0.6b, large = 8b
local = 1 #binary check if using getDocs or getDocsLocal

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

#print("would you like to set up dependencies") Y/N




# LLM setup
models = ["openai/gpt-5-nano","gemini-2.5-flash-lite-preview-09-2025-thinking", "x-ai/grok-4.1-fast"]
seen_topics_dict = {}
def generate_llm_prediction(question_data, topic_docs, model):
    """
    Constructs the prompt, calls the API, and parses the prediction.
    """
    if model is None:
        return 'A'

    # 1. Prepare the context
    docs_content = "\n".join([doc['content'] for doc in topic_docs['docs']])
    seen_topics_dict = {}
    chunkedDocs = []
    if local == 1:
        rel_docs, seen_topics_dict = getRelevantDocs('docs.json',question_data['target_event'],question_data['topic_id'], seen_topics_dict, k,"small")
    else: 
        rel_docs, seen_topics_dict = getRelevantDocs('docs.json',question_data['target_event'],question_data['topic_id'], seen_topics_dict, k)
    if rel_docs is not None and len(rel_docs) > 0:
        for i in range(len(rel_docs)):
            RelevantDocs = rel_docs[i]['title'] + "\n\n" + rel_docs[i]['content'] #include title and content, both useful
            chunkedDoc = chunker(question_data['target_event'],RelevantDocs,12)
            chunkedDocs.extend(chunkedDoc)
    else:
        RelevantDocs = ["",""]
    # if COT == 1:
    #     thinkingPrompt = "Let's think step by step."         
    # prompt from https://arxiv.org/abs/2205.11916, not feasible to edit each question like https://arxiv.org/abs/2201.11903
    # else:
    #     thinkingPrompt = None
    print(RelevantDocs)
    print(type(RelevantDocs))
     # take most useful doc, get most useful lines
    # Construct the prompt
    prompt = f"""
    You are an expert cause-effect analyst. Analyze the following documents and answer the question.

    ---
    Cause
    ---
    Question: "{question_data['target_event']}"
    ---
    Effect
    ---
    Options:
    A: {question_data['option_A']}
    B: {question_data['option_B']}
    C: {question_data['option_C']}
    D: {question_data['option_D']}
    ---
    Relevant Documents: "
    {"\n".join(chunkedDocs)}
    "
     ---
    
    Evaluation: Based on the information provided, select ANY OF (A, B, C, or D) that best explains the cause. you will get 1 point for an exactly correct guess (perfect match, e.g. guess A,C = answer A,C), 0.5 points for a partially correct guess
    (one or more matching letter. e.g. guess A , but correct was A, B, C), and zero points for no match. 
     ---
    Instructions:
    1. Restate the target event in your own words in one sentence.

    2. For each option (A, B, C, D), do:
    - Briefly explain how this option could cause the target event.
    - Cite at least two specific facts or sentences from the Relevant Documents that support this option, if any.
    - Point out any contradictions or missing links with the Relevant Documents.
    - Give this option a plausibility score from 0 to 1.

    3. Compare the four options and:
    - Identify which option(s) provide the most direct, well-supported explanation with the fewest extra assumptions.
    - If multiple options are plausible and not mutually exclusive, you may select more than one.

    4. Think again:
    - For the best option(s), briefly check if there is a strong reason they might be wrong given the documents.
    - If you find a serious problem, adjust your choice.

Requirement: After completing your reasoning, output strictly ONLY the letters you predict, separated by commas, with no spaces and no other text.

    """     #generate-evaluate loop https://arxiv.org/html/2509.24096v1
    print(prompt)
        # 2. Call the API
    try:
        #time.sleep(4)
        response = model.generate_content(prompt)
        
        # Parse the JSON response
        try:
            response_json = json.loads(response.text)
            # Extract the assistant's message content
            raw_output = response_json['choices'][0]['message']['content'].strip().upper()
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing API response: {e}")
            return 'A, B, C, D'


       
        print(raw_output)
        matches = re.findall(r'\b[ABCD]\b', raw_output)
        
        # Remove duplicates and sort 
        unique_matches = sorted(list(set(matches)))
        
        if unique_matches:
            return ", ".join(unique_matches)


        print(f"Warning: API output '{raw_output}' unclear. Defaulting to All.")
        return 'A, B, C, D'


    except Exception as e:
        print(f"Error during API call: {e}. Defaulting to All.")
        return 'A, B, C, D'



# --- Main Logic ---

def iterate_over_dataset(questions, docs, model):
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
        if  model is not None:
            preds[uuid] = generate_llm_prediction(question, topic_docs, model)
        else:
            preds[uuid] = 'A'

        # Simple progress indicator
        print(f"[{i + 1}/{total_questions}] UUID: {uuid} | Pred: {preds[uuid]} | Answer: {question['golden_answer']}")

    return preds


# --- File Loading ---

dataset_file = "questions.jsonl"
docs_file = "docs.json"


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

outfiles = Path("outfiles")
outfiles.mkdir(exist_ok=True)

task = 0
tasks = len(models)

for i in models:
    model_name = i
    model = Model(model_name, "chat")
    
    if '/' in model_name:
        m = model_name.split('/')[1]
    else: 
        m = i
        
    filename = f"out_{formatted_time}_{m}_{k}sim_{COT}cot_{size}.json"
    final_path = outfiles / filename  
    
    print("output file: ", final_path)
    print(f"Task {task} of {tasks} running...")
    
    predictions = iterate_over_dataset(questions_lines, docs, model)
    
    print(f"Writing {len(predictions)} predictions to {final_path}")
    
    with final_path.open('w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
        
    task += 1

print("Script finished.")
