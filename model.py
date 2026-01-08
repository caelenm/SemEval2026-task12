import subprocess
import json

# --- CONFIGURATION ---
APIKEY = "" # <--- PUT YOUR KEY HERE
# ---------------------

def call_paid_api(model_name, type, prompt):
    #  Call curl with the safe string
    if type == "embedding":
        url = "https://nano-gpt.com/api/v1/embeddings"  # correct endpoint [web:3]
        payload = {"model": model_name, "input": prompt}  # correct schema [web:3][web:4]
    else:
        url = "https://nano-gpt.com/api/v1/chat/completions"  # chat endpoint [web:6]
        payload = {"model": model_name, "messages": [{"role": "user", "content": prompt}]}  # chat schema [web:6]

    payload_str = json.dumps(payload)

    command = [
        "curl", "-sS", "-X", "POST", url,
        "-H", f"Authorization: Bearer {APIKEY}",
        "-H", "Content-Type: application/json",
        "-d", payload_str
    ]
     
    # Run safely with timeout
    try:
        result = subprocess.run(
            command, 
            check=False, 
            capture_output=True, 
            text=True,
            timeout=90  # 90 seconds timeout
        )
    except subprocess.TimeoutExpired:
        print("Error: API Request Timed Out")
        # Return a dummy object with error JSON so the main script handles it gracefully
        return type('obj', (object,), {'text': '{"error": "timeout"}'})()

    # Debug print (only if it fails)
    if result.returncode != 0:
        print(f"Curl Error: {result.stderr}")

    # Return response object
    class Response:
        def __init__(self, text):
            self.text = text
            
    return Response(result.stdout)


class Model:
    def __init__(self, model_name, type):
        self.model_name = model_name
        self.type = type

    def generate_content(self, input_text):
        return call_paid_api(self.model_name,self.type, input_text)
