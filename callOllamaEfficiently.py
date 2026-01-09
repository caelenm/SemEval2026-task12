import requests
import json

# Maintain a global session for connection pooling
session = requests.Session()

def callOllama(size, text):
    """
    Efficiently calls Ollama via HTTP API
    """
    if size == "small":
        model = "qwen3-embedding:0.6b"
    else:
        model = "qwen3-embedding:8b"

    url = "http://localhost:11434/api/embeddings"
    
    payload = {
        "model": model,
        "prompt": text
    }

    try:
        response = session.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        embedding = data.get('embedding', [])
        
        if not embedding:
            # Fallback check if API returns 'response' for some reason
            # though /api/embeddings standard is 'embedding'
            print(f"Warning: No embedding found for text: {text[:30]}...")
            return []
            
        return embedding

    except requests.exceptions.RequestException as e:
        print(f"Ollama Connection Error: {e}")
        print("Ensure 'ollama serve' is running.")
        return []
    except json.JSONDecodeError:
        print("Error: Could not decode Ollama response")
        return []
