import json
import re

from model import Model  # Assuming your `Model` class is defined in `model.py`
"""
this is solely to confirm the api is working
"""
def test_model_api():
    # Define the model name to test
    model_name = "Meta-Llama-3-1-8B-Instruct-FP8"  
    try:
        print(f"Initializing model: {model_name}")
        model = Model(model_name, "chat")  # Initialize the model

        # Define a simple prompt for testing
        prompt = """
        You are an expert cause-effect analyst. Analyze the following question.

        ---
        Question: "What is the impact of policies on economic growth?"
        ---
        Options:
        A: Increases GDP
        B: Reduces unemployment
        C: Promotes inflation
        D: No significant impact
        ---
        Instructions:
        Select the option(s) (A, B, C, D) that best explain the cause. Return strictly the letters, separated by commas, with no spaces or additional text.
        """

        print("Sending prompt to the API...")
        response = model.generate_content(prompt)  # Call the model's API

        # Test if response is valid
        if not response:
            print("Warning: Model response is empty.")
            return

        print("\n-- Raw API Response --")
        print(response)

        # Attempt to parse JSON response (if applicable)
        try:
            response_json = json.loads(response.text)  # Assuming the response has a `.text` attribute
            raw_output = response_json['choices'][0]['message']['content'].strip().upper()
            print("\n-- Parsed Response --")
            print(raw_output)
        except json.JSONDecodeError:
            print("Error: Failed to parse JSON response.")
        except (KeyError, IndexError) as e:
            print(f"Error: Unexpected response structure. {e}")

        # If response is plain text, demonstrate regex extraction:
        matches = re.findall(r'\b[ABCD]\b', response.text if hasattr(response, 'text') else response)
        print("\n-- Extracted Options (Regex Matches) --")
        print(", ".join(sorted(set(matches))) if matches else "No valid matches found.")

    except Exception as e:
        print(f"Error during API call or processing: {e}")

if __name__ == "__main__":
    test_model_api()