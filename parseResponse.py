import re

def parse_llm_response(response_json):
    """
    Parses the response from the LLM to extract the final predicted options (A, B, C, D).
    
    Args:
        response_json (dict): The raw JSON response dictionary from the API.
        
    Returns:
        str: A comma-separated string of sorted unique options (e.g., "A, C") 
             or None if no valid options are found.
    """
    try:
        
        content = response_json['choices'][0]['message']['content']
        
        if not content:
            return None

        full_text = content.strip()
        
        # check for answer, must tell llm to use this form
        if "FINAL ANSWER:" in full_text:
            answer_zone = full_text.split("FINAL ANSWER:")[-1].strip().upper()
        
        #backup
        else:
            lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            if not lines:
                return None
            answer_zone = lines[-1].upper()


        matches = re.findall(r'\b[ABCD]\b', answer_zone)
        
       #format output
        unique_matches = sorted(list(set(matches)))
        
        if unique_matches:
            return ", ".join(unique_matches)
            
        return None

    except (KeyError, IndexError, AttributeError) as e:
        print(f"Error parsing response structure in parse_llm_response: {e}")
        return None
