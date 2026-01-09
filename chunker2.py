from collections import OrderedDict
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from callOllamaEfficiently import callOllama
from sklearn.metrics.pairwise import cosine_similarity


nltk.download("punkt")
nltk.download("punkt_tab")


def chunker(question: str, document: str, k: int, size: str = "small") -> List[str]:
    """
    this function takes in a question (short) and document (long), both strings
    the goal is to return k most relevant sentences of the document (chunks) in the form of a list of strings
    by using callOllama(size, text) . cosine similarity used for comparison
    """
    try:
        q_embedding = callOllama(size, question)
        if not q_embedding:
            print(f"Warning: Empty embedding for question '{question[:30]}...'")
            return []

        sentences = [s.strip() for s in sent_tokenize(document) if s.strip()]

        sims: List[Tuple[float, str]] = []
        for sent in sentences:
            try:
                # Sanitize for CLI flags
                clean_sent = sent
                if sent.startswith("-"):
                    clean_sent = " " + sent  # Prepend space to avoid flag confusion
                
                s_embedding = callOllama(size, clean_sent)
                
                # Check for failed embedding
                if not s_embedding:
                    continue

                # Compute similarity (wrap in list for 2D array expected by sklearn)
                sim = cosine_similarity([s_embedding], [q_embedding])[0][0]
                sims.append((sim, sent))
            
            except Exception as inner_e:
                print(f"Error processing sentence '{sent[:30]}...': {inner_e}")
                continue

        sims.sort(key=lambda x: x[0], reverse=True)
        top_k = sims[:k]
        return [s for _, s in top_k]

    except Exception as e:
        print(f"Error in chunker: {e}")
        return []
