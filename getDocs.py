
"""
 This code intends to take the list of documents, the topic number, and the provided question,
  and return the k most relevant documents

 """
import json
from sklearn.metrics.pairwise import cosine_similarity
from model import Model
from collections import OrderedDict
import sys
from model import Model
import time

def getRelevantDocs(docs_data, question, topic, seen_topics_dict, k):

    """where  docs_file is the collection of documents sorted by topic
        question is the event the model must reason about
        topic is the topic number
        k is the number of relevant documents from docs_file to be returned
    """
    # open documents

    if k==0:
        k_similar = None
        seen_topics_dict = None
        return k_similar, seen_topics_dict

    # try:
    #     with open(docs_file, 'r') as file:
    #         data = json.load(file)
    # except Exception as e:
    #     print(f"Error reading files: {e}")
    #     sys.exit(1)

    #find set that match topic  
    titles = []
    links = []
    snippets = []
    content = []
    matching_topic = next((block for block in docs_data if block["topic_id"] == topic), None)
    for doc in matching_topic['docs']:
        titles.append(doc['title'])
        links.append(doc['link'])
        snippets.append(doc['snippet'])
        content.append(doc['content'])
    
    # create embedding model object
    embedding_model = "Qwen/Qwen3-Embedding-8B" #selected for cost and performance, top 3 on per-task leaderboard here: https://huggingface.co/spaces/mteb/leaderboard
    model = Model(embedding_model, "embedding")

    # embed question
    question_embed = model.generate_content(question)
    try:
        #question_embed = json.loads(question_embed.text)
        question_embedding = question_embed['data'][0]['embedding']
            
    except (KeyError, IndexError) as e: 
        print(f"Error parsing API response: {e}")
        return 'NULL'


    #embed title        #Change this to only run once on new topic


    
    if topic not in seen_topics_dict:
        print("Starting embeddings...")
        title_embeddings = []
        
        for t in titles:
            # Retry logic for timeouts
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    title_embed = model.generate_content(t)

                    # Check if response is valid dict before accessing keys
                    if isinstance(title_embed, dict) and 'data' in title_embed:
                        title_embeddings.append(title_embed['data'][0]['embedding'])
                        success = True
                        break # Success, exit retry loop
                    else:
                        # If response is an error string or invalid
                        print(f"Warning: Invalid API response (Attempt {attempt+1}): {title_embed}")
                        time.sleep(2) # Wait before retry

                except Exception as e:
                    print(f"Error during embedding (Attempt {attempt+1}): {e}")
                    time.sleep(2) # Wait before retry
            
            if not success:
                 print(f"Skipping document '{t[:20]}...' after failing {max_retries} times.")

            
        # SAVE TO CACHE ONLY AFTER LOOP FINISHES
        if title_embeddings:
            seen_topics_dict[topic] = title_embeddings
            print("embeddings done")
        else:
            print("Warning: No embeddings were generated for this topic.")
            
    else: #(topic titles have already been embedded)
        print("topic already embedded, using cached version")
        title_embeddings = seen_topics_dict[topic]
    
    print("Calculating similarity scores...")



    
    # compare question and title(s)
    scores = [cosine_similarity([question_embedding], [t])[0][0] for t in title_embeddings]
    k_similar = [doc for doc, _ in sorted(zip(matching_topic['docs'], scores), key=lambda x: x[1], reverse=True)[:k]]


    return k_similar, seen_topics_dict
