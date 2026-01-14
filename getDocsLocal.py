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
from callOllamaEfficiently import callOllama

def getRelevantDocs(docs_file, question, topic, seen_topics_dict, k, size):

    """where  docs_file is the collection of documents sorted by topic
        question is the event the model must reason about
        topic is the topic number
        k is the number of relevant documents from docs_file to be returned
        size is whether or not we should use qwen 0.6b or 8b
    """
    # open documents

    if k==0:
        k_similar = None
        seen_topics_dict = None
        return k_similar, seen_topics_dict

    try:
        with open(docs_file, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)

    #find set that match topic  
    titles = []
    links = []
    snippets = []
    content = []
    matching_topic = next((block for block in data if block["topic_id"] == topic), None)
    for doc in matching_topic['docs']:
        titles.append(doc['title'])
        links.append(doc['link'])
        snippets.append(doc['snippet'])
        content.append(doc['content'])
    
    
    # embed question
    question_embedding = callOllama(size, question)

    #embed title        


    #not sure yet if caching works
    if topic not in seen_topics_dict:
        print("Starting embeddings...")
        title_embeddings = []
        for t in titles:
            title_embed = callOllama(size, t)
            
            try:
                title_embeddings.append(title_embed)
                seen_topics_dict[topic] = title_embeddings
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Error parsing API response: {e}")
                continue
        print("embeddings done")
    else: #(topic titles have already been embedded)
        print("topic already embedded, using cached version")
        title_embeddings = seen_topics_dict[topic]
    
    print("Calculating similarity scores...")


    
    # compare question and title(s)

    scores = [cosine_similarity([question_embedding], [t])[0][0] for t in title_embeddings]
    k_similar = [doc for doc, _ in sorted(zip(matching_topic['docs'], scores), key=lambda x: x[1], reverse=True)[:k]]


    return k_similar, seen_topics_dict
