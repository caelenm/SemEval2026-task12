
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

def getRelevantDocs(docs_file, question, topic, k):

    """where  docs_file is the collection of documents sorted by topic
        question is the event the model must reason about
        topic is the topic number
        k is the number of relevant documents from docs_file to be returned
    """
    # open documents

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
    
    # create embedding model object
    embedding_model = "Qwen/Qwen3-Embedding-8B" #selected for cost and performance, top 3 on per-task leaderboard here: https://huggingface.co/spaces/mteb/leaderboard
    model = Model(embedding_model, "embedding")

    # embed question
    question_embed = model.generate_content(question)
    try:
        question_embed = json.loads(question_embed.text)
        question_embedding = question_embed['data'][0]['embedding']
            
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing API response: {e}")
        return 'NULL'


    #embed title
    print("Starting embeddings...")
    title_embeddings = []
    for t in titles:
        title_embed = model.generate_content(t)
        

        try:
            title_embed = json.loads(title_embed.text)
            title_embeddings.append(title_embed['data'][0]['embedding'])
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing API response: {e}")
            return 'NULL'
    print("embeddings done")
    print("Calculating similarity scores...")


    
    # compare question and title(s)
    scores = [cosine_similarity([question_embedding], [t])[0][0] for t in title_embeddings]
    k_similar = [doc for doc, _ in sorted(zip(matching_topic['docs'], scores), key=lambda x: x[1], reverse=True)[:k]]


    print(k_similar)
    return k_similar
