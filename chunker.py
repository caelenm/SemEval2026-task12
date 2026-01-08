from callOllama import callOllama
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity

def chunker(question, document, k, size="small"):

    """
    this function takes in a question (short) and document (long), both strings
    the goal is to return k most relevant sentences of the document (chunks) in the form of a list of strings
    by using callOllama(size, text) . cosine similarity used for comparison
    """
    embeddings = OrderedDict({})
    sentences = {}

    q_embedding = callOllama(size, question)
    splitDoc = document.split(".")
    for i in splitDoc:
        i_embedding = callOllama(size,i)
        embeddings[i_embedding] = cosine_similarity(i_embedding, q_embedding)
        sentences[i_embedding] = i

    
    embeddings_sorted = OrderedDict(sorted.items(),key=lambda item: item[1]) #sort embeddding:sim by sim

    trucated = embeddings_sorted[:k] #keep top k

    for x in truncated:
        index = sentences[x.get()]


    return


