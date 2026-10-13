from indexing import vector_store
from langchain.tools import tool

last_retrieved_context=[]

@tool
def collecting_info(query:str)->str:
    """ search the pdf knowledge base and return relevent information for the users question"""

    global last_retrieved_context
    docs=vector_store.similarity_search(query,k=3)

    last_retrieved_context=[doc.page_content for doc in docs]
    return "\n\n".join(last_retrieved_context)