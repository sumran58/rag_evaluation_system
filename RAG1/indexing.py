from dotenv import load_dotenv
from file_reader import extract_paragraph
from langchain_core.vectorstores import InMemoryVectorStore 
#InMemoryVectorStore is a vector database provided by LangChain that stores text embeddings temporarily in your computer's RAM (memory) while your Python program is running.
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Load the Hugging Face embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create an in-memory vector store
vector_store = InMemoryVectorStore(embedding)

# Extract text from the PDF
chunks = extract_paragraph("../pdfs/Penguins_ACL.pdf")

# Generate embeddings and store the text with its vectors
vector_store.add_texts(texts=chunks)

print(f"Indexed {len(chunks)} chunks")
