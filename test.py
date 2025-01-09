from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os


load_dotenv("key.env")
openai_api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

index_name = "smartpreppeprr"
pc = Pinecone(api_key="pcsk_4oqbPN_EqGQuGKCHfdVH4osGkQk48wHcQ1kzx77QfM7fm7TbA6bjwfxwU9FAABMeLNWhDM")  

if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
pinecone_index = pc.Index(index_name)


embeddings = OpenAIEmbeddings()
text_key = "text"
vectorstore = PineconeVectorStore(pinecone_index, embeddings, text_key)


# Function to process and store a single PDF
def process_pdf_and_store(filepath):
    try:
        # Load the PDF
        loader = PyPDFLoader(filepath)
        documents = loader.load()

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Prepare metadata and store the chunks in Pinecone
        for i, doc in enumerate(texts):
            doc.metadata["source"] = f"{os.path.basename(filepath)}_chunk_{i}"

        # Add documents to vector store
        vectorstore.add_documents(texts)

        print(f"Successfully stored {filepath} in Pinecone.")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


pdf_files = [
    r"C:\Users\ASUS\Desktop\chatbot\India2022.pdf"
]

# Process and store PDF
for pdf_file in pdf_files:
    if os.path.exists(pdf_file):
        process_pdf_and_store(pdf_file)
    else:
        print(f"File not found: {pdf_file}")

print("All PDFs have been processed.")
