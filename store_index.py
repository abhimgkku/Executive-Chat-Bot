from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import os
from pinecone import Pinecone, ServerlessSpec



load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
#PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name="exec-chat-bot"
#initialize pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
index = pc.Index(index_name) 


#Creating Embeddings for Each of The Text Chunks & storing
for i, t in zip(range(len(text_chunks)), text_chunks):
   query_result = embeddings.embed_query(t.page_content)
   index.upsert(
   vectors=[
        {
            "id": str(i),  # Convert i to a string
            "values": query_result, 
            "metadata": {"text":str(text_chunks[i].page_content)} # meta data as dic
        }
    ],
    namespace="real" 
)

    