import pinecone
import os

from langchain.vectorstores import Pinecone
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# LOCAL STUFF
# Set Key Variables
from dotenv import load_dotenv
load_dotenv()
pinecone.api_key = os.getenv("PINECONE_API_KEY")
# END LOCAL STUFF

"""
Pinecone Setup
"""
# Initialize pinecone
pinecone.init(
    environment="us-central1-gcp"  # next to api key in console
)
index_name = "langchain-demo1"
"""
END Pinecone Setup
"""

# First, we need to initialize load URLs and load the text
# Opens the file and reads the URLs
with open('./urls.txt', 'r') as file:
        urls = [line.strip() for line in file]

# Loads the list of URLS and all the text (Consider using Selenium URL loader)
loader = UnstructuredURLLoader(urls=urls)
kb_data = loader.load()

# Second, split the text into chunks, using Tiktoken
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
kb_docs = text_splitter.split_documents(kb_data) #Note that split_documents works but not split_text because we're dealing with a list of documents

# Create the embeddings
embeddings = OpenAIEmbeddings()

# Store in the DB
kb_db_store = Pinecone.from_documents(kb_docs, embeddings, index_name=index_name, namespace="lsvt")