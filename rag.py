import os
import openai
import tiktoken
import chromadb

from langchain.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
loader = PyPDFLoader("cssa.pdf")
pdfData = loader.load()

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
splitData = text_splitter.split_documents(pdfData)

collection_name = "pega_collection"
local_directory = "pega_vect_embedding"
persist_directory = os.path.join(os.getcwd(), local_directory)

openai_key=os.environ.get('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vectDB = Chroma.from_documents(splitData,
                      embeddings,
                      collection_name=collection_name,
                      persist_directory=persist_directory
                      )
vectDB.persist()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chatQA = ConversationalRetrievalChain.from_llm(
            OpenAI(openai_api_key=openai_key,
               temperature=0, model_name="gpt-3.5-turbo"), 
            vectDB.as_retriever(), 
            memory=memory)


chat_history = []
qry = ""
while qry != 'done':
    qry = input('Question: ')
    if qry != exit:
        response = chatQA({"question": qry, "chat_history": chat_history})
        print(response["answer"])