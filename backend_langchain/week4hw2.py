from pinecone import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
pc = Pinecone()
# print(pc.list_indexes())

index_name = 'llm-rag3'



index = pc.Index(index_name)
index.describe_index_stats()



# Connect to the Pinecone index
index = pc.Index(index_name)


# print("Loading documents")
# # Load documents from local disk
# loader = DirectoryLoader('../data', glob=["*.pdf", "*.txt", "*.docx"])
# documents = loader.load()

# Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# chunks = text_splitter.split_documents(documents)
# Add documents to Pinecone
print("Adding documents to Pinecone")

vectorstore = PineconeVectorStore(index, OpenAIEmbeddings(model='text-embedding-ada-002'), "text")
# vectorstore.add_documents(chunks)






# Create retriever
retriever = vectorstore.as_retriever()

print("Creating chains")
llm = ChatOpenAI()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

# Main loop
while True:
    user_input = input("> ")
    result = conversation.invoke(user_input)
    print(result["answer"])
