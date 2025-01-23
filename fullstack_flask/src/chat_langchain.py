from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone

from .models import db, ChatMessage

pc = Pinecone()

print("Connecting to Pinecone index")
index_name = 'llm-rag3'
index = pc.Index(index_name)
index.describe_index_stats()

text_field = "text"
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
vectorstore = PineconeVectorStore(index, embeddings, text_field)

print("Creating chains")
template = """You are an exceptionally knowledgeable and thorough assistant. I will provide you with a list of articles, each containing a title and text. When answering questions, please provide detailed, comprehensive, and well-explained responses. Ensure that your answers include:

1. **In-depth explanations** of the concepts involved.
2. **Step-by-step reasoning** where applicable.
3. **References** to relevant sections of the provided articles to support your answers.
4. **Examples** or analogies to illustrate complex ideas.

Articles:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(streaming=True)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectorstore.as_retriever()

retrieval_chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

def call_chat(question):
    answer = ""
    for chunk in retrieval_chain.stream(question):
        answer += chunk
        yield {"token": chunk}

    chat_message = ChatMessage(user_id=1, question=question, answer=answer)
    db.session.add(chat_message)
    db.session.commit()

