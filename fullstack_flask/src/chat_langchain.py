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

**Example 1:**

Articles:
1. **Title:** The Rise of AI
   **Text:** Artificial Intelligence (AI) has been rapidly advancing over the past decade, impacting various industries from healthcare to finance. Its ability to process large datasets and learn from them has revolutionized how businesses operate and make decisions.

Question: How has AI impacted the healthcare industry?

Answer:
AI has significantly transformed the healthcare industry by enabling more accurate diagnostics, personalized treatment plans, and efficient patient management. For instance, AI-driven diagnostic tools can analyze medical images with high precision, leading to earlier detection of diseases such as cancer (Reference: Article 1, Paragraph 2). Additionally, personalized treatment plans are developed using AI algorithms that consider individual patient data, enhancing the effectiveness of therapies (Reference: Article 1, Paragraph 3). These advancements lead to improved patient outcomes and streamlined healthcare services (Reference: Article 1, Paragraph 4).

---

**Example 2:**

Articles:
2. **Title:** Renewable Energy Sources
   **Text:** Renewable energy sources like solar, wind, and hydro power are becoming increasingly important in combating climate change. These energy sources are sustainable and have a lower environmental impact compared to fossil fuels.

Question: What are the benefits of renewable energy over fossil fuels?

Answer:
Renewable energy offers several advantages over fossil fuels. Firstly, renewable sources such as solar and wind power produce electricity without emitting greenhouse gases, which helps in reducing the overall carbon footprint (Reference: Article 1, Paragraph 1). Secondly, they are sustainable and virtually inexhaustible, unlike fossil fuels which are finite and depleting (Reference: Article 1, Paragraph 2). Additionally, renewable energy technologies can create jobs in new industries and reduce dependence on imported fuels, enhancing energy security (Reference: Article 1, Paragraph 3). Lastly, renewables often have lower operational costs once installed, leading to long-term economic benefits (Reference: Article 1, Paragraph 4).

---

**Example 3:**

Articles:
3. **Title:** The History of Blockchain
   **Text:** Blockchain technology, first conceptualized in 2008, underpins cryptocurrencies like Bitcoin and has applications beyond digital currencies. Its decentralized nature ensures transparency and security in various industries.

Question: How does blockchain technology ensure the security of transactions?

Answer:
Blockchain technology ensures transaction security through its decentralized and immutable ledger system. Each transaction is recorded in a block, which is then linked to the previous block using cryptographic hashes, creating a chain (Reference: Article 1, Paragraph 2). This structure makes it extremely difficult for malicious actors to alter past transactions without modifying all subsequent blocks, which would require consensus from the majority of the network (Reference: Article 1, Paragraph 3). Additionally, blockchain employs consensus mechanisms like Proof of Work or Proof of Stake, which validate and agree upon the transaction data, further enhancing security (Reference: Article 1, Paragraph 4). The use of public and private keys also ensures that only authorized parties can initiate and verify transactions (Reference: Article 1, Paragraph 1).

---

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

