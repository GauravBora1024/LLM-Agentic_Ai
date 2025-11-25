import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://developers.google.com/machine-learning/crash-course?utm_source=chatgpt.com")
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
splits = text_splitter.split_documents(docs)
# print(len(splits))
# print(splits[0])
# print("======================================================================")
# print(splits[1])

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


vectorstore = Chroma.from_documents(documents=splits , embedding = OpenAIEmbeddings())

# print(vectorstore._collection.get())


retriever = vectorstore.as_retriever()

from langsmith import Client
prompt = Client().pull_prompt("rlm/rag-prompt")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser())

result = rag_chain.invoke("give a speific user feedback avaiable for a random course")
print(result)