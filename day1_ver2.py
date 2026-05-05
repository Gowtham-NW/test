from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Loading document, using langchain community document loaders
loader = PyPDFLoader("Data Scientist.pdf")
documents = loader.load()

# Split text, using langchain_text_splitters
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Vector DB storage by langchain vectorstores, using FAISS for smart search
vectorstore = FAISS.from_documents(docs, embeddings)


# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM flan t5 base
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=120,
    do_sample=False
)

# user Query
query = "Summarize the entire document and explain what it is about"

# Retrieve relevant docs using fewer retrieved chunks for pdf
retrieved_docs = retriever.invoke(query)[:2]
# Combine context(limiting this for pdf)
context = "\n".join([doc.page_content for doc in retrieved_docs])[:1000]

# prompt engineering
prompt = f"""
You are an expert assistant.

Based on the context below, explain what the document is about in 2 clear sentences.

Context:
{context}

Answer:
"""

# Generate response (direct pipeline)
result = pipe(prompt)

# Extract answer
answer = result[0]["generated_text"]

print("\n✅ Final Answer:\n")
print(answer.strip())