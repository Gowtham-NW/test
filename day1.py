from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Loading document, using langchain community document loaders
loader = TextLoader("info.txt")
documents = loader.load()

# Split text, using langchain_text_splitters
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# 5. Vector DB storage by langchain vectorstores, using FAISS for smart search
vectorstore = FAISS.from_documents(docs, embeddings)


# 6. Retriever
retriever = vectorstore.as_retriever()

# 7. LLM (Better model)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=120,
    do_sample=False
)

# user Query
query = "what is langchain"

# Retrieve relevant docs
retrieved_docs = retriever.invoke(query)

# Combine context
context = "\n".join([doc.page_content for doc in retrieved_docs])

# prompt engineering
prompt = f"""
Answer the question clearly in 1-2 sentences.

Context:
{context}

Question: {query}

Answer:
"""

# Generate response (direct pipeline)
result = pipe(prompt)

# Extract answer
answer = result[0]["generated_text"]

print("\n✅ Final Answer:\n")
print(answer.strip())