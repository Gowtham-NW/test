# onboarding.py

# 1. Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# 2. Load document
loader = TextLoader("info.txt")
documents = loader.load()

# 3. Split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 4. Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# 5. Vector DB
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

# 8. Query
query = "what is langchain"

# 9. Retrieve relevant docs
retrieved_docs = retriever.invoke(query)

# 10. Combine context
context = "\n".join([doc.page_content for doc in retrieved_docs])

# 11. prompt engineering
prompt = f"""
Answer the question clearly in 1-2 sentences.

Context:
{context}

Question: {query}

Answer:
"""

# 12. Generate response (direct pipeline)
result = pipe(prompt)

# 13. Extract answer
answer = result[0]["generated_text"]

print("\n✅ Final Answer:\n")
print(answer.strip())