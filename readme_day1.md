# 📘 Day 1 – RAG Fundamentals with Text File

## 🚀 Overview

This project demonstrates a basic **Retrieval-Augmented Generation (RAG)** pipeline using a **text file** as input.

The system:

* Loads a `.txt` document
* Converts it into embeddings
* Stores embeddings in a vector database
* Retrieves relevant information based on a query
* Uses an LLM to generate answers

---

## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) combines:

* **Retrieval** → Finding relevant information
* **Generation** → Producing answers using an LLM

Instead of relying only on pre-trained knowledge, the model uses **your document as context**.

---

## 🔁 Pipeline Flow

```id="flow1"
Text File → Split → Embeddings → FAISS → Retriever → LLM → Answer
```

---

## 🧩 Technologies Used

* **LangChain** – Pipeline orchestration
* **FAISS** – Vector database
* **HuggingFace Transformers** – LLM (FLAN-T5)
* **Sentence Transformers** – Embeddings

---

## 📂 Project Structure

```id="structure1"
├── day1.py       # Main RAG script
├── info.txt      # Input text file
├── README_day1.md
```

---

## ⚙️ Installation

```bash id="install1"
pip install langchain langchain-community langchain-text-splitters
pip install langchain-huggingface
pip install faiss-cpu
pip install transformers sentence-transformers
```

---

## ▶️ How to Run

```bash id="run1"
venv\Scripts\python.exe day1.py
```

---

## 🧠 Key Concepts

### 🔹 Embeddings

Convert text into numerical vectors for similarity comparison.

### 🔹 Chunking

Splitting text into smaller parts improves retrieval accuracy.

### 🔹 Vector Store (FAISS)

Stores embeddings and enables similarity search.

### 🔹 Retriever

Finds relevant chunks based on user query.

### 🔹 LLM

Generates answers using retrieved context.

---

## ⚠️ Challenges Faced

* LangChain import errors (version changes)
* OpenAI API quota issues → switched to HuggingFace
* Model mismatch issues (text-generation vs text2text-generation)
* Output quality issues with smaller models

---

## 🔧 Fixes Applied

* Used `flan-t5-base` for better instruction following
* Improved prompt engineering
* Adjusted chunk size and retrieval logic

---

## 🎯 Sample Output

```id="output1"
LangChain is a framework used to build applications powered by large language models such as chatbots and RAG systems.
```

---

## 🧠 Learnings

* RAG pipelines depend on retrieval + prompt + model
* Model choice affects output quality significantly
* LangChain simplifies complex workflows

---

## 📌 Conclusion

Successfully built a basic RAG system using a text file, understanding the full pipeline from data loading to answer generation.
