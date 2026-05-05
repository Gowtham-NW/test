# 📕 Day 1 (Version 2) – RAG with PDF Document

## 🚀 Overview

This version extends the RAG pipeline to work with **multi-page PDF documents**.

The system:

* Loads a PDF file
* Extracts text page-wise
* Splits into chunks
* Performs semantic search
* Generates answers using an LLM

---

## 🔁 Pipeline Flow

```id="flow2"
PDF → Pages → Split → Embeddings → FAISS → Retriever → LLM → Answer
```

---

## 🧩 Technologies Used

* **PyPDFLoader** – PDF parsing
* **LangChain** – Pipeline orchestration
* **FAISS** – Vector database
* **HuggingFace Transformers** – LLM
* **Sentence Transformers** – Embeddings

---

## 📂 Project Structure

```id="structure2"
├── day1_ver2.py           # PDF RAG script
├── Data Scientist.pdf     # Input PDF
├── README_day1_ver2.md
```

---

## ⚙️ Installation

```bash id="install2"
pip install pypdf
pip install langchain langchain-community langchain-text-splitters
pip install langchain-huggingface
pip install faiss-cpu
pip install transformers sentence-transformers
```

---

## ▶️ How to Run

```bash id="run2"
venv\Scripts\python.exe day1_ver2.py
```

---

## 🧠 Key Differences from Day 1

| Feature    | Day 1     | Day 1_ver2     |
| ---------- | --------- | -------------- |
| Input      | Text file | PDF            |
| Structure  | Simple    | Multi-page     |
| Complexity | Low       | Medium         |
| Chunking   | Basic     | More important |

---

## ⚠️ Challenges Faced

### 1. Token Limit Error

```id="token_err"
Token indices sequence length is longer than model limit
```

**Cause:**

* Too much context sent to model

**Fix:**

* Reduced chunk size
* Limited retrieved documents
* Truncated context

---

### 2. Poor Output Quality

Example:

```id="bad_output"
iTuring.ai
```

**Cause:**

* Weak query
* Poor prompt
* Incorrect retrieval

**Fix:**

* Improved query:
  `"Summarize the document"`
* Better prompt design
* Limited top-k retrieval

---

### 3. Retrieval Issues

**Fix:**

```python id="retriever_fix"
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

---

## 🔧 Optimizations Applied

* Chunk size: `400`
* Chunk overlap: `50`
* Top-k retrieval: `2–3`
* Context truncation
* Improved prompt engineering

---

## 🧪 Sample Query

```python id="query2"
query = "Summarize the document and explain what it is about"
```

---

## 🎯 Sample Output

```id="output2"
The document describes a Data Scientist job role at CyborgIntell, including company details, responsibilities, and required skills.
```

---

## 🧠 Key Learnings

* PDF RAG requires careful chunking
* Token limits are critical
* Retrieval quality directly affects output
* Prompt engineering is essential

---

## 📌 Conclusion

Successfully extended RAG pipeline to handle PDFs, addressing real-world challenges like token limits, retrieval accuracy, and prompt design.
