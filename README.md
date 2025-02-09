# QA_algo

A Python-based tool for performing question-answering over documents with integrated OCR fallback.

---

## Overview

This solution enables users to extract and query key information from PDF documents. It first attempts to extract text using PyPDF2 and, if that fails (especially for scanned or low-quality PDFs), falls back to OCR using Tesseract. The extracted text is then split into overlapping chunks, indexed via FAISS using embeddings from Sentence Transformers, and processed by a fine-tuned language model (Flan-T5) to generate accurate answers.

---

## Features

- **Text Extraction:**  
  Extracts text from PDFs using PyPDF2; if the extracted text is insufficient, it automatically uses OCR (via Tesseract).

- **Context Chunking:**  
  Splits the document into overlapping chunks to improve context retrieval.

- **Fast Retrieval:**  
  Uses Sentence Transformers to generate embeddings and FAISS to quickly retrieve the most relevant text pieces based on a query.

- **Answer Generation:**  
  Leverages a fine-tuned language model (Flan-T5) to produce concise answers based solely on the document content.

- **Interactive Querying:**  
  Supports interactive, real-time questioning through the terminal.

---

## Installation

To get started, clone the repository and install the required Python dependencies:

```bash
git clone https://github.com/yourusername/QA_algo.git
cd QA_algo
```

We need to install some external tools:
### Tesseract OCR

1. **Download Tesseract OCR** from [UB Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).  
2. **Install Tesseract**; by default, it installs to:
3. In the code, ensure you set:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Poppler for Windows
1. **Download Poppler for Windows** from [Poppler for Windows Releases]([https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/oschwartz10612/poppler-windows/releases/)).
2. Extract the ZIP file to a folder, for example: C:\Users\sasuk\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin
3. When calling the read_pdf function, pass this path as the poppler_path:
```python
poppler_path = r"C:\Users\sasuk\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
```

---

## Usage
Run the main script to start the application:
```python
python main.py
```

The system will:

1. Read and preprocess the specified PDF file (using PyPDF2 or OCR fallback).
2. Split the extracted text into manageable chunks for context retrieval.
3. Create a FAISS index to quickly find relevant text segments.
4. Load (and optionally fine-tune) the language model.
5. Allow you to ask questions in real time, returning answers based on the document content.
6. When prompted, enter your question in the terminal. Type exit to quit the interactive mode.
