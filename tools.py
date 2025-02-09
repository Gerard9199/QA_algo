import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"C:\Users\sasuk\Desktop\cache_transformers"
os.environ["HF_HOME"] = r"C:\Users\sasuk\Desktop\cache_transformers\huggingface"
os.environ["HF_HUB_CACHE"] = r"C:\Users\sasuk\Desktop\cache_transformers\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"C:\Users\sasuk\Desktop\cache_transformers\transformers_cache"

import torch
import faiss
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

def split_text(text, chunk_size=200, overlap=50):
    """
    We break the text into words and then create chunks with some overlapping words

    Args:
        text (str): The text to split.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of overlapping words between consecutive chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ocr_pdf(pdf_path, poppler_path):
    """
    Performs OCR on a PDF file using pdf2image and pytesseract.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: The OCR-extracted text.
    """
    print("Performing OCR on the PDF...")
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    ocr_text = ""
    for page in pages:
        ocr_text += pytesseract.image_to_string(page) + "\n"
    return ocr_text

def read_pdf(pdf_path, poppler_path):
    """
    Function to read a PDF file; if normal extraction fails, use OCR instead

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    with open(pdf_path, "rb") as file:
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print("Error reading PDF with PyPDF2:", str(e))
    # If the text is very short, we assume the PDF may be scanned and use OCR
    if len(text.strip()) < 100:
        print("Extracted text is insufficient. Falling back to OCR.")
        text = ocr_pdf(pdf_path, poppler_path)
    return text

def index_chunks(chunks, embedder):
    """
    Function to calculate embeddings for text chunks and create a FAISS index

    Args:
        chunks (List[str]): List of text chunks.
        embedder (SentenceTransformer): The model used to compute embeddings.

    Returns:
        Tuple[faiss.IndexFlatL2, int]: The FAISS index and the embedding dimension.
    """
    print("Calculating embeddings for the text chunks...")
    chunk_embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embedding_dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(chunk_embeddings)
    print(f"FAISS index created with {index.ntotal} vectors.")
    return index, embedding_dim

def retrieve_chunks(query, index, embedder, chunks, k=5):
    """
    Function to get the most relevant text pieces for a given question

    Args:
        query (str): The query or question.
        index (faiss.IndexFlatL2): The pre-built FAISS index.
        embedder (SentenceTransformer): The model used to compute embeddings.
        chunks (List[str]): The original list of text chunks.
        k (int): Number of chunks to retrieve.

    Returns:
        List[str]: A list of relevant text chunks.
    """
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [chunks[idx] for idx in indices[0]]

def generate_response(question, chunks_context, tokenizer, model):
    """
    Build a prompt that tells the model to act as an expert legal analyst
    and use only the provided context to answer the question

    Args:
        question (str): The question to answer.
        chunks_context (List[str]): List of relevant context chunks.
        tokenizer: The language model's tokenizer.
        model: The language generation model.

    Returns:
        str: The generated answer.
    """
    prompt = (
        "You are an expert legal analyst. Based on the following excerpt from a legal document, "
        "give a concise summary and list the key points in Spanish. Focus on the main parties, "
        "their identification, addresses, and any responsibilities or obligations mentioned. "
        "If the context does not have enough information, say so.\n\n"
        "Context:\n" + " ".join(chunks_context) + "\n\n"
        "Question: " + question + "\n\n"
        "Summary and key points (in Spanish):"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=2, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    del inputs, outputs
    torch.cuda.empty_cache()
    return answer

def load_language_model(model_name):
    """
    Function to load the language model and tokenizer

    Args:
        model_name (str): The identifier of the language model.

    Returns:
        Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]: The loaded tokenizer and model.
    """
    print("Loading language model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    return tokenizer, model

def prepare_qa_dataset(context):
    """
    Function to create a simple QA dataset from the document context

    Args:
        context (str): The context text to generate QA pairs from.

    Returns:
        List[Dict[str, Any]]: A list of QA pairs, each with "input_text" and "target_text".
    """
    dataset = [
        {
            "input_text": "Question: What is the document about?\nContext: " + context,
            "target_text": "The document is about the financial and operational analysis of a lease contract."
        }
    ]
    return dataset

def fine_tune_language_model(tokenizer, model, train_dataset):
    """
    Function to fine-tune the language model on a QA dataset with minimal resource usage

    Args:
        tokenizer: The language model's tokenizer.
        model: The language model to fine-tune.
        train_dataset (List[Dict[str, Any]]): A list of training examples with 'input_text' and 'target_text'.

    Returns:
        AutoModelForSeq2SeqLM: The fine-tuned language model.
    """

    # Convert our list of QA examples into a dataset
    hf_dataset = Dataset.from_list(train_dataset)

    # Tokenize the dataset using short maximum lengths to save memory
    def tokenize_function(example):
        inputs = tokenizer(example["input_text"], truncation=True, max_length=64)
        labels = tokenizer(text_target=example["target_text"], truncation=True, max_length=16)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=hf_dataset.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # Enable gradient checkpointing to use less memory
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Set training parameters to be as light as possible
    training_args = Seq2SeqTrainingArguments(
        output_dir="./fine_tuned_model_minimal",
        evaluation_strategy="no",      # No evaluation during training
        save_strategy="no",            # Do not save checkpoints to avoid extra memory use
        learning_rate=3e-5,
        per_device_train_batch_size=1,  # Use the smallest batch size
        gradient_accumulation_steps=8,  # Simulate a larger batch size without loading many examples at once
        weight_decay=0.01,
        num_train_epochs=1,             # Only one epoch to reduce resource usage
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        disable_tqdm=True,
        dataloader_num_workers=0
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    return model

def main():
    poppler_path = r"C:\Users\sasuk\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
    pdf_path = r"C:\Users\sasuk\Downloads\DS challenge_\DS challenge\CONTRATO_AP000000718.pdf"
    model_name = "google/flan-t5-xl"
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tuning = False

    print("Starting document ingestion and processing")
    pdf_content = read_pdf(pdf_path, poppler_path)
    pdf_chunks = split_text(pdf_content, chunk_size=200, overlap=50)
    global_chunks = pdf_chunks.copy()

    print("Loading embedding model")
    faiss_index, _ = index_chunks(global_chunks, embedder)
    tokenizer, model = load_language_model(model_name)

    if tuning:
        qa_dataset = prepare_qa_dataset(pdf_content)
        print("Finetuning the language model on the QA dataset")
        model = fine_tune_language_model(tokenizer, model, qa_dataset)

    while True:
        user_question = input("Type your question: ")
        if user_question.strip().lower() == "exit":
            print("***Closing the system***")
            break
        # Se recuperan los fragmentos relevantes para la pregunta
        relevant_chunks = retrieve_chunks(user_question, faiss_index, embedder, global_chunks, k=10)
        # Se genera la respuesta basada en el contexto recuperado
        answer = generate_response(user_question, relevant_chunks, tokenizer, model)
        print("\Answer:", answer, "\n")