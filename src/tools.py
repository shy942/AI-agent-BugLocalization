# Define tools

import os
import regex
import pickle
import openai
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from langchain.document_loaders import DirectoryLoader, TextLoader
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def readFile(folder_path: str) -> str:
    """Reads title.txt + description.txt """
    contents = []

    for name in ["title.txt", "description.txt"]:
        path = os.path.join(folder_path, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                contents.append(f.read().strip())

    return "\n".join(contents).strip()

# read the stopwords
def load_stopwords(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(word.strip() for word in file if word.strip())
    
def preprocess_text(bug_report_content:str) -> str:
   
    stopwords = load_stopwords("./stop_words_english.txt")
   
    # remove urls and the markdown link
    bug_report_content = regex.sub(r'\!\[.*?\]\(https?://\S+?\)', '', bug_report_content)
    bug_report_content = regex.sub(r'https?://\S+|www\.\S+', '', bug_report_content)
    
    # split camelCase and snake_case while keeping acronyms
    bug_report_content = regex.sub(r'([a-z0-9])([A-Z])', r'\1 \2', bug_report_content)
    bug_report_content = regex.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', bug_report_content)
    bug_report_content = bug_report_content.replace('_', ' ')
    
    # convert to lowercase and split for list comprehensions
    words = bug_report_content.lower().split()
    
    # remove stopwords 
    words = [word for word in words if word not in stopwords]
    
    # remove whitespace, punctuation, numbers
    text = ' '.join(words)
    text = regex.sub(r"[\s]+|[^\w\s]|[\d]+", " ", text)
    words = text.split()
    
    # remove stopwords again to catch any that were connected to punctuation
    words = [word for word in words if word not in stopwords]
    
    # remove words with fewer than 3 characters
    words = [word for word in words if len(word) >= 3]
    
    return ' '.join(words)

def processBugReportContent(bug_report_content: str) -> str:
    """Processes the content of a bug report and returns it as a string.

    Args:
        bug_report_path (str): The path to the bug report folder.

    Returns:
        str: The processed content of the bug report.
    """
    # Read the content of the bug report
    query=preprocess_text(bug_report_content)
    #print(query)
    return query 

def processBugReportQueryKeyBERT(process_content: str, top_n: int) -> str:
    """Processes the content of a bug report using KeyBERT and returns it as a string.

    Args:
        process_content (str): The content to process.
        top_n (int): The number of keywords to extract.

    Returns:
        str: The processed content.
    """
    print("Processing content with KeyBERT..."+str(process_content))
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=sentence_model)
    keywords_query = kw_model.extract_keywords(process_content, keyphrase_ngram_range=(1, 1), stop_words='english', use_maxsum=True, top_n=top_n)
    keywords_query = [word for word, _ in keywords_query]
    keywords_query = [word for word in keywords_query if len(word) >= 3]

    print("Keywords extracted: ", keywords_query)   

    return keywords_query

def processBugReportQueryReasoning(bug_report_content: str) -> str:
    """
    Use an LLM to analyze and summarize a bug report,
    including the functionality that triggers the bug.

    
    Args:
        bug_report_content (str): The content to process.

    Returns:
        str: The reasoning content.
    """

    if not bug_report_content or not isinstance(bug_report_content, str):
        return "Invalid bug report content."

    prompt = f"""
    You are a software test engineer. Given the following bug report:

    \"\"\"{bug_report_content}\"\"\"

    1. Summarize the main issue described.
    2. Explain the functionality the user executed before encountering the issue.
    Only provide the keywords, separated by commas, both summary and functionality in one line. 
    Do not include any other text, or numbers (such as 1, 2, 3, etc.).
    Duplicate keywords are allowed.
    """

    # Call your model (adjust according to your framework)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        api_key="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",
        messages=[
            {"role": "system", "content": "You are a professional software test engineer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return response["choices"][0]["message"]["content"].strip()

def index_source_code(source_code_dir: str) -> str:
    documents = []  # from DirectoryLoader, etc.
    # Load source code files (recursively from a folder)
    source_code_dir = source_code_dir  # Folder with 1000 source code files
    print("Indexing source code from: ", source_code_dir)
    # List of extensions you want to include
    source_extensions = ["*.py", "*.cpp", "*.c", "*.h", "*.hpp", "*.java", "*.js", "*.ts", "*.cs", "*.go", "*.php","*.vue"]
    # Create loaders for each extension
    loaders = [
        DirectoryLoader(
            path=source_code_dir,
            glob=ext,
            loader_cls=TextLoader,
            recursive=True
        )
        for ext in source_extensions
    ]

    # Combine all loaded documents
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} source code files.")

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        doc.metadata["filename"] = os.path.abspath(source)

    tokenized_corpus = [preprocess_text(doc.page_content) for doc in documents]
    
    # -----------------------
    # BM25 Index
    # -----------------------

    # Save only the tokenized_corpus
    # Check if index already exists
    index_path = "bm25_index.pkl"
    if os.path.exists(index_path):
        print("Index exists. Loading from file...")
        with open(index_path, "rb") as f:
            bm25_index = pickle.load(f)
    else:
        print("Index not found. Building new BM25 index...")
        bm25_index = BM25Okapi(tokenized_corpus)
    
        # Save the index
        with open(index_path, "wb") as f:
            pickle.dump(bm25_index, f)


    # -----------------------
    # FAISS Index
    # -----------------------

    # Create Document objects with processed text
    processed_documents = []
    for doc, processed_text in zip(documents, tokenized_corpus):
        processed_doc = Document(
            page_content=processed_text,
            metadata=doc.metadata
        )
        processed_documents.append(processed_doc)
    
    # Embed and build FAISS index
    model_name = "BAAI/bge-small-en-v1.5"
    #model_name = "microsoft/codebert-base"
    hf_embedder = HuggingFaceEmbeddings(model_name=model_name)

    # Define the FAISS index directory path
    faiss_index_dir = "faiss_index_dir"

    # Check if index already exists
    if os.path.exists(faiss_index_dir) and os.listdir(faiss_index_dir):
        print("FAISS index already exists. Loading it...")
        faiss_index = FAISS.load_local(faiss_index_dir, hf_embedder, allow_dangerous_deserialization=True)
    else:
        print("FAISS index not found. Creating a new one...")
        faiss_index = FAISS.from_documents(processed_documents, hf_embedder)
        # Save the new index
        faiss_index.save_local(faiss_index_dir)

    print("BM25 and FAISS indexes are loaded.")
    # Load the indexes
    bm25_index = pickle.load(open("bm25_index.pkl", "rb"))
    faiss_index = FAISS.load_local(faiss_index_dir, hf_embedder, allow_dangerous_deserialization=True)
    return bm25_index, faiss_index, processed_documents


def bug_localization_BM25_and_FAISS(bug_id: str, bug_report_query: str, top_n: int, bm25_index: BM25Okapi, faiss_index: FAISS, processed_documents: list[Document], bm25_weight: float, faiss_weight: float) -> str:
    """Localizes the bug report using BM25 and FAISS and returns it as a string.

    Args:
        bug_id (str): The ID of the bug report.
        bug_report_query (str): The content of the bug report.
        top_n (int): The number of top documents to retrieve.
        bm25_index (BM25Okapi): The BM25 index.
        faiss_index (FAISS): The FAISS index.
        processed_documents (list[Document]): The processed documents.
        bm25_weight (float): The weight of the BM25 index.
        faiss_weight (float): The weight of the FAISS index.

    Returns:
        str: The top n documents and their scores.
    """
    print("Bug localization started...")
    #print("Bug ID: ", bug_id)
    #print("Bug report query: ", bug_report_query)
    #print("Top n: ", top_n)
    print("BM25 index: ", bm25_index.corpus_size)
    print("FAISS index: ", faiss_index)
    # --- BM25 ---
    bm25_scores = bm25_index.get_scores(bug_report_query)
    
    # --- FAISS ---
    faiss_docs_and_scores = faiss_index.similarity_search_with_score(bug_report_query, k=len(processed_documents))
    faiss_scores_dict = {doc.page_content: score for doc, score in faiss_docs_and_scores}
    
    # --- Normalize scores ---
    bm25_scores_np = np.array(bm25_scores)
    bm25_norm = (bm25_scores_np - bm25_scores_np.min()) / (np.ptp(bm25_scores_np) + 1e-8)

    faiss_raw_scores = np.array([faiss_scores_dict.get(doc.page_content, 1e6) for doc in processed_documents])
    faiss_norm = 1 - ((faiss_raw_scores - faiss_raw_scores.min()) / (np.ptp(faiss_raw_scores) + 1e-8))  # invert since lower distance = more similar

    # --- Combine ---
    combined_score = bm25_weight * bm25_norm + faiss_weight * faiss_norm
    top_indices = np.argsort(combined_score)[::-1][:top_n]

    top_docs = [(processed_documents[i], combined_score[i]) for i in top_indices]
    
    string_top_docs = ""
    for i, (doc, score) in enumerate(top_docs):
        filename = doc.metadata.get('filename', 'unknown')
        short_filename = get_short_filename(filename)
        string_top_docs += f"{i+1},{short_filename},{score:.3f}"+"\n"
    return string_top_docs
     
def get_short_filename(filename_long: str) -> str:
    full_path = Path(filename_long)
    parts = full_path.parts

    # Find the first part that starts with 'Project'
    project_index = next((i for i, part in enumerate(parts) if part.startswith("Project")), None)

    if project_index is not None:
        sub_path = Path(*parts[project_index + 1:])  # Skip the ProjectXXX part
        dot_path = ".".join(sub_path.parts)
    else:
        print("No folder starting with 'Project' found in the path.")
    return dot_path


