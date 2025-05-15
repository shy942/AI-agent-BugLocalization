# Define tools

import os
import regex
import pickle
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

def index_source_code(source_code_dir: str) -> str:
    documents = []  # from DirectoryLoader, etc.
    # Load source code files (recursively from a folder)
    source_code_dir = source_code_dir  # Folder with 1000 source code files
    print("Indexing source code from: ", source_code_dir)
    # List of extensions you want to include
    source_extensions = ["*.py", "*.cpp", "*.c", "*.h", "*.hpp", "*.java", "*.js", "*.ts", "*.cs", "*.go", "*.php"]
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
        #print(loader.load())
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} source code files.")

    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        doc.metadata["filename"] = os.path.abspath(source)

    #tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    tokenized_corpus = [preprocess_text(doc.page_content) for doc in documents]
    #print(tokenized_corpus[0])

    # -----------------------
    # BM25 Index
    # -----------------------

    # Save only the tokenized_corpus
    # Check if index already exists
    index_path = "bm25_index.pkl"
    if os.path.exists(index_path):
        print("Index exists. Loading from file...")
        with open(index_path, "rb") as f:
            bm25 = pickle.load(f)
    else:
        print("Index not found. Building new BM25 index...")
        bm25 = BM25Okapi(tokenized_corpus)
    
        # Save the index
        with open(index_path, "wb") as f:
            pickle.dump(bm25, f)


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

    # Chunk the source code files
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(processed_documents)  # Use processed_documents instead of tokenized_corpus
    # Add filename to each chunk's metadata
    for chunk in chunks:
        #print(chunk)
        chunk.metadata["filename"] = chunk.metadata.get("source", "unknown")
    print(f"Generated {len(chunks)} chunks.")


    # Embed and build FAISS index
    model_name = "BAAI/bge-small-en-v1.5"
    #model_name = "microsoft/codebert-base"
    hf_embedder = HuggingFaceEmbeddings(model_name=model_name)

    # Define the FAISS index directory path
    faiss_index_dir = "faiss_index_dir"

    # Check if index already exists
    if os.path.exists(faiss_index_dir):
        print("FAISS index already exists. Loading it...")
        faiss_index = FAISS.load_local(faiss_index_dir, hf_embedder, allow_dangerous_deserialization=True)
    else:
        print("FAISS index not found. Creating a new one...")
        faiss_index = FAISS.from_documents(documents, hf_embedder)
        # Save the new index
        faiss_index.save_local(faiss_index_dir)

    print("FAISS index loaded.")


def load_index_bm25_and_faiss(bm25_index_path: str, faiss_index_path: str) -> tuple[BM25Okapi, FAISS]:
    """Loads the BM25 and FAISS indexes from the given paths.

    Args:
        bm25_index_path (str): The path to the BM25 index.
        faiss_index_path (str): The path to the FAISS index.    

    Returns:
        tuple[BM25Okapi, FAISS]: The BM25 and FAISS indexes.
    """
    bm25_index = pickle.load(open(bm25_index_path, "rb"))
    print("BM25 index loaded.")

    #faiss_index = FAISS.load_local(faiss_index_path, hf_embedder, allow_dangerous_deserialization=True)

        # Load BM25 index
    #bm25_index = pickle.load(open("bm25_index.pkl", "rb"))
    # with open(bm25_index_path, "rb") as f:
    #     tokenized_corpus = pickle.load(f)
    # bm25 = BM25Okapi(tokenized_corpus)
    # print("BM25 index loaded.")
    # Load FAISS index
    #faiss_index = pickle.load(open("faiss_index.pkl", "rb"))
    # Embed and build FAISS index
    model_name = "BAAI/bge-small-en-v1.5"
    #model_name = "microsoft/codebert-base"
    hf_embedder = HuggingFaceEmbeddings(model_name=model_name)
    faiss_index = FAISS.load_local(faiss_index_path, hf_embedder, allow_dangerous_deserialization=True)
    print("FAISS index loaded.")
    return bm25_index, faiss_index


def bug_localization_BM25_and_FAISS(bug_report_query: str, top_n: int, bm25_index: BM25Okapi, faiss_index: FAISS) -> str:
    """Localizes the bug report using BM25 and FAISS and returns it as a string.

    Args:
        bug_report_query (str): The content of the bug report.
        top_n (int): The number of keywords to extract.
        bm25_index (BM25Okapi): The BM25 index.
        faiss_index (FAISS): The FAISS index.

    Returns:
        str: The localized bug report.
    """
    print("Bug localization started...")
    print("Bug report query: ", bug_report_query)
    print("Top n: ", top_n)
    print("BM25 index: ", bm25_index.corpus_size)
    print("FAISS index: ", faiss_index)

 
    bm25_scores = bm25_index.get_scores(bug_report_query)
    print(f"BM25 scores: {bm25_scores}")

    # Get FAISS results
    faiss_results = faiss_index.similarity_search(bug_report_query, k=top_n)
    print("FAISS results: ", faiss_results)

    # Combine results
    combined_results = bm25_scores + faiss_results 
    
    # Return the combined results
    return combined_results

