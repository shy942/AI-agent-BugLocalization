# Define tools

import os
import regex
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT


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

def processBugRepotQueryKeyBERT(process_content: str, top_n: int) -> str:
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
