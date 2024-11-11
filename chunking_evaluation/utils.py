from enum import Enum
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import os
from vertexai.preview.language_models import TextEmbeddingModel
from google.auth import service_account
import tiktoken
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich import print
import vertexai
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from google.cloud import aiplatform
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel
from rich import print
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from google.oauth2 import service_account
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextGenerationModel
import os
from typing import Optional, List
from pydantic import BaseModel
from langchain import hub
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import json
from vertexai.preview.tokenization import get_tokenizer_for_model
# Set up Google API credentials
credentials = service_account.Credentials.from_service_account_file(
    'rbio-p-datasharing-b5c1d9a2deba.json'
)

def find_query_despite_whitespace(document, query):
    # Normalize spaces and newlines in the query
    normalized_query = re.sub(r'\s+', ' ', query).strip()
    
    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())
    
    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)
    
    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    else:
        return None

def rigorous_document_search(document: str, target: str):
    """
    This function performs a rigorous search of a target string within a document. 
    It handles issues related to whitespace, changes in grammar, and other minor text alterations.
    The function first checks for an exact match of the target in the document. 
    If no exact match is found, it performs a raw search that accounts for variations in whitespace.
    If the raw search also fails, it splits the document into sentences and uses fuzzy matching 
    to find the sentence that best matches the target.
    
    Args:
        document (str): The document in which to search for the target.
        target (str): The string to search for within the document.

    Returns:
        tuple: A tuple containing the best match found in the document, its start index, and its end index.
        If no match is found, returns None.
    """
    if target.endswith('.'):
        target = target[:-1]
    
    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    # Split the text into sentences
    sentences = re.split(r'[.!?]\s*|\n', document)

    # Find the sentence that matches the query best
    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if best_match[1] < 98:
        return None
    
    reference = best_match[0]

    start_index = document.find(reference)
    end_index = start_index + len(reference)

    return reference, start_index, end_index

def get_gemini_embedding_function():
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embedding_model

# Count the number of tokens in each page_content
def gemini_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = get_tokenizer_for_model("gemini-1.5-flash-001")
    #encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = encoding.count_tokens(string).total_tokens
    return num_tokens

class Language(str, Enum):
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"