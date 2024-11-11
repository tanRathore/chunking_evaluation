from base_chunker import BaseChunker
from chunking_evaluation.utils import gemini_token_count
from chunking_evaluation.chunking import RecursiveTokenChunker
from google.auth import service_account
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel
import os
import re
import numpy as np
from tqdm import tqdm
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
import pi_heif
import getpass
import os
import anthropic
import backoff
# Setting up Google API credentials
credentials = service_account.Credentials.from_service_account_file(
    'rbio-p-datasharing-b5c1d9a2deba.json'
)
aiplatform.init(credentials=credentials, project='rbio-p-datasharing')
vertexai.init(project="rbio-p-datasharing", location="us-west1")

class GeminiClient:
    def __init__(self, model_name, project, location, api_key=None):
        self.model_name = model_name
        self.project = project
        self.location = location
        if api_key is not None:
            os.environ["GOOGLE_API_KEY"] = api_key

    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        model = TextGenerationModel.from_pretrained(self.model_name)
        full_prompt = system_prompt + "\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        response = model.predict(full_prompt, max_output_tokens=max_tokens, temperature=temperature)
        return response.text
class AnthropicClient:
    def __init__(self, model_name, api_key=None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages
            )
            return message.content[0].text
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e
        
class OpenAIClient:
    def __init__(self, model_name, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        try:
            gpt_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages

            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=gpt_messages,
                temperature=temperature
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e

class LLMSemanticChunker(BaseChunker):
    """
    LLMSemanticChunker is a class designed to split text into thematically consistent sections based on suggestions from a Language Model (LLM).
    """
    def __init__(self, organisation: str = "gemini", api_key: str = None, model_name: str = None):
        if organisation == "gemini":
            if model_name is None:
                model_name = "text-bison@001"
            self.client = GeminiClient(model_name, project="rbio-p-datasharing", location="us-west1", api_key=api_key)
        elif organisation == "openai":
            if model_name is None:
                model_name = "gpt-4o"
            self.client = OpenAIClient(model_name, api_key=api_key)
        elif organisation == "anthropic":
            if model_name is None:
                model_name = "claude-3-5-sonnet-20240620"
            self.client = AnthropicClient(model_name, api_key=api_key)
        else:
            raise ValueError("Invalid organisation. Currently only 'gemini' is supported.")

        self.splitter = RecursiveTokenChunker(
            chunk_size=50,
            chunk_overlap=0,
            length_function=gemini_token_count
        )

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                    "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER. "
                    "Your response should be in the form: 'split_after: 3, 5'."
                )
            },
            {
                "role": "user", 
                "content": (
                    "CHUNKED_TEXT: " + chunked_input + "\n\n"
                    "Respond only with the IDs of the chunks where you believe a split should occur. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: " + str(current_chunk) + "." + (f"\nThe previous response of {invalid_response} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again." if invalid_response else "")
                )
            },
        ]
        return messages

    def split_text(self, text):
        chunks = self.splitter.split_text(text)
        split_indices = []
        short_cut = len(split_indices) > 0
        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            while True and not short_cut:
                if current_chunk >= len(chunks) - 4:
                    break

                token_count = 0
                chunked_input = ''

                for i in range(current_chunk, len(chunks)):
                    token_count += gemini_token_count(chunks[i])
                    chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
                    if token_count > 800:
                        break

                messages = self.get_prompt(chunked_input, current_chunk)
                while True:
                    result_string = self.client.create_message(messages[0]['content'], messages[1:], max_tokens=200, temperature=0.2)
                    split_after_line = [line for line in result_string.split('\n') if 'split_after:' in line][0]
                    numbers = re.findall(r'\d+', split_after_line)
                    numbers = list(map(int, numbers))

                    if not (numbers != sorted(numbers) or any(number < current_chunk for number in numbers)):
                        break
                    else:
                        messages = self.get_prompt(chunked_input, current_chunk, numbers)
                        print("Response: ", result_string)
                        print("Invalid response. Please try again.")

                split_indices.extend(numbers)
                current_chunk = numbers[-1]

                if len(numbers) == 0:
                    break

                pbar.update(current_chunk - pbar.n)

        pbar.close()
        chunks_to_split_after = [i - 1 for i in split_indices]
        docs = []
        current_chunk = ''
        for i, chunk in enumerate(chunks):
            current_chunk += chunk + ' '
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            docs.append(current_chunk.strip())

        return docs
