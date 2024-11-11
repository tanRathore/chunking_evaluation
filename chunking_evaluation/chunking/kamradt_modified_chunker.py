from typing import Optional
from .base_chunker import BaseChunker
from .recursive_token_chunker import RecursiveTokenChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chromadb.api.types import (
    Embeddable,
    EmbeddingFunction,
)

import numpy as np

class KamradtModifiedChunker(BaseChunker):
    """
    A chunker that splits text into chunks of approximately a specified average size based on semantic similarity.

    This was adapted from Greg Kamradt's notebook on chunking but with the modification of including an average chunk size parameter. The original code can be found at: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

    This class extends the functionality of the BaseChunker by incorporating a method to combine sentences based on a buffer size, calculate cosine distances between combined sentences, and perform a binary search on similarity thresholds to achieve chunks of desired average size.
    """
    def __init__(
        self, 
        avg_chunk_size: int = 400, 
        min_chunk_size: int = 50, 
        embedding_function: Optional[EmbeddingFunction[Embeddable]] = None, 
        length_function: Optional[Callable[[str], int]] = None
    ):
        """
        Initializes the KamradtModifiedChunker with the specified parameters.

        Args:
            avg_chunk_size (int, optional): The desired average chunk size in tokens. Defaults to 400.
            min_chunk_size (int, optional): The minimum chunk size in tokens. Defaults to 50.
            embedding_function (EmbeddingFunction[Embeddable], optional): A function to obtain embeddings for text. Defaults to Gemini's embedding function if not provided.
            length_function (function, optional): A function to calculate token length of a text. Defaults to Gemini's token count function if not provided.
        """
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=length_function or self._gemini_token_count
        )
        
        self.avg_chunk_size = avg_chunk_size
        if embedding_function is None:
            embedding_function = self._get_gemini_embedding_function()
        self.embedding_function = embedding_function
        self.length_function = length_function or self._gemini_token_count

    def _get_gemini_embedding_function(self):
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001").embed_documents

    def _gemini_token_count(self, text: str) -> int:
        # Assuming token count can be approximated by the length of the embedding vector
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001").embed_documents([text])[0]
        return len(embedding)

    def combine_sentences(self, sentences, buffer_size=1):
        for i in range(len(sentences)):
            combined_sentence = ''
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '
            combined_sentence += sentences[i]['sentence']
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']
            sentences[i]['combined_sentence'] = combined_sentence
        return sentences

    def calculate_cosine_distances(self, sentences):
        BATCH_SIZE = 500
        distances = []
        embedding_matrix = None
        for i in range(0, len(sentences), BATCH_SIZE):
            batch_sentences = sentences[i:i + BATCH_SIZE]
            batch_sentences = [sentence['combined_sentence'] for sentence in batch_sentences]
            embeddings = self.embedding_function(batch_sentences)
            batch_embedding_matrix = np.array(embeddings)
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = embedding_matrix / norms

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
        
        for i in range(len(sentences) - 1):
            similarity = similarity_matrix[i, i + 1]
            distance = 1 - similarity
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance

        return distances, sentences

    def split_text(self, text):
        sentences_strips = self.splitter.split_text(text)
        sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences_strips)]
        sentences = self.combine_sentences(sentences, 3)
        distances, sentences = self.calculate_cosine_distances(sentences)

        total_tokens = sum(self.length_function(sentence['sentence']) for sentence in sentences)
        avg_chunk_size = self.avg_chunk_size
        number_of_cuts = total_tokens // avg_chunk_size

        lower_limit = 0.0
        upper_limit = 1.0
        distances_np = np.array(distances)

        while upper_limit - lower_limit > 1e-6:
            threshold = (upper_limit + lower_limit) / 2.0
            num_points_above_threshold = np.sum(distances_np > threshold)
            
            if num_points_above_threshold > number_of_cuts:
                lower_limit = threshold
            else:
                upper_limit = threshold

        indices_above_thresh = [i for i, x in enumerate(distances) if x > threshold] 
        start_index = 0
        chunks = []

        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks