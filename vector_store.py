
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import pickle
from pathlib import Path
import uuid
from collections import defaultdict, Counter
import math
import re

class SimpleVectorStore:
    """Simple in-memory vector store using TF-IDF and cosine similarity"""

    def __init__(self):
        self.documents = {}  # doc_id -> document info
        self.chunks = {}     # chunk_id -> chunk info
        self.vectors = {}    # chunk_id -> vector
        self.vocabulary = set()
        self.idf_scores = {}
        self.doc_counter = 0
        self.chunk_counter = 0

    def add_document(self, document_name: str, chunks: List[Dict[str, Any]]) -> str:
        """Add a document with its chunks to the vector store"""
        doc_id = str(uuid.uuid4())

        # Store document info
        self.documents[doc_id] = {
            'name': document_name,
            'chunks': [],
            'added_at': self._get_timestamp()
        }

        chunk_ids = []

        # Process each chunk
        for chunk_data in chunks:
            chunk_id = self._add_chunk(doc_id, document_name, chunk_data)
            chunk_ids.append(chunk_id)

        self.documents[doc_id]['chunks'] = chunk_ids

        # Recompute IDF scores
        self._compute_idf_scores()

        # Recompute all vectors
        self._recompute_vectors()

        return doc_id

    def _add_chunk(self, doc_id: str, doc_name: str, chunk_data: Dict[str, Any]) -> str:
        """Add a single chunk to the store"""
        chunk_id = str(uuid.uuid4())

        # Extract and process text
        text = chunk_data['text']
        tokens = self._tokenize(text)

        # Update vocabulary
        self.vocabulary.update(tokens)

        # Store chunk info
        self.chunks[chunk_id] = {
            'id': chunk_id,
            'document_id': doc_id,
            'document_name': doc_name,
            'text': text,
            'tokens': tokens,
            'metadata': {
                'word_count': chunk_data.get('word_count', len(tokens)),
                'start_sentence': chunk_data.get('start_sentence', 0),
                'end_sentence': chunk_data.get('end_sentence', 0)
            }
        }

        return chunk_id

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Filter out very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}

        tokens = [token for token in tokens if len(token) > 2 and token not in stop_words]

        return tokens

    def _compute_idf_scores(self):
        """Compute IDF scores for all terms"""
        total_docs = len(self.chunks)

        # Count document frequency for each term
        doc_freq = defaultdict(int)

        for chunk in self.chunks.values():
            unique_tokens = set(chunk['tokens'])
            for token in unique_tokens:
                doc_freq[token] += 1

        # Compute IDF scores
        self.idf_scores = {}
        for token in self.vocabulary:
            if doc_freq[token] > 0:
                self.idf_scores[token] = math.log(total_docs / doc_freq[token])
            else:
                self.idf_scores[token] = 0

    def _compute_tf_idf_vector(self, tokens: List[str]) -> np.ndarray:
        """Compute TF-IDF vector for a list of tokens"""
        # Create vocabulary mapping
        vocab_list = sorted(list(self.vocabulary))
        vocab_index = {word: i for i, word in enumerate(vocab_list)}

        # Initialize vector
        vector = np.zeros(len(vocab_list))

        # Compute term frequencies
        tf_scores = Counter(tokens)
        total_tokens = len(tokens)

        # Compute TF-IDF for each token
        for token, count in tf_scores.items():
            if token in vocab_index:
                tf = count / total_tokens
                idf = self.idf_scores.get(token, 0)
                vector[vocab_index[token]] = tf * idf

        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def _recompute_vectors(self):
        """Recompute vectors for all chunks"""
        for chunk_id, chunk in self.chunks.items():
            self.vectors[chunk_id] = self._compute_tf_idf_vector(chunk['tokens'])

    def search(self, query: str, top_k: int = 5, 
               similarity_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        if not self.chunks:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Compute query vector
        query_vector = self._compute_tf_idf_vector(query_tokens)

        # Compute similarities
        similarities = []

        for chunk_id, chunk_vector in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, chunk_vector)

            if similarity >= similarity_threshold:
                chunk_info = self.chunks[chunk_id]
                similarities.append({
                    'chunk_id': chunk_id,
                    'document_id': chunk_info['document_id'],
                    'document': chunk_info['document_name'],
                    'text': chunk_info['text'],
                    'score': float(similarity),
                    'metadata': chunk_info['metadata']
                })

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0

        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def remove_document(self, doc_id: str):
        """Remove a document and all its chunks"""
        if doc_id not in self.documents:
            return

        # Remove chunks
        chunk_ids = self.documents[doc_id]['chunks']
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                del self.chunks[chunk_id]
            if chunk_id in self.vectors:
                del self.vectors[chunk_id]

        # Remove document
        del self.documents[doc_id]

        # Recompute vocabulary and vectors
        self._recompute_vocabulary()
        self._compute_idf_scores()
        self._recompute_vectors()

    def _recompute_vocabulary(self):
        """Recompute vocabulary from remaining chunks"""
        self.vocabulary = set()
        for chunk in self.chunks.values():
            self.vocabulary.update(chunk['tokens'])

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a document"""
        return self.documents.get(doc_id)

    def get_chunk_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a chunk"""
        return self.chunks.get(chunk_id)

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'vocabulary_size': len(self.vocabulary),
            'avg_chunk_length': np.mean([len(chunk['tokens']) for chunk in self.chunks.values()]) if self.chunks else 0
        }

# Alias for main class
VectorStore = SimpleVectorStore
