
import re
from typing import List, Dict, Any, Optional
import time

class RAGPipeline:
    """Main RAG pipeline for question answering"""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.chunk_size = 400
        self.overlap = 50
        self.top_k = 5
        self.similarity_threshold = 0.1

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Generate answer for a question using RAG"""
        # Step 1: Retrieve relevant chunks
        relevant_chunks = self.vector_store.search(
            question, 
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )

        if not relevant_chunks:
            return {
                'answer': "I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing your question or upload more relevant documents.",
                'sources': [],
                'confidence': 0.0
            }

        # Step 2: Generate answer using retrieved chunks
        answer = self._generate_answer(question, relevant_chunks)

        # Step 3: Calculate confidence based on source quality
        confidence = self._calculate_confidence(relevant_chunks)

        return {
            'answer': answer,
            'sources': relevant_chunks,
            'confidence': confidence
        }

    def _generate_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate answer based on retrieved chunks"""
        # For this implementation, we'll use a rule-based approach
        # In a production system, this would use an LLM like GPT-4, Claude, etc.

        # Combine relevant text from chunks
        context_text = " ".join([chunk['text'] for chunk in chunks])

        # Simple extractive answering approach
        answer = self._extract_answer(question, context_text, chunks)

        return answer

    def _extract_answer(self, question: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Extract answer from context using rule-based approach"""
        question_lower = question.lower()
        context_lower = context.lower()

        # Question type detection
        question_type = self._detect_question_type(question_lower)

        if question_type == 'definition':
            return self._answer_definition_question(question, context, chunks)
        elif question_type == 'comparison':
            return self._answer_comparison_question(question, context, chunks)
        elif question_type == 'list':
            return self._answer_list_question(question, context, chunks)
        elif question_type == 'how':
            return self._answer_how_question(question, context, chunks)
        elif question_type == 'why':
            return self._answer_why_question(question, context, chunks)
        else:
            return self._answer_general_question(question, context, chunks)

    def _detect_question_type(self, question: str) -> str:
        """Detect the type of question"""
        if any(word in question for word in ['what is', 'what are', 'define', 'definition']):
            return 'definition'
        elif any(word in question for word in ['difference', 'compare', 'vs', 'versus', 'between']):
            return 'comparison'
        elif any(word in question for word in ['list', 'types', 'kinds', 'examples']):
            return 'list'
        elif question.startswith('how'):
            return 'how'
        elif question.startswith('why'):
            return 'why'
        else:
            return 'general'

    def _answer_definition_question(self, question: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Answer definition-type questions"""
        # Extract key term from question
        key_terms = self._extract_key_terms(question)

        # Find sentences that define the key terms
        sentences = self._split_into_sentences(context)
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in key_terms):
                # Look for definition patterns
                if any(pattern in sentence_lower for pattern in [' is ', ' are ', ' means ', ' refers to ', ' defined as ']):
                    relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            # Combine and clean up the definition
            answer = self._combine_sentences(relevant_sentences[:2])  # Use top 2 sentences
            return f"Based on the documents: {answer}"
        else:
            # Fallback: return most relevant chunk
            best_chunk = chunks[0]['text'][:300] + "..." if len(chunks[0]['text']) > 300 else chunks[0]['text']
            return f"From the most relevant section: {best_chunk}"

    def _answer_comparison_question(self, question: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Answer comparison-type questions"""
        # Extract comparison terms
        comparison_terms = self._extract_comparison_terms(question)

        sentences = self._split_into_sentences(context)
        relevant_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence mentions both terms or comparison keywords
            term_count = sum(1 for term in comparison_terms if term in sentence_lower)
            has_comparison = any(word in sentence_lower for word in ['difference', 'while', 'whereas', 'unlike', 'compared to', 'but', 'however'])

            if term_count >= 2 or (term_count >= 1 and has_comparison):
                relevant_sentences.append(sentence.strip())

        if relevant_sentences:
            answer = self._combine_sentences(relevant_sentences[:3])
            return f"Based on the comparison in the documents: {answer}"
        else:
            # Provide information about each term separately
            term_info = []
            for term in comparison_terms[:2]:  # Max 2 terms
                term_sentences = [s for s in sentences if term in s.lower()]
                if term_sentences:
                    term_info.append(f"{term.title()}: {term_sentences[0][:200]}...")

            if term_info:
                return "Here's what I found about each: " + " | ".join(term_info)
            else:
                return f"I found relevant information: {chunks[0]['text'][:400]}..."

    def _answer_list_question(self, question: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Answer list-type questions"""
        sentences = self._split_into_sentences(context)

        # Look for list patterns
        list_sentences = []
        for sentence in sentences:
            if any(pattern in sentence.lower() for pattern in ['types of', 'kinds of', 'include', 'such as', 'are:', '1.', '2.', '•', '-']):
                list_sentences.append(sentence.strip())

        if list_sentences:
            answer = self._combine_sentences(list_sentences[:3])
            return f"Based on the documents: {answer}"
        else:
            # Extract potential list items using patterns
            items = self._extract_list_items(context)
            if items:
                return f"Based on the documents, here are the key items: {', '.join(items[:5])}."
            else:
                return f"Here's the relevant information: {chunks[0]['text'][:400]}..."

    def _answer_how_question(self, question: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Answer how-type questions"""
        sentences = self._split_into_sentences(context)

        # Look for process/method descriptions
        how_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['process', 'method', 'way', 'steps', 'first', 'then', 'next', 'finally', 'by']):
                how_sentences.append(sentence.strip())

        if how_sentences:
            answer = self._combine_sentences(how_sentences[:3])
            return f"Based on the process described: {answer}"
        else:
            return f"Here's the relevant information about the process: {chunks[0]['text'][:400]}..."

    def _answer_why_question(self, question: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Answer why-type questions"""
        sentences = self._split_into_sentences(context)

        # Look for causal/reason patterns
        why_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['because', 'due to', 'reason', 'cause', 'result', 'leads to', 'therefore', 'thus', 'since']):
                why_sentences.append(sentence.strip())

        if why_sentences:
            answer = self._combine_sentences(why_sentences[:2])
            return f"Based on the explanation: {answer}"
        else:
            return f"Here's the relevant context: {chunks[0]['text'][:400]}..."

    def _answer_general_question(self, question: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Answer general questions"""
        # Use the most relevant chunks
        best_chunks = chunks[:2]  # Top 2 chunks

        combined_text = " ".join([chunk['text'] for chunk in best_chunks])

        # Truncate if too long
        if len(combined_text) > 500:
            combined_text = combined_text[:500] + "..."

        return f"Based on the relevant information found: {combined_text}"

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from question"""
        # Remove question words and common words
        question_words = {'what', 'is', 'are', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'which', 'who', 'define', 'definition', 'of'}
        words = re.findall(r'\b\w+\b', question.lower())
        key_terms = [word for word in words if word not in question_words and len(word) > 2]
        return key_terms

    def _extract_comparison_terms(self, question: str) -> List[str]:
        """Extract terms being compared"""
        # Look for patterns like "X vs Y", "X and Y", "between X and Y"
        question_lower = question.lower()

        # Pattern 1: "between X and Y"
        between_match = re.search(r'between ([^and]+) and ([^?]+)', question_lower)
        if between_match:
            return [between_match.group(1).strip(), between_match.group(2).strip()]

        # Pattern 2: "X vs Y" or "X versus Y"
        vs_match = re.search(r'([^vs]+)\s+(?:vs|versus)\s+([^?]+)', question_lower)
        if vs_match:
            return [vs_match.group(1).strip(), vs_match.group(2).strip()]

        # Pattern 3: Look for "and" connections
        and_match = re.search(r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)', question_lower)
        if and_match:
            return [and_match.group(1).strip(), and_match.group(2).strip()]

        # Fallback: extract key terms
        return self._extract_key_terms(question)

    def _extract_list_items(self, text: str) -> List[str]:
        """Extract potential list items from text"""
        items = []

        # Look for numbered lists
        numbered_items = re.findall(r'\d+\.\s*([^.\n]+)', text)
        items.extend([item.strip() for item in numbered_items])

        # Look for bulleted lists
        bulleted_items = re.findall(r'[•-]\s*([^.\n]+)', text)
        items.extend([item.strip() for item in bulleted_items])

        # Look for comma-separated lists after "include" or "such as"
        include_match = re.search(r'(?:include|such as):?\s*([^.]+)', text, re.IGNORECASE)
        if include_match:
            comma_items = [item.strip() for item in include_match.group(1).split(',')]
            items.extend(comma_items)

        return items[:10]  # Return max 10 items

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _combine_sentences(self, sentences: List[str]) -> str:
        """Combine sentences into a coherent answer"""
        if not sentences:
            return ""

        # Join sentences and clean up
        combined = " ".join(sentences)

        # Remove excessive whitespace
        combined = re.sub(r'\s+', ' ', combined)

        # Ensure proper capitalization
        if combined:
            combined = combined[0].upper() + combined[1:] if len(combined) > 1 else combined.upper()

        return combined.strip()

    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval quality"""
        if not chunks:
            return 0.0

        # Base confidence on similarity scores
        scores = [chunk['score'] for chunk in chunks]
        avg_score = sum(scores) / len(scores)

        # Normalize to 0-1 range
        confidence = min(avg_score * 2, 1.0)  # Multiply by 2 to boost confidence

        return round(confidence, 3)

    def update_settings(self, chunk_size: int, overlap: int, top_k: int):
        """Update RAG pipeline settings"""
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
