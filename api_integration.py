
# API Integration Examples for RAG Document Q&A System
# This file shows how to integrate with various LLM APIs for production use

import os
import openai
import requests
import json
from typing import Dict, List, Any, Optional
import streamlit as st

class LLMIntegration:
    """Base class for LLM integrations"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_answer(self, question: str, context: str) -> str:
        raise NotImplementedError

class OpenAIIntegration(LLMIntegration):
    """OpenAI GPT integration"""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key)
        openai.api_key = api_key
        self.model = model

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
        You are a helpful assistant that answers questions based on the provided context.
        Use only the information from the context to answer the question.
        If the context doesn't contain enough information, say so.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return []

class AnthropicIntegration(LLMIntegration):
    """Anthropic Claude integration"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"

    def generate_answer(self, question: str, context: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        prompt = f"""
        Based on the following context, please answer the question. If the context doesn't contain 
        sufficient information to answer the question, please say so.

        Context:
        {context}

        Question: {question}
        """

        data = {
            "model": self.model,
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class HuggingFaceIntegration(LLMIntegration):
    """Hugging Face integration for open-source models"""

    def __init__(self, api_key: str, model: str = "microsoft/DialoGPT-medium"):
        super().__init__(api_key)
        self.model = model
        self.base_url = f"https://api-inference.huggingface.co/models/{model}"

    def generate_answer(self, question: str, context: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}

        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.3,
                "top_p": 0.9
            }
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").replace(prompt, "").strip()
            return "No response generated"
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class LocalLLMIntegration(LLMIntegration):
    """Integration with local LLM using Ollama"""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate_answer(self, question: str, context: str) -> str:
        url = f"{self.base_url}/api/generate"

        prompt = f"""
        Context: {context}

        Question: {question}

        Based on the context above, please provide a comprehensive answer to the question.
        If the context doesn't contain enough information, please indicate that.

        Answer:
        """

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 500
            }
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response generated")
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class EmbeddingIntegration:
    """Enhanced embedding integration"""

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key

        if provider == "openai" and api_key:
            openai.api_key = api_key

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings based on provider"""
        if self.provider == "openai":
            return self._openai_embeddings(texts)
        elif self.provider == "sentence-transformers":
            return self._sentence_transformer_embeddings(texts)
        elif self.provider == "huggingface":
            return self._huggingface_embeddings(texts)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate OpenAI embeddings"""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            st.error(f"OpenAI embedding error: {str(e)}")
            return []

    def _sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate sentence transformer embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts)
            return embeddings.tolist()
        except ImportError:
            st.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            return []
        except Exception as e:
            st.error(f"Sentence transformer error: {str(e)}")
            return []

    def _huggingface_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate Hugging Face embeddings"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

        try:
            response = requests.post(url, headers=headers, json={"inputs": texts})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Hugging Face embedding error: {str(e)}")
            return []

# Enhanced RAG Pipeline with LLM Integration
class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with LLM integration"""

    def __init__(self, vector_store, llm_provider: str = "rule-based", api_key: Optional[str] = None):
        self.vector_store = vector_store
        self.llm_provider = llm_provider

        # Initialize LLM integration
        if llm_provider == "openai" and api_key:
            self.llm = OpenAIIntegration(api_key)
        elif llm_provider == "anthropic" and api_key:
            self.llm = AnthropicIntegration(api_key)
        elif llm_provider == "huggingface" and api_key:
            self.llm = HuggingFaceIntegration(api_key)
        elif llm_provider == "local":
            self.llm = LocalLLMIntegration()
        else:
            self.llm = None  # Use rule-based approach

        # Initialize embeddings
        self.embedding_provider = EmbeddingIntegration(
            provider="sentence-transformers" if not api_key else "openai",
            api_key=api_key
        )

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Generate answer using enhanced RAG pipeline"""
        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.search(question, top_k=top_k)

        if not relevant_chunks:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'confidence': 0.0
            }

        # Combine context from chunks
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])

        # Generate answer
        if self.llm:
            # Use LLM for answer generation
            answer = self.llm.generate_answer(question, context)
        else:
            # Fallback to rule-based approach
            answer = self._rule_based_answer(question, context, relevant_chunks)

        # Calculate confidence
        confidence = self._calculate_confidence(relevant_chunks)

        return {
            'answer': answer,
            'sources': relevant_chunks,
            'confidence': confidence
        }

    def _rule_based_answer(self, question: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Fallback rule-based answer generation"""
        # Import the original RAG pipeline logic
        from rag_pipeline import RAGPipeline

        original_pipeline = RAGPipeline(self.vector_store)
        return original_pipeline._generate_answer(question, chunks)

    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score"""
        if not chunks:
            return 0.0

        scores = [chunk['score'] for chunk in chunks]
        avg_score = sum(scores) / len(scores)

        # Normalize confidence
        confidence = min(avg_score * 1.5, 1.0)
        return round(confidence, 3)

# Configuration management
class APIConfig:
    """API configuration management"""

    @staticmethod
    def load_from_secrets():
        """Load API keys from Streamlit secrets"""
        config = {}

        try:
            config['openai_api_key'] = st.secrets.get("OPENAI_API_KEY")
            config['anthropic_api_key'] = st.secrets.get("ANTHROPIC_API_KEY")
            config['huggingface_api_key'] = st.secrets.get("HUGGINGFACE_API_KEY")
        except Exception:
            pass

        return config

    @staticmethod
    def load_from_env():
        """Load API keys from environment variables"""
        return {
            'openai_api_key': os.getenv("OPENAI_API_KEY"),
            'anthropic_api_key': os.getenv("ANTHROPIC_API_KEY"),
            'huggingface_api_key': os.getenv("HUGGINGFACE_API_KEY")
        }

# Usage example in main application
def integrate_llm_in_app():
    """Example of how to integrate LLM in the main app"""

    # Load configuration
    config = APIConfig.load_from_secrets()

    # LLM selection in sidebar
    st.sidebar.subheader("🤖 LLM Configuration")

    llm_options = ["Rule-based", "OpenAI GPT", "Anthropic Claude", "Hugging Face", "Local (Ollama)"]
    selected_llm = st.sidebar.selectbox("Select LLM Provider", llm_options)

    # API key input based on selection
    api_key = None
    if selected_llm == "OpenAI GPT":
        api_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                       value=config.get('openai_api_key', ''))
    elif selected_llm == "Anthropic Claude":
        api_key = st.sidebar.text_input("Anthropic API Key", type="password",
                                       value=config.get('anthropic_api_key', ''))
    elif selected_llm == "Hugging Face":
        api_key = st.sidebar.text_input("Hugging Face API Key", type="password",
                                       value=config.get('huggingface_api_key', ''))

    # Map selection to provider
    provider_map = {
        "Rule-based": "rule-based",
        "OpenAI GPT": "openai",
        "Anthropic Claude": "anthropic",
        "Hugging Face": "huggingface",
        "Local (Ollama)": "local"
    }

    provider = provider_map[selected_llm]

    # Initialize enhanced RAG pipeline
    if 'enhanced_rag' not in st.session_state or st.session_state.get('llm_provider') != provider:
        st.session_state.enhanced_rag = EnhancedRAGPipeline(
            vector_store=st.session_state.vector_store,
            llm_provider=provider,
            api_key=api_key
        )
        st.session_state.llm_provider = provider

    return st.session_state.enhanced_rag

# Example of advanced features
class AdvancedFeatures:
    """Advanced features for the RAG system"""

    @staticmethod
    def multi_document_synthesis(question: str, document_answers: List[Dict[str, Any]]) -> str:
        """Synthesize answers from multiple documents"""
        # Combine information from multiple sources
        combined_context = "\n".join([
            f"From {ans['document']}: {ans['answer']}" 
            for ans in document_answers
        ])

        # Generate synthesized answer
        synthesis_prompt = f"""
        Based on the following information from multiple documents, provide a comprehensive answer:

        {combined_context}

        Question: {question}

        Synthesized Answer:
        """

        return synthesis_prompt  # Would be processed by LLM

    @staticmethod
    def generate_followup_questions(answer: str, context: str) -> List[str]:
        """Generate relevant follow-up questions"""
        # Simple rule-based follow-up generation
        followups = []

        answer_lower = answer.lower()

        if "because" in answer_lower or "due to" in answer_lower:
            followups.append("Can you explain this in more detail?")

        if "types" in answer_lower or "kinds" in answer_lower:
            followups.append("What are some examples of each type?")

        if "process" in answer_lower or "steps" in answer_lower:
            followups.append("What happens if one of these steps fails?")

        # Always add generic follow-ups
        followups.extend([
            "What are the implications of this?",
            "Are there any limitations or exceptions?",
            "How does this relate to other concepts?"
        ])

        return followups[:3]  # Return top 3

    @staticmethod
    def answer_quality_assessment(answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality of the generated answer"""
        quality_score = 0.0
        factors = {}

        # Length check
        if 50 <= len(answer) <= 500:
            quality_score += 0.2
            factors['length'] = 'appropriate'
        else:
            factors['length'] = 'too_short' if len(answer) < 50 else 'too_long'

        # Source diversity
        unique_docs = len(set(source['document'] for source in sources))
        if unique_docs > 1:
            quality_score += 0.3
            factors['source_diversity'] = 'high'
        else:
            factors['source_diversity'] = 'low'

        # Confidence from sources
        avg_confidence = sum(source['score'] for source in sources) / len(sources) if sources else 0
        if avg_confidence > 0.5:
            quality_score += 0.3
            factors['source_confidence'] = 'high'
        else:
            factors['source_confidence'] = 'low'

        # Completeness check (simple heuristic)
        if not any(phrase in answer.lower() for phrase in ['not found', 'no information', 'cannot answer']):
            quality_score += 0.2
            factors['completeness'] = 'complete'
        else:
            factors['completeness'] = 'incomplete'

        return {
            'overall_score': round(quality_score, 2),
            'factors': factors,
            'recommendations': get_quality_recommendations(factors)
        }

def get_quality_recommendations(factors: Dict[str, str]) -> List[str]:
    """Get recommendations for improving answer quality"""
    recommendations = []

    if factors.get('length') == 'too_short':
        recommendations.append("Try asking for more detailed information")
    elif factors.get('length') == 'too_long':
        recommendations.append("Try asking more specific questions")

    if factors.get('source_diversity') == 'low':
        recommendations.append("Upload more documents for diverse perspectives")

    if factors.get('source_confidence') == 'low':
        recommendations.append("Try rephrasing your question or uploading more relevant documents")

    if factors.get('completeness') == 'incomplete':
        recommendations.append("The available documents may not contain sufficient information for this question")

    return recommendations
