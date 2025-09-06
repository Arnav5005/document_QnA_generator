
# Testing Examples and Sample Data for RAG Document Q&A System

import unittest
import tempfile
import os
from pathlib import Path
import json

# Import your system components
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor"""

    def setUp(self):
        self.processor = DocumentProcessor()
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        # Clean up test files
        import shutil
        shutil.rmtree(self.test_dir)

    def test_txt_extraction(self):
        """Test text file extraction"""
        test_content = "This is a test document.\nIt has multiple lines.\nWith various content."
        test_file = self.test_dir / "test.txt"

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        extracted = self.processor.extract_text(str(test_file))
        self.assertEqual(extracted.strip(), test_content)

    def test_markdown_extraction(self):
        """Test markdown file extraction"""
        markdown_content = """
# Test Document

This is a **bold** text and this is *italic*.

## Section 2

- Item 1
- Item 2
- Item 3

[Link](http://example.com)
"""
        test_file = self.test_dir / "test.md"

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        extracted = self.processor.extract_text(str(test_file))

        # Check that markdown formatting is removed
        self.assertNotIn('**', extracted)
        self.assertNotIn('##', extracted)
        self.assertIn('Test Document', extracted)
        self.assertIn('bold', extracted)

    def test_html_extraction(self):
        """Test HTML file extraction"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <script>console.log('test');</script>
    <style>body { color: red; }</style>
</head>
<body>
    <h1>Main Title</h1>
    <p>This is a paragraph with <strong>bold text</strong>.</p>
    <ul>
        <li>List item 1</li>
        <li>List item 2</li>
    </ul>
</body>
</html>
"""
        test_file = self.test_dir / "test.html"

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        extracted = self.processor.extract_text(str(test_file))

        # Check that HTML tags are removed but content remains
        self.assertNotIn('<h1>', extracted)
        self.assertNotIn('<script>', extracted)
        self.assertIn('Main Title', extracted)
        self.assertIn('paragraph', extracted)
        self.assertIn('List item 1', extracted)

    def test_chunking(self):
        """Test text chunking functionality"""
        long_text = " ".join([f"This is sentence {i}." for i in range(100)])

        chunks = self.processor.chunk_text(long_text, chunk_size=50, overlap=10)

        # Check that chunks were created
        self.assertGreater(len(chunks), 1)

        # Check chunk structure
        for chunk in chunks:
            self.assertIn('text', chunk)
            self.assertIn('word_count', chunk)
            self.assertLessEqual(chunk['word_count'], 60)  # chunk_size + some tolerance

    def test_unsupported_format(self):
        """Test handling of unsupported file formats"""
        test_file = self.test_dir / "test.xyz"

        with open(test_file, 'w') as f:
            f.write("test content")

        with self.assertRaises(ValueError):
            self.processor.extract_text(str(test_file))

class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore"""

    def setUp(self):
        self.vector_store = VectorStore()

    def test_add_document(self):
        """Test adding a document to vector store"""
        chunks = [
            {'text': 'This is the first chunk about machine learning.', 'word_count': 8},
            {'text': 'This is the second chunk about artificial intelligence.', 'word_count': 9}
        ]

        doc_id = self.vector_store.add_document("test_doc.txt", chunks)

        # Check that document was added
        self.assertIsNotNone(doc_id)
        self.assertIn(doc_id, self.vector_store.documents)
        self.assertEqual(len(self.vector_store.chunks), 2)

    def test_search_functionality(self):
        """Test search functionality"""
        # Add test documents
        chunks1 = [{'text': 'Machine learning is a subset of artificial intelligence.', 'word_count': 9}]
        chunks2 = [{'text': 'Deep learning uses neural networks with multiple layers.', 'word_count': 9}]

        self.vector_store.add_document("ml_doc.txt", chunks1)
        self.vector_store.add_document("dl_doc.txt", chunks2)

        # Search for machine learning
        results = self.vector_store.search("machine learning", top_k=2)

        # Check results
        self.assertGreater(len(results), 0)
        self.assertIn('score', results[0])
        self.assertIn('text', results[0])
        self.assertIn('document', results[0])

    def test_remove_document(self):
        """Test document removal"""
        chunks = [{'text': 'Test content for removal.', 'word_count': 4}]
        doc_id = self.vector_store.add_document("temp_doc.txt", chunks)

        # Verify document exists
        self.assertIn(doc_id, self.vector_store.documents)

        # Remove document
        self.vector_store.remove_document(doc_id)

        # Verify document is removed
        self.assertNotIn(doc_id, self.vector_store.documents)

    def test_empty_search(self):
        """Test search on empty vector store"""
        results = self.vector_store.search("test query")
        self.assertEqual(len(results), 0)

class TestRAGPipeline(unittest.TestCase):
    """Test cases for RAG Pipeline"""

    def setUp(self):
        self.vector_store = VectorStore()
        self.rag_pipeline = RAGPipeline(self.vector_store)

        # Add sample documents
        self._add_sample_documents()

    def _add_sample_documents(self):
        """Add sample documents for testing"""
        ml_chunks = [
            {'text': 'Machine learning is a method of data analysis that automates analytical model building.', 'word_count': 15},
            {'text': 'It is a branch of artificial intelligence based on the idea that systems can learn from data.', 'word_count': 18}
        ]

        dl_chunks = [
            {'text': 'Deep learning is part of a broader family of machine learning methods based on neural networks.', 'word_count': 17},
            {'text': 'Deep learning architectures such as deep neural networks have been applied to many fields.', 'word_count': 15}
        ]

        self.vector_store.add_document("machine_learning.txt", ml_chunks)
        self.vector_store.add_document("deep_learning.txt", dl_chunks)

    def test_answer_generation(self):
        """Test answer generation"""
        question = "What is machine learning?"
        result = self.rag_pipeline.answer_question(question)

        # Check result structure
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertIn('confidence', result)

        # Check that answer is not empty
        self.assertGreater(len(result['answer']), 0)

        # Check that sources were found
        self.assertGreater(len(result['sources']), 0)

    def test_no_relevant_documents(self):
        """Test handling when no relevant documents are found"""
        question = "What is quantum computing?"
        result = self.rag_pipeline.answer_question(question)

        # Should return appropriate message when no relevant docs found
        self.assertIn("couldn't find relevant information", result['answer'])
        self.assertEqual(len(result['sources']), 0)
        self.assertEqual(result['confidence'], 0.0)

    def test_question_type_detection(self):
        """Test question type detection"""
        test_questions = [
            ("What is machine learning?", "definition"),
            ("What are the types of neural networks?", "list"),
            ("How does deep learning work?", "how"),
            ("Why is AI important?", "why"),
            ("What's the difference between ML and DL?", "comparison")
        ]

        for question, expected_type in test_questions:
            detected_type = self.rag_pipeline._detect_question_type(question.lower())
            self.assertEqual(detected_type, expected_type, 
                           f"Failed for question: {question}")

# Sample data for testing
SAMPLE_DOCUMENTS = {
    "machine_learning_basics.md": """
# Machine Learning Basics

## Introduction
Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train algorithms that can make predictions or classify data. Examples include:
- Linear regression
- Decision trees
- Support vector machines
- Neural networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in data without labeled examples. Common techniques include:
- Clustering (K-means, hierarchical)
- Dimensionality reduction (PCA, t-SNE)
- Association rules

### Reinforcement Learning
Reinforcement learning involves an agent learning to make decisions through trial and error, receiving rewards or penalties for actions.

## Applications
- Image recognition
- Natural language processing
- Recommendation systems
- Autonomous vehicles
- Medical diagnosis
""",

    "deep_learning_guide.md": """
# Deep Learning Guide

## What is Deep Learning?
Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to progressively extract higher-level features from raw input.

## Neural Network Basics
A neural network consists of:
- Input layer: Receives the raw data
- Hidden layers: Process the data through weighted connections
- Output layer: Produces the final result

## Popular Architectures

### Convolutional Neural Networks (CNNs)
- Primarily used for image processing
- Use convolution operations to detect features
- Applications: image classification, object detection

### Recurrent Neural Networks (RNNs)
- Designed for sequential data
- Can remember information from previous inputs
- Applications: language modeling, time series prediction

### Transformers
- Attention-based architecture
- Excellent for natural language processing
- Examples: BERT, GPT, T5

## Training Process
1. Forward propagation: Data flows through network
2. Loss calculation: Compare output with expected result
3. Backpropagation: Update weights to minimize loss
4. Repeat until convergence

## Challenges
- Requires large datasets
- Computationally expensive
- Prone to overfitting
- Black box nature (interpretability)
""",

    "rag_systems.txt": """
Retrieval-Augmented Generation (RAG) Systems

RAG is an AI framework that combines information retrieval with text generation to produce more accurate and contextual responses. The system works by first retrieving relevant information from a knowledge base or document collection, then using that information to generate responses.

Key Components:
1. Document Ingestion: Process and store documents in a searchable format
2. Text Chunking: Break documents into meaningful segments
3. Embedding Generation: Convert text chunks into vector representations
4. Vector Storage: Store embeddings in a database for efficient retrieval
5. Query Processing: Convert user questions into searchable vectors
6. Retrieval: Find most relevant chunks using similarity search
7. Generation: Use retrieved context to generate comprehensive answers

Benefits of RAG:
- Improved accuracy by grounding responses in factual information
- Reduced hallucination compared to pure generative models
- Ability to incorporate up-to-date information
- Transparency through source citations
- Cost-effective compared to fine-tuning large models

Applications:
- Question-answering systems
- Customer support chatbots
- Document analysis tools
- Research assistants
- Knowledge management systems

Technical Considerations:
- Chunk size and overlap strategies
- Embedding model selection
- Vector similarity metrics
- Retrieval ranking algorithms
- Context window management
- Response generation techniques
"""
}

def create_sample_files(output_dir: Path):
    """Create sample files for testing"""
    output_dir.mkdir(exist_ok=True)

    for filename, content in SAMPLE_DOCUMENTS.items():
        file_path = output_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())

    print(f"Created {len(SAMPLE_DOCUMENTS)} sample files in {output_dir}")

# Test scenarios for comprehensive testing
TEST_SCENARIOS = [
    {
        "name": "Basic Q&A Test",
        "documents": ["machine_learning_basics.md"],
        "questions": [
            "What is machine learning?",
            "What are the types of machine learning?",
            "What is supervised learning?",
            "Give me examples of supervised learning algorithms."
        ],
        "expected_keywords": ["artificial intelligence", "labeled data", "regression", "classification"]
    },
    {
        "name": "Deep Learning Focus",
        "documents": ["deep_learning_guide.md"],
        "questions": [
            "What is deep learning?",
            "What are CNNs used for?",
            "How do neural networks work?",
            "What are the challenges of deep learning?"
        ],
        "expected_keywords": ["neural networks", "layers", "image processing", "overfitting"]
    },
    {
        "name": "RAG System Understanding",
        "documents": ["rag_systems.txt"],
        "questions": [
            "What is RAG?",
            "How does RAG work?",
            "What are the benefits of RAG?",
            "What are RAG applications?"
        ],
        "expected_keywords": ["retrieval", "generation", "embeddings", "vector storage"]
    },
    {
        "name": "Multi-Document Queries",
        "documents": ["machine_learning_basics.md", "deep_learning_guide.md"],
        "questions": [
            "How is deep learning related to machine learning?",
            "What's the difference between traditional ML and deep learning?",
            "What are neural networks?",
            "Compare supervised learning and deep learning approaches."
        ],
        "expected_keywords": ["subset", "neural networks", "multiple layers", "algorithms"]
    }
]

class IntegrationTest:
    """Integration test runner"""

    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.rag_pipeline = RAGPipeline(self.vector_store)

    def run_scenario(self, scenario: dict) -> dict:
        """Run a test scenario"""
        print(f"\nRunning scenario: {scenario['name']}")

        # Load documents
        for doc_name in scenario['documents']:
            doc_path = self.test_dir / doc_name
            if doc_path.exists():
                try:
                    text = self.processor.extract_text(str(doc_path))
                    chunks = self.processor.chunk_text(text)
                    self.vector_store.add_document(doc_name, chunks)
                    print(f"  ✓ Loaded {doc_name} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"  ✗ Failed to load {doc_name}: {e}")

        # Test questions
        results = []
        for question in scenario['questions']:
            try:
                result = self.rag_pipeline.answer_question(question)
                results.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources_count': len(result['sources']),
                    'confidence': result['confidence'],
                    'success': True
                })
                print(f"  ✓ {question} -> {result['confidence']:.2f} confidence")
            except Exception as e:
                results.append({
                    'question': question,
                    'error': str(e),
                    'success': False
                })
                print(f"  ✗ {question} -> Error: {e}")

        return {
            'scenario': scenario['name'],
            'results': results,
            'success_rate': sum(1 for r in results if r['success']) / len(results)
        }

    def run_all_scenarios(self) -> dict:
        """Run all test scenarios"""
        all_results = []

        for scenario in TEST_SCENARIOS:
            # Reset vector store for each scenario
            self.vector_store = VectorStore()
            self.rag_pipeline = RAGPipeline(self.vector_store)

            result = self.run_scenario(scenario)
            all_results.append(result)

        # Calculate overall statistics
        total_questions = sum(len(r['results']) for r in all_results)
        total_success = sum(sum(1 for q in r['results'] if q['success']) for r in all_results)

        summary = {
            'total_scenarios': len(all_results),
            'total_questions': total_questions,
            'total_success': total_success,
            'overall_success_rate': total_success / total_questions if total_questions > 0 else 0,
            'scenario_results': all_results
        }

        return summary

def run_tests():
    """Main test runner function"""
    print("🧪 Running RAG Document Q&A System Tests")

    # Create test directory and sample files
    test_dir = Path("test_data")
    create_sample_files(test_dir)

    # Run unit tests
    print("\n📋 Running Unit Tests...")
    unittest.main(verbosity=2, exit=False)

    # Run integration tests
    print("\n🔗 Running Integration Tests...")
    integration_test = IntegrationTest(test_dir)
    results = integration_test.run_all_scenarios()

    # Print summary
    print("\n📊 Test Summary:")
    print(f"Scenarios: {results['total_scenarios']}")
    print(f"Questions: {results['total_questions']}")
    print(f"Success Rate: {results['overall_success_rate']:.1%}")

    # Detailed results
    for scenario_result in results['scenario_results']:
        print(f"\n{scenario_result['scenario']}: {scenario_result['success_rate']:.1%} success rate")

    return results

if __name__ == "__main__":
    run_tests()
