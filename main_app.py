import streamlit as st
import os
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any, Optional
import hashlib
import time

# Import our RAG components
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from utils import setup_directories, validate_file

class DocumentQAApp:
    def __init__(self):
        self.setup_app_config()
        self.initialize_components()
        self.setup_directories()

    def setup_app_config(self):
        st.set_page_config(
            page_title="Document Q&A System",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize session state
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'vector_store_ready' not in st.session_state:
            st.session_state.vector_store_ready = False
        if 'processing' not in st.session_state:
            st.session_state.processing = False

    def initialize_components(self):
        """Initialize RAG system components"""
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.rag_pipeline = RAGPipeline(self.vector_store)

    def setup_directories(self):
        """Setup necessary directories for file storage"""
        self.temp_dir = Path(tempfile.gettempdir()) / "rag_qa_system"
        self.temp_dir.mkdir(exist_ok=True)

    def render_header(self):
        """Render application header"""
        st.title("🤖 Document Q&A System")
        st.markdown("### Intelligent Question Answering with Retrieval-Augmented Generation")
        st.markdown("---")

    def render_sidebar(self):
        """Render sidebar with document management"""
        st.sidebar.header("📁 Document Management")

        # File upload section
        uploaded_files = st.sidebar.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'md', 'html'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, Markdown, HTML"
        )

        if uploaded_files and not st.session_state.processing:
            if st.sidebar.button("Process Documents", type="primary"):
                self.process_uploaded_files(uploaded_files)

        # Document list
        if st.session_state.documents:
            st.sidebar.subheader("📋 Uploaded Documents")
            for i, doc in enumerate(st.session_state.documents):
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.write(f"📄 {doc['name']}")
                    st.caption(f"Chunks: {doc['chunks']} | Size: {doc['size']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete document"):
                        self.delete_document(i)

        # Settings
        st.sidebar.subheader("⚙️ Settings")
        chunk_size = st.sidebar.slider("Chunk Size (tokens)", 200, 800, 400)
        overlap = st.sidebar.slider("Chunk Overlap", 20, 100, 50)
        top_k = st.sidebar.slider("Retrieved Chunks", 3, 10, 5)

        # Update settings if changed
        if (chunk_size != self.rag_pipeline.chunk_size or 
            overlap != self.rag_pipeline.overlap or 
            top_k != self.rag_pipeline.top_k):
            self.rag_pipeline.update_settings(chunk_size, overlap, top_k)

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and add to vector store"""
        st.session_state.processing = True

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        try:
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")

                # Validate file
                if not validate_file(uploaded_file):
                    st.sidebar.error(f"Invalid file: {uploaded_file.name}")
                    continue

                # Save temporary file
                temp_file_path = self.temp_dir / uploaded_file.name
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process document
                text_content = self.doc_processor.extract_text(str(temp_file_path))
                chunks = self.doc_processor.chunk_text(text_content)

                # Add to vector store
                doc_id = self.vector_store.add_document(uploaded_file.name, chunks)

                # Update session state
                st.session_state.documents.append({
                    'id': doc_id,
                    'name': uploaded_file.name,
                    'chunks': len(chunks),
                    'size': f"{len(text_content)} chars"
                })

                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))

            st.session_state.vector_store_ready = True
            status_text.text("✅ All documents processed successfully!")

        except Exception as e:
            st.sidebar.error(f"Error processing documents: {str(e)}")
        finally:
            st.session_state.processing = False
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

    def delete_document(self, doc_index):
        """Delete a document from the system"""
        doc = st.session_state.documents[doc_index]
        self.vector_store.remove_document(doc['id'])
        st.session_state.documents.pop(doc_index)

        if not st.session_state.documents:
            st.session_state.vector_store_ready = False

        st.experimental_rerun()

    def render_main_interface(self):
        """Render main Q&A interface"""
        if not st.session_state.vector_store_ready:
            self.render_welcome_screen()
        else:
            self.render_qa_interface()

    def render_welcome_screen(self):
        """Render welcome screen when no documents are loaded"""
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 50px;">
                <h2>Welcome to Document Q&A System</h2>
                <p style="font-size: 18px; color: #666;">
                    Upload your documents using the sidebar to get started with intelligent question answering.
                </p>
                <br>
                <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h4>Supported Features:</h4>
                    <ul style="text-align: left; display: inline-block;">
                        <li>📄 Multiple document formats (PDF, TXT, MD, HTML)</li>
                        <li>🔍 Intelligent text chunking and retrieval</li>
                        <li>💬 Natural language question answering</li>
                        <li>📊 Source citations and confidence scores</li>
                        <li>💾 Chat history and export functionality</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def render_qa_interface(self):
        """Render Q&A interface when documents are loaded"""

        # Flag to clear input safely before widget creation
        if 'clear_question' not in st.session_state:
            st.session_state.clear_question = False

        # Set input default value based on flag
        input_value = "" if st.session_state.clear_question else st.session_state.get('question_input', "")

        # Reset flag immediately after consuming it
        if st.session_state.clear_question:
            st.session_state.clear_question = False

        # Display chat history
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for i, exchange in enumerate(st.session_state.chat_history):
                    st.markdown(f"**🙋 Question {i+1}:**")
                    st.markdown(f"> {exchange['question']}")
                    st.markdown(f"**🤖 Answer:**")
                    st.markdown(exchange['answer'])
                    if exchange.get('sources'):
                        with st.expander(f"📚 Sources & Citations ({len(exchange['sources'])} chunks)"):
                            for j, source in enumerate(exchange['sources'][:3]):
                                st.markdown(f"**Source {j+1}** (Score: {source['score']:.3f})")
                                st.markdown(f"*From: {source['document']}*")
                                st.code(source['text'][:200] + "..." if len(source['text']) > 200 else source['text'])
                    st.markdown("---")

        # Question input form with explicit value
        st.markdown("### 💭 Ask a Question")
        with st.form("question_form"):
            question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know about your documents?",
                key="question_input",
                value=input_value
            )
            ask_button = st.form_submit_button("Ask")

        if ask_button and question.strip():
            self.process_question(question.strip())
            # Set flag to clear input on next rerun
            st.session_state.clear_question = True

        # Export and clear history buttons
        if st.session_state.chat_history:
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("📥 Export Chat"):
                    self.export_chat_history()
            with col2:
                if st.button("🗑️ Clear History"):
                    st.session_state.chat_history = []
                    # No rerun call needed; state change forces rerun


    def process_question(self, question: str):
        """Process user question and generate answer safely"""

        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                # Generate answer using RAG pipeline
                result = self.rag_pipeline.answer_question(question)

                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'timestamp': time.time()
                })

                # DO NOT assign to st.session_state.question_input here to avoid error

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")



    def process_question(self, question: str):
        """Process user question and generate answer"""
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                # Generate answer using RAG pipeline
                result = self.rag_pipeline.answer_question(question)

                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'timestamp': time.time()
                })

                # Clear input and rerun
                st.session_state.question_input = ""
                st.experimental_rerun()

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

    def export_chat_history(self):
        """Export chat history as downloadable file"""
        if not st.session_state.chat_history:
            return

        # Create export content
        export_content = "# Document Q&A Session Export\n\n"
        export_content += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for i, exchange in enumerate(st.session_state.chat_history):
            export_content += f"## Question {i+1}\n"
            export_content += f"{exchange['question']}\n\n"
            export_content += f"**Answer:**\n{exchange['answer']}\n\n"

            if exchange.get('sources'):
                export_content += f"**Sources:**\n"
                for j, source in enumerate(exchange['sources']):
                    export_content += f"{j+1}. {source['document']} (Score: {source['score']:.3f})\n"
                    export_content += f"   {source['text'][:100]}...\n\n"

            export_content += "---\n\n"

        # Provide download
        st.download_button(
            label="📄 Download as Markdown",
            data=export_content,
            file_name=f"qa_session_{int(time.time())}.md",
            mime="text/markdown"
        )

    def run(self):
        """Run the Streamlit application"""
        self.render_header()
        self.render_sidebar()
        self.render_main_interface()

# Main entry point
if __name__ == "__main__":
    app = DocumentQAApp()
    app.run()
