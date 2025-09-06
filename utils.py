
import os
import streamlit as st
from pathlib import Path
import tempfile
import hashlib
import mimetypes
from typing import Optional

def setup_directories():
    """Setup necessary directories for the application"""
    temp_dir = Path(tempfile.gettempdir()) / "rag_qa_system"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

def validate_file(uploaded_file) -> bool:
    """Validate uploaded file"""
    if not uploaded_file:
        return False

    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File size too large. Maximum size is 10MB.")
        return False

    # Check file extension
    allowed_extensions = {'.pdf', '.txt', '.md', '.html', '.htm'}
    file_extension = Path(uploaded_file.name).suffix.lower()

    if file_extension not in allowed_extensions:
        st.error(f"Unsupported file type: {file_extension}")
        return False

    return True

def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content"""
    return hashlib.md5(file_content).hexdigest()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def get_mime_type(file_path: str) -> Optional[str]:
    """Get MIME type of file"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type

def clean_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    # Remove or replace unsafe characters
    import re
    cleaned = re.sub(r'[<>:"/\|?*]', '_', filename)
    return cleaned

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def highlight_text(text: str, terms: list) -> str:
    """Highlight search terms in text"""
    highlighted = text
    for term in terms:
        import re
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(f"**{term}**", highlighted)
    return highlighted

def export_chat_to_markdown(chat_history: list) -> str:
    """Export chat history to markdown format"""
    import time

    markdown_content = "# Document Q&A Session Export\n\n"
    markdown_content += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for i, exchange in enumerate(chat_history):
        markdown_content += f"## Question {i+1}\n"
        markdown_content += f"{exchange['question']}\n\n"
        markdown_content += f"**Answer:**\n{exchange['answer']}\n\n"

        if exchange.get('sources'):
            markdown_content += f"**Sources:**\n"
            for j, source in enumerate(exchange['sources']):
                markdown_content += f"{j+1}. {source['document']} (Score: {source['score']:.3f})\n"
                markdown_content += f"   {source['text'][:100]}...\n\n"

        markdown_content += "---\n\n"

    return markdown_content

class PerformanceTimer:
    """Simple performance timer"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        import time
        self.start_time = time.time()

    def stop(self):
        import time
        self.end_time = time.time()

    def elapsed(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def elapsed_str(self) -> str:
        elapsed = self.elapsed()
        return f"{elapsed:.2f}s"
