
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import hashlib

# Import document processing libraries
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

import markdown
import html2text

class DocumentProcessor:
    """Handle document ingestion and text extraction"""

    def __init__(self):
        self.supported_formats = {
            '.pdf': self._extract_pdf_text,
            '.txt': self._extract_txt_text,
            '.md': self._extract_markdown_text,
            '.html': self._extract_html_text,
            '.htm': self._extract_html_text
        }
        self.chunk_size = 400
        self.overlap = 50

    def extract_text(self, file_path: str) -> str:
        """Extract text from document based on file extension"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")

        try:
            return self.supported_formats[extension](file_path)
        except Exception as e:
            raise RuntimeError(f"Error extracting text from {file_path.name}: {str(e)}")

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        text = ""

        # Try pypdf first (recommended)
        if PYPDF_AVAILABLE:
            try:
                with open(file_path, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text.strip()
            except Exception:
                pass

        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text.strip()
            except Exception:
                pass

        raise RuntimeError("No PDF processing library available. Install pypdf or PyPDF2.")

    def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from TXT files"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue

        raise RuntimeError(f"Could not decode text file with any supported encoding")

    def _extract_markdown_text(self, file_path: Path) -> str:
        """Extract text from Markdown files"""
        # Read markdown content
        md_content = self._extract_txt_text(file_path)

        # Convert to HTML first, then extract text
        html_content = markdown.markdown(md_content)

        if BS4_AVAILABLE:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        else:
            # Fallback: simple regex-based markdown stripping
            # Remove markdown formatting
            text = re.sub(r'#{1,6}\s+', '', md_content)  # Headers
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
            text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links
            text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
            text = re.sub(r'```[\s\S]*?```', '', text)  # Code blocks

            return text.strip()

    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML files"""
        html_content = self._extract_txt_text(file_path)

        if BS4_AVAILABLE:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator=' ', strip=True)
        else:
            # Fallback using html2text
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            return h.handle(html_content).strip()

    def chunk_text(self, text: str, chunk_size: Optional[int] = None, 
                   overlap: Optional[int] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.overlap

        # Clean and prepare text
        text = self._clean_text(text)

        # Split into sentences for better chunking
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence.split())

            # If adding this sentence would exceed chunk size, create new chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_sentence': i - len(current_chunk),
                    'end_sentence': i - 1,
                    'word_count': current_length
                })

                # Start new chunk with overlap
                overlap_sentences = max(1, min(overlap // 20, len(current_chunk)))
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                current_length = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_sentence': len(sentences) - len(current_chunk),
                'end_sentence': len(sentences) - 1,
                'word_count': current_length
            })

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page breaks and form feeds
        text = re.sub(r'[\f\r]', '', text)

        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)

        return text.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with NLTK or spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        return sentences

    def update_settings(self, chunk_size: int, overlap: int):
        """Update chunking settings"""
        self.chunk_size = chunk_size
        self.overlap = overlap
