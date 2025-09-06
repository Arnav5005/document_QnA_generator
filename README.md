# document_QnA_generator


# Deployment Guide for RAG Document Q&A System

## Local Development Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Step-by-Step Setup

1. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. **Run the Application**
```bash
streamlit run main_app.py
```

3. **Access the Application**
- Open your browser
- Navigate to `http://localhost:8501`
- Start uploading documents and asking questions!

## Cloud Deployment Options

### 1. Streamlit Cloud (Free)

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

**Pros:**
- Free hosting
- Easy deployment
- Automatic updates from GitHub

**Cons:**
- Limited resources
- Public repositories only (free tier)

### 2. Heroku Deployment

**Additional Files Needed:**

Create `Procfile`:
```
web: streamlit run main_app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

**Deploy Steps:**
```bash
# Install Heroku CLI
# Create Heroku app
heroku create your-app-name

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git add .
git commit -m "Deploy RAG Q&A System"
git push heroku main
```

### 3. Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  rag-qa-system:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./temp:/app/temp
    environment:
      - PYTHONPATH=/app
```

**Deploy with Docker:**
```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8501
```

### 4. AWS EC2 Deployment

1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS
   - Select t2.micro (free tier eligible)
   - Configure security group to allow port 8501

2. **Setup on EC2**
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip -y

# Clone your repository
git clone https://github.com/your-username/rag-qa-system.git
cd rag-qa-system

# Install dependencies
pip3 install -r requirements.txt

# Install screen for background running
sudo apt install screen -y

# Run in background
screen -S rag-app
streamlit run main_app.py --server.port=8501 --server.address=0.0.0.0

# Detach from screen: Ctrl+A, then D
```

## Production Considerations

### 1. Performance Optimization

**Memory Management:**
```python
# Add to main_app.py
import gc
import psutil

def monitor_memory():
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    if memory_usage > 500:  # 500MB threshold
        gc.collect()
```

**File Size Limits:**
```python
# Increase for production
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TOTAL_FILES = 100
```

### 2. Security Enhancements

**File Validation:**
```python
import magic

def validate_file_type(file_path):
    mime = magic.from_file(file_path, mime=True)
    allowed_mimes = [
        'application/pdf',
        'text/plain',
        'text/html',
        'text/markdown'
    ]
    return mime in allowed_mimes
```

**Input Sanitization:**
```python
import html
import re

def sanitize_input(text):
    # Remove HTML tags
    text = html.escape(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1000]  # Limit length
```

### 3. Database Integration

For production, consider replacing in-memory storage:

**PostgreSQL + pgvector:**
```python
import psycopg2
from pgvector.psycopg2 import register_vector

class PostgreSQLVectorStore:
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
        register_vector(self.conn)
        self.setup_tables()

    def setup_tables(self):
        with self.conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    name TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID REFERENCES documents(id),
                    text TEXT,
                    embedding vector(384),
                    metadata JSONB
                )
            ''')
            self.conn.commit()
```

### 4. Monitoring and Logging

**Application Monitoring:**
```python
import logging
import time
from functools import wraps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_app.log'),
        logging.StreamHandler()
    ]
)

def log_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## Scaling Considerations

### 1. Horizontal Scaling

**Load Balancer Configuration (nginx):**
```nginx
upstream rag_app {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://rag_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. Caching Layer

**Redis Integration:**
```python
import redis
import pickle

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def get_cached_answer(self, question_hash):
        cached = self.redis_client.get(f"answer:{question_hash}")
        return pickle.loads(cached) if cached else None

    def cache_answer(self, question_hash, answer, ttl=3600):
        self.redis_client.setex(
            f"answer:{question_hash}", 
            ttl, 
            pickle.dumps(answer)
        )
```

### 3. Asynchronous Processing

**Celery Integration:**
```python
from celery import Celery

celery_app = Celery('rag_tasks', broker='redis://localhost:6379')

@celery_app.task
def process_document_async(document_path):
    # Process document in background
    processor = DocumentProcessor()
    return processor.extract_text(document_path)
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce chunk size
   - Implement file size limits
   - Add garbage collection

2. **Slow Performance**
   - Optimize vector operations
   - Add caching layer
   - Use async processing

3. **Deployment Failures**
   - Check Python version compatibility
   - Verify all dependencies
   - Test locally first

### Debug Mode

Enable debug logging:
```python
import streamlit as st

if st.secrets.get("DEBUG", False):
    import logging
    logging.basicConfig(level=logging.DEBUG)
```

## Support and Maintenance

### Health Checks

```python
def health_check():
    status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "components": {
            "vector_store": "ok",
            "document_processor": "ok",
            "rag_pipeline": "ok"
        }
    }
    return status
```

### Backup Strategy

```python
import shutil
import datetime

def backup_system():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_{timestamp}"
    shutil.copytree("rag_qa_system", backup_dir)
    return backup_dir
```

This deployment guide provides comprehensive instructions for getting your RAG Document Q&A System running in various environments, from local development to production deployment.
