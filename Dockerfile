# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies with cleanup
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Download only specific NLTK data to save space
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" && \
    rm -rf /root/nltk_data/tokenizers/punkt/PY3 && \
    find /root/nltk_data -name "*.zip" -delete

# Copy project files (use .dockerignore to exclude unnecessary files)
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]
