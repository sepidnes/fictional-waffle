# FROM python:3.11-slim

# WORKDIR /app

# # Install system dependencies required for unstructured and other packages
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     libmagic1 \
#     poppler-utils \
#     tesseract-ocr \
#     libreoffice \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements and lock file first to leverage Docker cache
# COPY requirements.txt ./
# COPY uv.lock ./

# # Install Python dependencies with version pinning
# RUN pip3 install --no-cache-dir -r requirements.txt

# # Specifically install packages that might be missing
# RUN pip3 install "unstructured[all]" python-docx html2text beautifulsoup4 arxiv>=1.4.7

# # Copy application code
# COPY . .

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH=/app

# # Expose port for Chainlit
# EXPOSE 8000

# # Start the application
# CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]

# # Copy application code
# COPY . .

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH=/app

# # Expose port for Chainlit
# EXPOSE 8000

# # Start the application
# CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
# Get a distribution that has uv already installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Add user - this is the user that will run the app
# If you do not set user, the app will run as root (undesirable)
RUN useradd -m -u 1000 user
USER user

# Set the home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH        

ENV UVICORN_WS_PROTOCOL=websockets

# Set the working directory
WORKDIR $HOME/app

# Copy the app to the container
COPY --chown=user . $HOME/app

# Install the dependencies
# RUN uv sync --frozen
RUN uv sync

# Expose the port
EXPOSE 7860

# Run the app
CMD ["uv", "run", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]