# 1. Start with the base Python image
FROM python:3.11-slim

# 2. Stay as ROOT to install system tools
USER root

# 3. Install the compiler (build-essential) and Git
# This is the "magic" step that fixes the Triton & Unsloth errors
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Create the Hugging Face user
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 5. Copy requirements and install them
# We stay as ROOT for a second here to ensure global permissions are clean
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the code
COPY --chown=user . .

# 7. Switch to USER for final execution (HF Security Requirement)
USER user

# 8. Launch the server (which triggers train.py via the startup event)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]