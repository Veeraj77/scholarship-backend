# Dockerfile

# 1. Start from a standard Python image
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /code

# 3. Set Hugging Face cache to a writable directory
#    This is CRITICAL for models to download!
ENV HF_HOME=/tmp
ENV TRANSFORMERS_CACHE=/tmp

# 4. Copy your requirements file and install libraries
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy your entire project (api.py, faiss_index folder, etc.)
COPY . /code

# 6. Expose the port Hugging Face expects
EXPOSE 7860

# 7. The command to run your app
#    We tell uvicorn to run on host 0.0.0.0 and port 7860.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]