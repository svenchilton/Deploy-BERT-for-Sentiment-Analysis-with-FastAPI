# Start with a PyTorch base image
# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Start with a miniconda base image
FROM continuumio/miniconda3:4.10.3

# Install git and wget
RUN apt update && apt install -y git wget

# Clone and enter the app repo
WORKDIR /
RUN git clone https://github.com/svenchilton/Deploy-BERT-for-Sentiment-Analysis-with-FastAPI.git
WORKDIR /Deploy-BERT-for-Sentiment-Analysis-with-FastAPI/

# Install the dependencies 
RUN conda install -y pytorch -c pytorch
RUN pip install pyngrok nest_asyncio httpie transformers fastapi uvicorn

# Download the pretrained model
RUN ./bin/download_model

# Expose Port 8000
EXPOSE 8000

# Start the server
CMD python sentiment_analyzer/api.py &