# Start FROM Nvidia Tensorrt image
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Install linux packages
RUN apt update && apt install -y libgl1-mesa-dev

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install --no-cache -r requirements.txt
RUN pip install --no-cache torch torchvision -f https://download.pytorch.org/whl/torch_stable.html

# Install pytorch-quantization toolkit
#RUN pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

RUN apt-get update && apt-get install libturbojpeg

RUN pip install PyTurboJPEG
RUN pip install scikit-learn
RUN pip install piexif
RUN pip install ultralytics
RUN pip install streamlit

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install wget

RUN pip install opencv-python==4.8.0.74
RUN pip install pillow==10.3.0
RUN pip install requests==2.32.3
RUN pip install av
RUN pip install dill
RUN pip install ultralytics==8.2.41
# Create working directory
RUN mkdir -p /workspace
WORKDIR /workspace
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=100000"]