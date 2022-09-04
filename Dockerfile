FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel 

# Install all dependencies and usefull apps
RUN apt-get update -y || echo
RUN apt-get install -y freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev git vim htop wget

# Fetch the repos
WORKDIR /workspace
RUN git clone https://github.com/NVlabs/ffhq-dataset.git
RUN git clone https://github.com/Mazurel/GPEN.git

# Setup GPEN repo
WORKDIR /workspace/GPEN
RUN rmdir DPR
RUN git clone https://github.com/Mazurel/DPR
RUN pip3 install -r requirements.txt

# Download the weights for GPEN
WORKDIR /workspace/GPEN/weights
RUN wget --quiet "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/model_ir_se50.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116170&Signature=jEyBslytwpWoh5DfKvYe2H31GgE%3D" -O model_ir_se50.pth

# Download ffhq-dataset manifest, so that download.py will work out of the box
WORKDIR /workspace/ffhq-dataset
RUN pip3 install gdown
# Download ffhq-dataset-v2.json:
RUN gdown https://drive.google.com/u/0/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA

WORKDIR /workspace
