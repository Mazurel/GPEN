FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel 

# Install all dependencies and usefull apps
RUN apt-get update -y || echo # The || echo is needed for ignoring update errors
RUN apt-get install -y freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev libglfw3-dev libgles2-mesa-dev git vim htop wget && \
    pip3 install pydrive2 requests

# Fetch the repos
WORKDIR /workspace
RUN git clone -b pydrive https://github.com/jeremyfix/ffhq-dataset && \
    git clone --recurse-submodules https://github.com/Mazurel/GPEN.git

# Setup GPEN repo
WORKDIR /workspace/GPEN
RUN pip3 install -r requirements.txt && \
    wget --quiet "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/model_ir_se50.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116170&Signature=jEyBslytwpWoh5DfKvYe2H31GgE%3D" -O weights/model_ir_se50.pth && \
    wget --quiet "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth?OSSAccessKeyId=LTAI4G6bfnyW4TA4wFUXTYBe&Expires=1961116208&Signature=hBgvVvKVSNGeXqT8glG%2Bd2t2OKc%3D" -O weights/GPEN-BFR-512.pth

WORKDIR /workspace
