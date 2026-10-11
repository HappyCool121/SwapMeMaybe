FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG COMFY_COMMIT=7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217
ARG RIFE_COMMIT=5d8adbdd40e12c2c8f91930eff838aebe561c086
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LIVESWAP_DATA_DIR=/data/liveswap \
    COMFY_DIR=/opt/comfy RIFE_REPO=/opt/rife RIFE_WEIGHTS=/models/rife \
    COMFY_COMMIT=${COMFY_COMMIT} RIFE_COMMIT=${RIFE_COMMIT} HF_HOME=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/comfy && cd /opt/comfy && git remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git fetch --depth 1 origin ${COMFY_COMMIT} && git checkout FETCH_HEAD
RUN git init /opt/rife && cd /opt/rife && git remote add origin https://github.com/hzwer/ECCV2022-RIFE.git \
    && git fetch --depth 1 origin ${RIFE_COMMIT} && git checkout FETCH_HEAD
WORKDIR /opt/app
COPY pyproject.toml uv.lock README.md ./
COPY deploy/requirements-worker.txt ./deploy/requirements-worker.txt
RUN pip install -r /opt/comfy/requirements.txt 'opencv-python-headless<4.12' \
    && pip install --require-hashes -r deploy/requirements-worker.txt
COPY src ./src
RUN pip install --no-deps '.[worker]' && pip check
COPY profiles ./profiles
COPY deploy/entrypoint.py ./deploy/entrypoint.py
COPY comfy_nodes/liveswap_nodes /opt/comfy/custom_nodes/liveswap_nodes
COPY models/rife/flownet.pkl /models/rife/flownet.pkl
COPY deploy/loras /opt/comfy/models/loras
EXPOSE 8000
ENTRYPOINT ["python", "/opt/app/deploy/entrypoint.py"]
