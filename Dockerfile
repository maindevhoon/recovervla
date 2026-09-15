FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive MUJOCO_GL=egl PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libegl1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/recovervla
COPY . .
RUN pip install --no-cache-dir -e '.[sim,train]'
CMD ["bash"]
