FROM rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0
WORKDIR /home/ubuntu
ARG COMFY_VERSION="v0.36.0"
ENV VIRTUAL_ENV=/opt/venv \
    PIP_CACHE_DIR=/root/.cache/pip
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --branch ${COMFY_VERSION} --depth 1 https://github.com/Comfy-Org/ComfyUI.git comfyui \
 && cd comfyui \
 && pip install -r requirements.txt -r manager_requirements.txt comfy-cli
COPY ./comfy-config.ini /.root/.config/comfy-cli/config.ini

WORKDIR /home/ubuntu/comfyui
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/comfyui-impact-pack \
 && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF custom-nodes/comfyui-gguf \
 && pip install -r custom_nodes/comfyui-impact-pack/requirements.txt gguf

WORKDIR /home/ubuntu/comfyui/
ARG USERID=1000
# default to the video group
ARG GROUPID=44

RUN chown --recursive ${USERID}:${GROUPID} /home/ubuntu
