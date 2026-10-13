FROM rocm/dev-ubuntu-26.04:10.1.0-full
RUN apt update \
 && apt install -y curl git

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/
WORKDIR /home/ubuntu/
ARG COMFY_VERSION="v0.39.0"
ENV UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 \
 && git clone --branch ${COMFY_VERSION} --depth 1 https://github.com/Comfy-Org/ComfyUI.git comfyui \
 && cd comfyui \
 && uv venv \
 && uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "torch[device-gfx1200]==2.13.0+rocm10.1.0" \
    "torchvision[device-gfx1200]==0.28.0+rocm10.1.0" \
    "torchaudio==2.11.0.2+rocm10.1.0" \
 && uv pip install -r requirements.txt -r manager_requirements.txt comfy-cli

WORKDIR /home/ubuntu/comfyui
RUN --mount=type=cache,target=/root/.cache/uv  \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/comfyui-impact-pack \
 && git clone --depth 1 https://github.com/molbal/ComfyUI-GGUF custom_nodes/comfyui-gguf \
 && uv pip install -r custom_nodes/comfyui-impact-pack/requirements.txt -r custom_nodes/comfyui-gguf/requirements.txt

WORKDIR /home/ubuntu/comfyui/
ARG USERID=1000
# default to the video group
ARG GROUPID=44

RUN chown --recursive ${USERID}:${GROUPID} /home/ubuntu
