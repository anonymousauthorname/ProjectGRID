ARG GRID_VERL_BASE_IMAGE=verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
FROM ${GRID_VERL_BASE_IMAGE}

ARG GRID_VERL_REPO=https://github.com/volcengine/verl.git
ARG GRID_VERL_REF=v0.7.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRID_OUTPUT_DIR=/workspace/grid/docker_output \
    PYTHONPATH=/workspace/grid:/workspace/verl

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --branch "${GRID_VERL_REF}" "${GRID_VERL_REPO}" /workspace/verl

WORKDIR /workspace/grid

COPY requirements.txt requirements_optional.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "-m", "src.grid.docker_cli"]
CMD ["verl-sft-rl-export"]
