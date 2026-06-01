FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRID_OUTPUT_DIR=/workspace/grid/docker_output

WORKDIR /workspace/grid

COPY requirements.txt requirements_optional.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "-m", "src.grid.docker_cli"]
CMD ["smoke"]
