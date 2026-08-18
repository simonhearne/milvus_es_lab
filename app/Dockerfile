# Slim runtime for the demo app + the loaders it runs.
# Built once by the `demo` service in docker-compose.yml; the workspace is
# bind-mounted at runtime, so only third-party deps are baked in here.
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "pymilvus>=3.0" \
    "elasticsearch>=9.4,<10" \
    pyarrow pandas numpy tqdm \
    "fastapi>=0.115" "uvicorn[standard]>=0.30"

WORKDIR /workspace
