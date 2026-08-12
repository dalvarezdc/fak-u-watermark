FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY packages ./packages
COPY api ./api
COPY ui ./ui
COPY cli ./cli

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
