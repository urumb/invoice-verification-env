FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=""
ENV API_BASE_URL=""
ENV MODEL_NAME=""

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
