FROM python:3.11-slim

WORKDIR /app

ENV PORT=7860
ENV LOCAFIND_EMBEDDING_BACKEND=tfidf
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
