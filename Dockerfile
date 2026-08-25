FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY api_prediction_rx.py .
COPY modele_rx_pneumonie_finetuned.keras .

EXPOSE 8000

CMD ["uvicorn", "api_prediction_rx:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
