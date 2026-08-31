FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python src/ml/train.py

EXPOSE 8000

ENV PORT=8000
ENV FLASK_ENV=production

CMD ["gunicorn", "app:create_app()", "--bind", "0.0.0.0:8000"]
