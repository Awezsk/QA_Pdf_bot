FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -v --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]