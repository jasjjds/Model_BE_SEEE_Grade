# Dockerfile
FROM python:3.10-slim

# 1. Cài đặt thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy code vào container
COPY . /app

# 3. Cài đặt thư viện Python
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# 4. Mở port 8000 (Port chuẩn của FastAPI)
EXPOSE 8000

# 5. Lệnh chạy server (Sử dụng uvicorn thay vì streamlit)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]