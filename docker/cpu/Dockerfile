# 使用轻量级 Python 镜像
FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1
ENV MODEL_ROOT=/app/models

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

# 2. 拷贝本地源码包
# 注意：路径是相对于构建上下文根目录的
COPY python-package /app/insightface-sdk

# 3. 拷贝依赖文件并安装
COPY docker/cpu/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 拷贝 FastAPI 代码 (假设在根目录或指定位置)
COPY docker/app.py .

EXPOSE 5004

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5004"]