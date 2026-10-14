FROM python:3.12-slim
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd -m -u 10001 app
COPY src/ ./src/
RUN mkdir -p /app/work && chown -R app:app /app/work
USER app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "serve:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
