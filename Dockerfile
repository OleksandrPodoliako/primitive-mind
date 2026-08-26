FROM python:3.11-slim

WORKDIR /app

COPY model/ ./model/
COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860"]
