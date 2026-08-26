FROM python:3.11-slim

WORKDIR /app

COPY model/ ./model/
COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN pip install --no-cache-dir fastapi uvicorn[standard] numpy
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8080

CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080"]
