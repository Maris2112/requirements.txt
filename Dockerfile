FROM python:3.10-slim

# Установка Tesseract OCR и зависимостей
RUN apt-get update && \
   apt-get install -y tesseract-ocr \
  tesseract-ocr-eng \
  tesseract-ocr-rus \
  tesseract-ocr-kaz \
  tesseract-ocr-tur \
  libglib2.0-0 libsm6 libxext6 libxrender-dev \

    poppler-utils gcc && \
    apt-get clean

# Копируем файлы
WORKDIR /app
COPY . .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем приложение
CMD ["python", "app.py"]
