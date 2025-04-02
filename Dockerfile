FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    tesseract-ocr-tur \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка казахских языковых данных для Tesseract
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata
WORKDIR /tmp
RUN apt-get update && apt-get install -y wget unzip
RUN wget https://github.com/tesseract-ocr/tessdata/raw/main/kaz.traineddata \
    -P /usr/share/tesseract-ocr/4.00/tessdata/

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Удаление hashlib из requirements (он встроен в Python)
RUN grep -v "hashlib" requirements.txt > requirements_fixed.txt && \
    mv requirements_fixed.txt requirements.txt

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY app.py .

# Настройка переменных окружения
ENV PYTHONUNBUFFERED=1

# Открытие порта (используем стандартное значение 5000)
EXPOSE 5000

# Запуск сервера с использованием порта из переменной окружения
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
