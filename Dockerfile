FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Экспонирование порта
EXPOSE 5000

# Команда запуска
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}"]
