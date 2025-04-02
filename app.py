from flask import Flask, request, jsonify
import pytesseract
from PIL import Image, UnidentifiedImageError
import io
import base64
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import fitz  # PyMuPDF
import os
import hashlib
import logging
import datetime
from werkzeug.utils import secure_filename

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Константы и конфигурация
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "rus+kaz+eng+tur")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "dm_docs")
VECTOR_SIZE = 1536

# Максимальный размер загружаемого файла (10 МБ)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def ping():
    logger.info("Пинг-сервер запущен")
    return "👋 GPT-инженер в деле! Сервер жив!"


def preprocess_image(image, preprocessing_level="default"):
    try:
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if preprocessing_level == "light":
            return cv2.GaussianBlur(gray, (3, 3), 0)
        elif preprocessing_level == "aggressive":
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            kernel = np.ones((1, 1), np.uint8)
            return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        else:
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
            return thresh
    except Exception as e:
        logger.error(f"Ошибка в preprocess_image: {e}")
        raise


def extract_images_from_pdf(pdf_bytes):
    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    encoded = base64.b64encode(img_bytes).decode('utf-8')
                    images.append({"image": encoded, "page": page_num + 1, "index": img_index})
                except Exception as e:
                    logger.error(f"PDF Image Extraction Error: {e}")
    except Exception as e:
        logger.error(f"PDF Read Error: {e}")
    return images


def init_qdrant():
    openai_key = os.environ.get("OPENAI_API_KEY")
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")

    if not all([openai_key, qdrant_url, qdrant_key]):
        raise ValueError("Одна или несколько переменных окружения отсутствуют")

    embed_fn = OpenAIEmbeddings(openai_api_key=openai_key)
    raw_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    if not raw_client.collection_exists(collection_name=COLLECTION_NAME):
        raw_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    return Qdrant(url=qdrant_url, api_key=qdrant_key, prefer_grpc=False, embedding_function=embed_fn)


@app.route("/ocr", methods=["POST"])
def ocr():
    if 'file' not in request.files:
        return jsonify({"error": "Файл не был загружен"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Недопустимый тип файла"}), 400

    preprocessing_level = request.form.get('preprocessing', 'default')
    ocr_options = request.form.get('ocr_options', 'normal')
    store_in_qdrant = request.form.get('store_in_qdrant', 'true').lower() == 'true'

    try:
        filename = secure_filename(file.filename.lower())
        file_bytes = file.read()

        results = []
        all_text = ""

        if filename.endswith('.pdf'):
            images = convert_from_bytes(file_bytes)
            for page_num, img in enumerate(images):
                pre_img = preprocess_image(img, preprocessing_level)
                config = "--oem 1 --psm 6" if ocr_options == "accurate" else ""
                text = pytesseract.image_to_string(pre_img, lang=OCR_LANGUAGES, config=config)
                results.append({"page": page_num + 1, "text": text})
                all_text += f"\n{text}"
            extracted_images = extract_images_from_pdf(file_bytes)
        else:
            image = Image.open(io.BytesIO(file_bytes))
            pre_img = preprocess_image(image, preprocessing_level)
            config = "--oem 1 --psm 6" if ocr_options == "accurate" else ""
            text = pytesseract.image_to_string(pre_img, lang=OCR_LANGUAGES, config=config)
            results.append({"page": 1, "text": text})
            all_text = text
            buffered = io.BytesIO()
            image.save(buffered, format=image.format or "JPEG")
            extracted_images = [{"image": base64.b64encode(buffered.getvalue()).decode('utf-8'), "page": 1, "index": 0}]

        hash_id = hashlib.md5(all_text.encode('utf-8')).hexdigest()

        if store_in_qdrant:
            client = init_qdrant()
            documents = [Document(
                page_content=page["text"],
                metadata={
                    "page": page["page"],
                    "source": filename,
                    "hash": hash_id,
                    "file_type": os.path.splitext(filename)[1][1:],
                    "timestamp": datetime.datetime.now().isoformat()
                }) for page in results]
            client.add_documents(documents, collection_name=COLLECTION_NAME)

        return jsonify({
            "status": "ok",
            "hash": hash_id,
            "pages": len(results),
            "text": results,
            "images": extracted_images
        })

    except Exception as e:
        logger.error(f"Ошибка OCR: {e}")
        return jsonify({"error": f"Ошибка обработки: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    logger.info(f"Старт сервера на порту {port}, debug={debug_mode}")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)






