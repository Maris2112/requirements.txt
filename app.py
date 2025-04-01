from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import fitz  # PyMuPDF
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

app = Flask(__name__)


def preprocess_image(image):
    # Преобразование PIL в OpenCV
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    return thresh


def extract_tables(image):
    # Простая заглушка для таблиц
    return []


def extract_images_from_pdf(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            encoded = base64.b64encode(img_bytes).decode('utf-8')
            images.append(encoded)
    return images


@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename.lower()
        file_bytes = file.read()

        results = []
        all_text = ""

        if filename.endswith('.pdf'):
            images = convert_from_bytes(file_bytes)
            for page_num, img in enumerate(images):
                pre_img = preprocess_image(img)
                text = pytesseract.image_to_string(pre_img, lang='rus+kaz+eng+tur')
                tables = extract_tables(pre_img)
                results.append({"page": page_num + 1, "text": text, "tables": tables})
                all_text += f"\n{text}"
            extracted_images = extract_images_from_pdf(file_bytes)
        else:
            image = Image.open(io.BytesIO(file_bytes))
            pre_img = preprocess_image(image)
            text = pytesseract.image_to_string(pre_img, lang='rus+kaz+eng+tur')
            tables = extract_tables(pre_img)
            results.append({"page": 1, "text": text, "tables": tables})
            all_text = text
            extracted_images = []

        # 🧠 HASH
        import hashlib
        hash_id = hashlib.md5(all_text.encode('utf-8')).hexdigest()

        # 🧠 QDRANT + EMBEDDINGS
        openai_key = os.environ.get("OPENAI_API_KEY")
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_key = os.environ.get("QDRANT_API_KEY")

        embed_fn = OpenAIEmbeddings(openai_api_key=openai_key)

        # Создание коллекции, если её нет
        raw_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_key,
        )

        collection_name = "dm_docs"
        if not raw_client.collection_exists(collection_name=collection_name):
            raw_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

        client = Qdrant(
            url=qdrant_url,
            prefer_grpc=False,
            api_key=qdrant_key,
            embedding_function=embed_fn,
        )

        documents = [
            Document(
                page_content=page["text"],
                metadata={
                    "page": page["page"],
                    "source": filename,
                    "hash": hash_id,
                },
            )
            for page in results
        ]

        client.add_documents(documents, collection_name=collection_name)

        return jsonify({
            "status": "ok",
            "hash




