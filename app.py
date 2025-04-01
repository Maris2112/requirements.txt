from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import fitz  # PyMuPDF for image extraction from PDF

app = Flask(__name__)  # <-- ЭТО ГЛАВНОЕ

# ⬇️ Вставь сюда функции preprocess_image, extract_tables, extract_images_from_pdf

@app.route("/ocr", methods=["POST"])
def ocr():
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

    import hashlib
    hash_id = hashlib.md5(all_text.encode('utf-8')).hexdigest()

    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Qdrant
    from langchain.schema import Document

    import os
    openai_key = os.environ.get("OPENAI_API_KEY")
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")

    embed_fn = OpenAIEmbeddings(openai_api_key=openai_key)
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

    client.add_documents(documents, collection_name="dm_docs")

    return jsonify({
        "status": "ok",
        "hash": hash_id,
        "pages": len(results),
        "images": extracted_images
    })

# ✅ Вне функции, с правильными отступами
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)




