from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes
import cv2
import numpy as np
import fitz  # PyMuPDF for image extraction from PDF

app = Flask(__name__)

# 🔧 Функция препроцессинга изображений
def preprocess_image(pil_image):
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    denoised = cv2.medianBlur(thresh, 3)
    return Image.fromarray(denoised)

# 🧾 Функция извлечения таблиц (пока простая)
def extract_tables(image):
    data = pytesseract.image_to_data(image, lang='rus+kaz+eng+tur', output_type=pytesseract.Output.DICT)
    lines = []
    last_line = -1
    line = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:
            current_line = data['line_num'][i]
            if current_line != last_line:
                if line:
                    lines.append(" ".join(line))
                line = [data['text'][i]]
                last_line = current_line
            else:
                line.append(data['text'][i])
    if line:
        lines.append(" ".join(line))
    return lines

# 🖼️ Извлечение изображений из PDF с помощью PyMuPDF
def extract_images_from_pdf(file_bytes):
    images = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page_index in range(len(doc)):
        for img_index, img in enumerate(doc.get_page_images(page_index)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes))
            images.append({"page": page_index + 1, "image": base64.b64encode(image_bytes).decode('utf-8')})
    return images

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename.lower()
    file_bytes = file.read()

    results = []
    
    if filename.endswith('.pdf'):
        images = convert_from_bytes(file_bytes)
        for page_num, img in enumerate(images):
            pre_img = preprocess_image(img)
            text = pytesseract.image_to_string(pre_img, lang='rus+kaz+eng+tur')
            tables = extract_tables(pre_img)
            results.append({"page": page_num + 1, "text": text, "tables": tables})

        extracted_images = extract_images_from_pdf(file_bytes)
    else:
        image = Image.open(io.BytesIO(file_bytes))
        pre_img = preprocess_image(image)
        text = pytesseract.image_to_string(pre_img, lang='rus+kaz+eng+tur')
        tables = extract_tables(pre_img)
        results.append({"page": 1, "text": text, "tables": tables})
        extracted_images = []

    return jsonify({"results": results, "images": extracted_images})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
