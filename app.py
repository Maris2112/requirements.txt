from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes
import cv2
import numpy as np

app = Flask(__name__)

# 🔧 Функция препроцессинга изображений
def preprocess_image(pil_image):
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    denoised = cv2.medianBlur(thresh, 3)
    return Image.fromarray(denoised)

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = file.filename.lower()

    results = []

    if filename.endswith('.pdf'):
        images = convert_from_bytes(file.read())
        for page_num, img in enumerate(images):
            preprocessed_img = preprocess_image(img)
            text = pytesseract.image_to_string(preprocessed_img, lang='rus+kaz+eng+tur')
            results.append({"page": page_num + 1, "text": text})
    else:
        image = Image.open(file.stream)
        preprocessed_img = preprocess_image(image)
        text = pytesseract.image_to_string(preprocessed_img, lang='rus+kaz+eng+tur')
        results.append({"page": 1, "text": text})

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
