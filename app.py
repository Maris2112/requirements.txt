from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes

app = Flask(__name__)

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
text = pytesseract.image_to_string(image, lang='rus+kaz+eng+tur')
            results.append({"page": page_num + 1, "text": text})
    else:
        image = Image.open(file.stream)
        text = pytesseract.image_to_string(image, lang='eng')
        results.append({"page": 1, "text": text})

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
