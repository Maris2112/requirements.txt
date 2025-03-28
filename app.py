from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file.stream)
    text = pytesseract.image_to_string(image, lang='eng')

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
