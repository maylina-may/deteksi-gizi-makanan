from ultralytics import YOLO
from flask import Flask, request, jsonify
import numpy as np
import os
import cv2
from flask_cors import CORS
import requests
import io
import base64

app = Flask(__name__)
CORS(app)

# ========== Konfigurasi Groq ==========
KEY = os.environ.get("GROQ_API_KEY")
if not KEY:
    raise ValueError("GROQ_API_KEY environment variable tidak ditemukan! Set dengan: set GROQ_API_KEY=your_key")
MODEL = "llama-3.3-70b-versatile"  # Model yang tersedia di Groq

# ========== Load YOLO Model ==========
# Coba model baru dulu (train3), fallback ke train2
model_path_new = os.path.join(".", "runs", "detect", "train3", "weights", "best.pt")
model_path_old = os.path.join(".", "runs", "detect", "train2", "weights", "best.pt")

if os.path.exists(model_path_new):
    model_path = model_path_new
    print(f"Loading model from: {model_path_new}")
else:
    model_path = model_path_old
    print(f"Loading model from: {model_path_old}")

modelyolo = YOLO(model_path)

class_names = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

# ========== Konfigurasi Deteksi ==========
# Confidence threshold umum
CONF_THRESHOLD = 0.15

# Confidence threshold spesifik per class (index) untuk mengurangi false positive
# Khusus nasi putih (index 13) butuh confidence lebih tinggi karena sering jadi false positive
CLASS_CONF_THRESHOLDS = {
    13: 0.35,   # nasi putih - butuh confidence lebih tinggi
}

# Minimum area bounding box (dalam piksel) untuk menyaring deteksi noise
MIN_BBOX_AREA = 500

# Maksimum rasio aspek bounding box (mencegah box yang terlalu gepeng/memanjang)
MAX_ASPECT_RATIO = 5.0
MIN_ASPECT_RATIO = 0.2

# ========== Helper Functions ==========
def preprocess_image(image_data):
    image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def draw_boxes(image, results):
    if results.boxes is None:
        return image
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = class_names[cls]
        conf = float(box.conf[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image

def query_llm(prompt, context=""):
    """
    Query ke Groq API dengan error handling.
    Mengembalikan teks response atau pesan error.
    """
    system_message = (
        "Kamu adalah asisten gizi makanan Indonesia yang membantu pengguna "
        "memahami informasi gizi dari makanan serta memberikan saran pola makan yang sehat. "
        "Jika diminta informasi gizi, berikan dalam bentuk tabel yang rapi."
    )

    if context:
        full_prompt = f"{context}\n\nPengguna: {prompt}"
    else:
        full_prompt = prompt

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.7,
            },
            timeout=30,
        )

        # Cek apakah response dari Groq sukses
        if response.status_code != 200:
            error_detail = response.json().get("error", {}).get("message", "Unknown error")
            return f"Maaf, terjadi error saat menghubungi AI: {error_detail}"

        completion = response.json()
        return completion["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        return "Maaf, permintaan ke AI timeout. Silakan coba lagi."
    except requests.exceptions.ConnectionError:
        return "Maaf, gagal terhubung ke layanan AI. Periksa koneksi internet."
    except (KeyError, IndexError, ValueError) as e:
        return f"Maaf, terjadi kesalahan saat memproses response AI: {str(e)}"
    except Exception as e:
        return f"Maaf, terjadi kesalahan: {str(e)}"

# ========== Endpoint: Deteksi Gizi dari Gambar ==========
@app.route("/detect-gizi", methods=["POST"])
def detect_gizi():
    try:
        image_file = request.files.get("image")
        chat_history = request.form.get("chat_history", "")

        if not image_file:
            return jsonify({"error": "Tidak ada file gambar yang diberikan"}), 400

        image_data = image_file.read()
        image = preprocess_image(image_data)

        # Gunakan confidence threshold yang lebih tinggi untuk inference
        results = modelyolo(image, conf=CONF_THRESHOLD, iou=0.5, agnostic_nms=True)[0]

        detected_objects = []
        makanan_list = []

        # Handle jika tidak ada deteksi
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = class_names[cls]
                conf = float(box.conf[0])

                # Filter bounding box terlalu kecil (noise)
                box_width = x2 - x1
                box_height = y2 - y1
                bbox_area = box_width * box_height
                if bbox_area < MIN_BBOX_AREA:
                    continue

                # Filter rasio aspek bounding box yang tidak wajar
                aspect_ratio = max(box_width, box_height) / max(box_height, box_width, 1)
                if aspect_ratio > MAX_ASPECT_RATIO:
                    continue

                # Filter confidence spesifik per class
                min_conf = CLASS_CONF_THRESHOLDS.get(cls, CONF_THRESHOLD)
                if conf < min_conf:
                    continue

                makanan_list.append(label)
                detected_objects.append({
                    "nama": label,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "box_area": bbox_area
                })

        # Prompt ke LLM berdasarkan deteksi
        if len(makanan_list) == 0:
            prompt = (
                "Saya tidak bisa mendeteksi makanan dalam gambar ini. "
                "Mohon unggah gambar yang berisi makanan dengan jelas. "
                "Berikan saran tentang cara mengambil foto makanan yang baik untuk analisis."
            )
        else:
            makanan_str = ', '.join(list(set(makanan_list)))  # Unique
            prompt = (
                f"Dari gambar yang diunggah, saya mendeteksi makanan berikut: {makanan_str}. "
                "Bisakah kamu memberikan informasi tentang kandungan gizi dari makanan tersebut? "
                "Berikan dalam bentuk tabel yang rapi dan bahasa Indonesia."
            )

        response_text = query_llm(prompt, chat_history)

        boxed_image = draw_boxes(image.copy(), results)
        _, img_encoded = cv2.imencode('.jpg', boxed_image)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        return jsonify({
            "objects": detected_objects,
            "image": "data:image/jpeg;base64," + img_base64,
            "gizi": response_text,
            "response": response_text,
            "detected_foods": makanan_list
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ========== Endpoint: Chat Text Saja ==========
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "")
        chat_history = data.get("chat_history", "")

        if not user_message:
            return jsonify({"error": "Tidak ada pesan yang diberikan"}), 400

        response = query_llm(user_message, chat_history)

        return jsonify({
            "response": response
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ========== Main ==========
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode, use_reloader=False)

