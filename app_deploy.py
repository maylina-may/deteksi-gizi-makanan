import streamlit as st
import numpy as np
import cv2
import os
import io
import requests
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from typing import List, Dict, Optional, Tuple

# ============================================================
# Konfigurasi Halaman
# ============================================================
st.set_page_config(
    page_title="Deteksi Gizi Makanan",
    page_icon="🍽️",
    layout="wide",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "Aplikasi Deteksi Gizi Makanan (YOLO + AI)"
    }
)

# ============================================================
# Groq API Config
# ============================================================
GROQ_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ============================================================
# Load YOLO Model
# ============================================================
@st.cache_resource(show_spinner="Memuat model deteksi...")
def load_model():
    possible_paths = [
        os.path.join(".", "best.pt"),
        os.path.join(".", "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(".", "runs", "detect", "train2", "weights", "best.pt"),
        os.path.join(".", "runs", "detect", "train3", "weights", "best.pt"),
    ]

    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        st.error("Model YOLO (best.pt) tidak ditemukan! Pastikan file best.pt ada.")
        st.stop()

    return YOLO(model_path)

model = load_model()

CLASS_NAMES = [
    'ayam bakar', 'ayam goreng', 'bakso', 'bakwan', 'batagor', 'bihun', 'capcay', 'gado-gado',
    'ikan goreng', 'kerupuk', 'martabak telur', 'mie', 'nasi goreng', 'nasi putih', 'nugget',
    'opor ayam', 'pempek', 'rendang', 'roti', 'sate', 'sosis', 'soto', 'steak', 'tahu',
    'telur', 'tempe', 'terong balado', 'tumis kangkung', 'udang'
]

CONF_THRESHOLD = 0.15
CLASS_CONF_THRESHOLDS = {13: 0.35}
MIN_BBOX_AREA = 500
MAX_ASPECT_RATIO = 5.0

# ============================================================
# Helper Functions
# ============================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode byte gambar ke numpy array (BGR format untuk OpenCV)."""
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def detect_objects(image: np.ndarray) -> Tuple[List[Dict], List[str]]:
    """Deteksi objek menggunakan YOLO, return list objek + list nama makanan."""
    results = model(image, conf=CONF_THRESHOLD, iou=0.5, agnostic_nms=True)[0]
    detected_objects = []
    makanan_list = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = CLASS_NAMES[cls]
            conf = float(box.conf[0])

            box_width = x2 - x1
            box_height = y2 - y1
            bbox_area = box_width * box_height

            # Filter: area minimal
            if bbox_area < MIN_BBOX_AREA:
                continue

            # Filter: aspect ratio
            aspect_ratio = max(box_width, box_height) / max(box_height, box_width, 1)
            if aspect_ratio > MAX_ASPECT_RATIO:
                continue

            # Filter: confidence per kelas
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

    return detected_objects, makanan_list

def draw_boxes(image: np.ndarray, detected_objects: List[Dict]) -> bytes:
    """Gambar bounding box di image, return JPEG bytes."""
    img_copy = image.copy()
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["nama"]
        conf = obj["confidence"]

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(img_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    _, img_encoded = cv2.imencode('.jpg', img_copy)
    return img_encoded.tobytes()

def query_groq(prompt: str, context: str = "") -> str:
    """Query Groq LLM untuk info gizi."""
    system_message = (
        "Kamu adalah asisten gizi makanan Indonesia yang membantu pengguna "
        "memahami informasi gizi dari makanan serta memberikan saran pola makan yang sehat. "
        "Jika diminta informasi gizi, berikan dalam bentuk tabel yang rapi dan bahasa Indonesia."
    )

    full_prompt = f"{context}\n\nPengguna: {prompt}" if context else prompt

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": 0.7,
            },
            timeout=30,
        )

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

def inject_custom_css():
    """Inject CSS kustom."""
    st.markdown("""
    <style>
    .chip {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px 6px 0 0;
        border-radius: 16px;
        background: #EEF2FF;
        color: #3730A3;
        font-size: 12px;
        border: 1px solid #E0E7FF;
        white-space: nowrap;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================

def main():
    inject_custom_css()

    st.title("🍽️ Deteksi Gizi Makanan")
    st.write(
        "Upload foto makanan atau ambil foto langsung. Sistem akan mendeteksi makanan "
        "dengan YOLO dan menampilkan informasi kandungan gizi."
    )

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        conf_filter = st.slider(
            "Filter Confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Minimal confidence untuk ditampilkan di tabel"
        )
        render_markdown_table = st.toggle(
            "Render Tabel Gizi sebagai Markdown", value=True
        )

        # Info status
        st.divider()
        st.caption("**Status Model:** ✅ YOLO dimuat")
        st.caption(f"**Kelas makanan:** {len(CLASS_NAMES)} jenis")

    # ---- INPUT METHOD ----
    input_method = st.radio(
        "Pilih metode input",
        options=["📁 Upload Gambar", "📸 Ambil Foto"],
        horizontal=True,
        index=0,
        label_visibility="collapsed"
    )

    image_bytes = None
    image_source_name = None
    preview_image = None

    if input_method == "📁 Upload Gambar":
        uploaded = st.file_uploader(
            "Upload Gambar (JPG/JPEG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )
        if uploaded:
            image_bytes = uploaded.read()
            image_source_name = getattr(uploaded, "name", "uploaded.jpg")
            preview_image = Image.open(io.BytesIO(image_bytes))
    else:
        st.markdown("""
        <div style="margin-bottom:8px;font-size:14px;color:#666;">
            💡 Gunakan tombol di bawah untuk mengakses kamera.
        </div>
        """, unsafe_allow_html=True)

        cam = st.camera_input(
            "Ambil Foto",
            label_visibility="collapsed",
            key="camera_main"
        )

        if cam:
            image_bytes = cam.read()
            image_source_name = "camera_capture.jpg"
            preview_image = Image.open(io.BytesIO(image_bytes))

    # ---- TOMBOL DETEKSI ----
    col_preview, col_action = st.columns([3, 2], vertical_alignment="bottom")

    with col_preview:
        if preview_image:
            st.image(preview_image, caption="Pratinjau Gambar", use_container_width=True)
        detect_btn = st.button(
            "🔎 Deteksi Gizi",
            type="primary",
            use_container_width=True,
            disabled=(image_bytes is None)
        )

    # ---- PROSES DETEKSI ----
    if detect_btn and image_bytes is not None:
        with st.spinner("🔍 Mendeteksi makanan..."):
            try:
                # Preprocess
                image = preprocess_image(image_bytes)

                # Deteksi
                detected_objects, makanan_list = detect_objects(image)

                # Generate gambar berannotasi
                annotated_bytes = draw_boxes(image, detected_objects)
                annotated_image = Image.open(io.BytesIO(annotated_bytes))

                # Query Groq untuk info gizi
                if len(makanan_list) > 0:
                    makanan_str = ', '.join(list(set(makanan_list)))
                    gizi_prompt = (
                        f"Dari gambar yang diunggah, saya mendeteksi makanan berikut: {makanan_str}. "
                        "Bisakah kamu memberikan informasi tentang kandungan gizi dari makanan tersebut? "
                        "Berikan dalam bentuk tabel yang rapi dan bahasa Indonesia."
                    )
                else:
                    gizi_prompt = (
                        "Saya tidak bisa mendeteksi makanan dalam gambar ini. "
                        "Mohon unggah gambar yang berisi makanan dengan jelas. "
                        "Berikan saran tentang cara mengambil foto makanan yang baik untuk analisis."
                    )

                with st.spinner("🧠 Meminta info gizi dari AI..."):
                    gizi_text = query_groq(gizi_prompt)

                # ---- TAMPILKAN HASIL ----
                # Gambar berannotasi
                col_result, _ = st.columns([3, 2], gap="large")
                with col_result:
                    st.subheader("📷 Hasil Deteksi")
                    st.image(annotated_image, caption="Gambar dengan Bounding Box", use_container_width=True)
                    st.download_button(
                        label="💾 Unduh Gambar Hasil",
                        data=annotated_bytes,
                        file_name="hasil_deteksi.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )

                # Detail objek
                st.subheader("📦 Detail Objek Terdeteksi")
                filtered = [o for o in detected_objects if float(o.get("confidence", 0.0)) >= conf_filter]

                if filtered:
                    def bbox_area(b):
                        try:
                            x1, y1, x2, y2 = b
                            return max(0, x2 - x1) * max(0, y2 - y1)
                        except Exception:
                            return None

                    rows = []
                    for o in filtered:
                        rows.append({
                            "Nama": o.get("nama", "-"),
                            "Confidence": round(float(o.get("confidence", 0.0)), 4),
                            "BBox": o.get("bbox", []),
                            "Luas (px²)": bbox_area(o.get("bbox", []))
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Tidak ada objek yang memenuhi filter confidence saat ini.")

                # Tabel gizi
                st.subheader("🥗 Kandungan Gizi")
                if gizi_text:
                    if render_markdown_table:
                        st.markdown(gizi_text)
                    else:
                        st.text(gizi_text)
                else:
                    st.info("Tidak ada info gizi yang didapatkan.")

                st.toast("✅ Selesai memproses!")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()

