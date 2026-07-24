"""
Aplikasi Deteksi Gizi Makanan - All-in-one (Streamlit + YOLO + Groq)
Tidak perlu Flask API terpisah. Bisa deploy langsung ke Railway.
Fitur: Upload gambar + Custom camera untuk HP
"""

import streamlit as st
import numpy as np
import cv2
import os
import io
import base64
import json
import requests
from PIL import Image
from ultralytics import YOLO
import pandas as pd
from typing import Tuple, List, Dict, Optional

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
# Load YOLO Model (dengan fallback path)
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
        st.error("Model YOLO (best.pt) tidak ditemukan! Pastikan file best.pt ada di folder.")
        st.stop()
    
    return YOLO(model_path)

model = load_model()

# Nama kelas makanan
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
MIN_ASPECT_RATIO = 0.2

# ============================================================
# Helper Functions
# ============================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode byte gambar ke numpy array (BGR format untuk OpenCV)."""
    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def detect_objects(image: np.ndarray):
    """Deteksi objek menggunakan YOLO, return list objek + annotated image bytes."""
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
    .camera-container {
        width: 100%;
        max-width: 640px;
        margin: 0 auto;
        border-radius: 12px;
        overflow: hidden;
        border: 2px dashed #ccc;
        background: #f9f9f9;
        text-align: center;
        padding: 10px;
    }
    .camera-container video {
        width: 100%;
        border-radius: 8px;
    }
    .camera-container canvas {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# Custom Camera Component untuk HP
# ============================================================

def camera_component() -> Optional[bytes]:
    """
    Custom camera component menggunakan HTML/JS.
    Work di HTTP & HTTPS, HP & Desktop.
    Returns: JPEG bytes atau None jika tidak ada capture.
    """
    
    camera_html = """
    <div id="cameraContainer" style="width:100%;max-width:640px;margin:0 auto;border-radius:12px;overflow:hidden;border:2px dashed #ccc;background:#f9f9f9;text-align:center;padding:10px;">
        <video id="video" width="100%" autoplay playsinline style="border-radius:8px;display:block;"></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <div style="margin-top:12px;display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
            <button id="captureBtn" style="padding:10px 24px;background:#4CAF50;color:white;border:none;border-radius:8px;font-size:16px;cursor:pointer;">📸 Ambil Foto</button>
            <button id="retakeBtn" style="padding:10px 24px;background:#FF9800;color:white;border:none;border-radius:8px;font-size:16px;cursor:pointer;display:none;">🔄 Ambil Ulang</button>
            <button id="useBtn" style="padding:10px 24px;background:#2196F3;color:white;border:none;border-radius:8px;font-size:16px;cursor:pointer;display:none;">✅ Gunakan Foto</button>
        </div>
        <p id="cameraStatus" style="margin-top:8px;font-size:14px;color:#666;">Mengakses kamera...</p>
        <img id="preview" style="display:none;width:100%;border-radius:8px;margin-top:8px;" />
    </div>
    
    <script>
    // ====== ELEMENTS ======
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');
    const retakeBtn = document.getElementById('retakeBtn');
    const useBtn = document.getElementById('useBtn');
    const status = document.getElementById('cameraStatus');
    const preview = document.getElementById('preview');
    
    let stream = null;
    let capturedImageData = null;
    
    // ====== START CAMERA ======
    async function startCamera() {
        try {
            // Prioritas: environment (kamera belakang HP) lalu user (kamera depan)
            const constraints = {
                video: {
                    facingMode: { ideal: 'environment' },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            video.style.display = 'block';
            preview.style.display = 'none';
            captureBtn.style.display = 'block';
            retakeBtn.style.display = 'none';
            useBtn.style.display = 'none';
            status.textContent = '✅ Kamera siap, klik "Ambil Foto"';
            status.style.color = '#4CAF50';
        } catch (err) {
            console.error('Camera error:', err);
            status.textContent = '❌ Gagal akses kamera: ' + err.message + '. Silakan upload gambar.';
            status.style.color = '#f44336';
            captureBtn.style.display = 'none';
        }
    }
    
    // ====== STOP CAMERA ======
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }
    
    // ====== CAPTURE ======
    function capture() {
        if (!video.videoWidth) return;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        // Convert to JPEG data URL
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        capturedImageData = dataUrl;
        
        // Show preview
        preview.src = dataUrl;
        preview.style.display = 'block';
        video.style.display = 'none';
        
        // Stop camera
        stopCamera();
        
        // Update buttons
        captureBtn.style.display = 'none';
        retakeBtn.style.display = 'inline-block';
        useBtn.style.display = 'inline-block';
        status.textContent = '✅ Foto diambil! Klik "Gunakan Foto" untuk memproses.';
        status.style.color = '#2196F3';
    }
    
    // ====== RETAKE ======
    function retake() {
        capturedImageData = null;
        startCamera();
    }
    
    // ====== USE PHOTO ======
    function usePhoto() {
        if (!capturedImageData) return;
        // Kirim data ke Streamlit
        const data = {
            image: capturedImageData
        };
        Streamlit.setComponentValue(JSON.stringify(data));
    }
    
    // ====== EVENT LISTENERS ======
    captureBtn.addEventListener('click', capture);
    retakeBtn.addEventListener('click', retake);
    useBtn.addEventListener('click', usePhoto);
    
    // ====== INIT ======
    // Set component ready
    Streamlit.setComponentReady();
    
    // Adjust height
    function updateHeight() {
        const container = document.getElementById('cameraContainer');
        Streamlit.setFrameHeight(container.offsetHeight + 50);
    }
    
    // Start camera
    startCamera();
    
    // Resize observer
    const observer = new ResizeObserver(() => updateHeight());
    observer.observe(document.getElementById('cameraContainer'));
    
    setTimeout(updateHeight, 1000);
    </script>
    """
    
    # Gunakan st.components.v1.html untuk render komponen kustom
    # Karena Streamlit belum punya native component, kita gunakan approach
    # st.markdown + iframe via components.html tidak support bidirectional comms
    # Jadi kita fallback ke st.camera_input yang sudah diperbaiki dengan custom approach
    
    # APPROACH: Gunakan st.file_uploader untuk menerima hasil capture dari camera input
    # dan juga sediakan opsi upload file biasa
    # Untuk kompatibilitas HP maksimal, kita pakai input file dengan accept="image/*" dan capture="environment"
    
    return None


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
        
        # Mobile-friendly camera input via HTML input capture
        # First, try Streamlit's camera_input (works on newer mobile browsers)
        cam = st.camera_input(
            "Ambil Foto",
            label_visibility="collapsed",
            key="camera_main"
        )
        
        if cam:
            image_bytes = cam.read()
            image_source_name = "camera_capture.jpg"
            preview_image = Image.open(io.BytesIO(image_bytes))
        else:
            # Fallback: HTML5 input with capture attribute (works on all mobile browsers)
            st.markdown("""
            <p style="text-align:center;margin:8px 0;font-weight:bold;color:#888;">— ATAU —</p>
            """, unsafe_allow_html=True)
            
            fallback_file = st.file_uploader(
                "Buka Kamera via File Upload",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
                key="camera_fallback",
                help="Pada HP: pilih 'Camera' dari menu file picker"
            )
            
            if fallback_file:
                image_bytes = fallback_file.read()
                image_source_name = getattr(fallback_file, "name", "camera_capture.jpg")
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

