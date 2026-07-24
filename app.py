import streamlit as st
import requests
import base64
import io
from PIL import Image
import pandas as pd
from typing import Tuple, List, Dict

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(
    page_title="Deteksi Gizi Makanan",
    page_icon="🍽️",
    layout="wide",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "UI Streamlit untuk Deteksi Gizi Makanan (YOLO + LLM)."
    }
)

# -----------------------------
# Helper Functions
# -----------------------------
def parse_data_url(data_url: str) -> Tuple[str, bytes]:
    """
    Mem-parse data URL "data:<mime>;base64,<payload>" menjadi (mime, bytes)
    """
    if not data_url.startswith("data:"):
        raise ValueError("Bukan data URL yang valid")
    header, b64data = data_url.split(",", 1)
    mime = header.split(";")[0].split(":", 1)[1]
    raw = base64.b64decode(b64data)
    return mime, raw

def post_image(api_url: str, file_name: str, file_bytes: bytes, mime: str, timeout: int = 60) -> Dict:
    files = {
        "image": (file_name, file_bytes, mime or "application/octet-stream")
    }
    resp = requests.post(api_url, files=files, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def chip(text: str) -> str:
    """
    Menghasilkan HTML sederhana untuk chip/tag.
    """
    return f"""
    <span class="chip">{text}</span>
    """

def inject_style():
    st.markdown(
        """
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
        .metric-card {
            padding: 14px 16px;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            background: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_base = st.text_input(
        "API Base URL",
        value="http://localhost:5000",
        help="Isi dengan alamat API Flask kamu."
    )
    endpoint_path = st.text_input(
        "Endpoint Deteksi",
        value="/detect-gizi",
        help="Path endpoint pada API Flask."
    )
    conf_filter = st.slider(
        "Filter Confidence (untuk tampilan list, bukan mempengaruhi bounding box dari server)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05
    )
    render_markdown_table = st.toggle(
        "Render Tabel Gizi sebagai Markdown",
        value=True,
        help="Jika dimatikan, akan ditampilkan sebagai teks apa adanya."
    )
    st.caption("Pastikan API sudah berjalan. Contoh: app Flask berjalan di http://localhost:5000.")

inject_style()

# -----------------------------
# Header
# -----------------------------
st.title("🍽️ Deteksi Gizi Makanan")
st.write(
    "Unggah foto makanan, sistem akan mendeteksi objek (makanan) dengan YOLO dan meminta LLM untuk "
    "menyusun tabel kandungan gizi. Hasil deteksi divisualisasikan dengan bounding box."
)

# -----------------------------
# Upload & Action
# -----------------------------
input_method = st.radio(
    "Pilih metode input:",
    options=["📁 Upload Gambar", "📸 Ambil Foto"],
    horizontal=True,
    index=0,
    label_visibility="collapsed"
)

uploaded = None
cam_image = None

if input_method == "📁 Upload Gambar":
    uploaded = st.file_uploader(
        "Unggah Gambar (JPG/JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
else:
    cam_image = st.camera_input(
        "Ambil Foto",
        label_visibility="collapsed"
    )

col_preview, col_action = st.columns([3, 2], vertical_alignment="bottom")

with col_preview:
    preview_img = uploaded or cam_image
    if preview_img:
        st.image(preview_img, caption="Pratinjau Gambar", use_container_width=True)
    st.write("")
    st.write("")
    detect_btn = st.button("🔎 Deteksi Gizi", type="primary", use_container_width=True, disabled=not preview_img)

# -----------------------------
# Hasil
# -----------------------------
if detect_btn and (uploaded or cam_image):
    api_url = api_base.rstrip("/") + "/" + endpoint_path.lstrip("/")
    try:
        with st.spinner("Memproses..."):
            source = uploaded or cam_image
            file_bytes = source.read()
            mime = source.type or "image/jpeg"
            file_name = getattr(source, "name", "camera_capture.jpg")
            result = post_image(api_url, file_name, file_bytes, mime, timeout=90)

        # Validasi respon
        if not isinstance(result, dict):
            st.error("Format respon API tidak sesuai.")
        else:
            # Deteksi error dari API
            if "error" in result:
                st.error(f"API Error: {result['error']}")
            else:
                # Gambar beranotasi
                img_data_url = result.get("image")
                objects = result.get("objects", [])
                # Coba baca dari key 'gizi' dulu, fallback ke 'response'
                gizi_text = result.get("gizi") or result.get("response", "")

                # Gambar beranotasi (pakai lebar kolom yang sama dengan pratinjau)
                col_result, _ = st.columns([3, 2], gap="large")
                with col_result:
                    st.subheader("📷 Hasil Deteksi")
                    if img_data_url:
                        try:
                            mime, img_bytes = parse_data_url(img_data_url)
                            image = Image.open(io.BytesIO(img_bytes))
                            st.image(image, caption="Gambar dengan Bounding Box", use_container_width=True)
                            st.download_button(
                                label="Unduh Gambar Hasil",
                                data=img_bytes,
                                file_name="hasil_deteksi.jpg",
                                mime=mime,
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"Gagal menampilkan gambar beranotasi: {e}")
                    else:
                        st.info("API tidak mengembalikan gambar beranotasi.")

                # Detail objek
                st.subheader("📦 Detail Objek Terdeteksi")
                # Filter berdasarkan threshold untuk tampilan
                filtered = [o for o in objects if float(o.get("confidence", 0.0)) >= conf_filter]
                if filtered:
                    # Bentuk tabel data
                    def bbox_area(b):
                        try:
                            x1, y1, x2, y2 = b
                            return max(0, x2 - x1) * max(0, y2 - y1)
                        except Exception:
                            return None

                    rows = []
                    for o in filtered:
                        nama = o.get("nama", "-")
                        conf = float(o.get("confidence", 0.0))
                        bbox = o.get("bbox", [None, None, None, None])
                        rows.append({
                            "Nama": nama,
                            "Confidence": round(conf, 4),
                            "BBox": bbox,
                            "Luas (px^2)": bbox_area(bbox)
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("Tidak ada objek yang memenuhi filter confidence saat ini.")

                # Tabel gizi (Markdown dari LLM)
                st.subheader("🥗 Kandungan Gizi")
                if gizi_text:
                    if render_markdown_table:
                        st.markdown(gizi_text)
                    else:
                        st.text(gizi_text)
                else:
                    st.info("API belum mengembalikan teks gizi.")
        st.toast("Selesai memproses ✅")
    except requests.exceptions.ConnectionError:
        st.error(f"Gagal terhubung ke API di {api_url}. Pastikan API berjalan dan URL benar.")
    except requests.exceptions.Timeout:
        st.error("Permintaan ke API melebihi batas waktu (timeout). Coba lagi.")
    except requests.HTTPError as he:
        try:
            err_json = he.response.json()
        except Exception:
            err_json = he.response.text
        st.error(f"API mengembalikan error {he.response.status_code}: {err_json}")
    except Exception as e:
        st.exception(e)

