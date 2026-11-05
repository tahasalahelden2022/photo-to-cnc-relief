
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import torch
from transformers import pipeline
import trimesh

st.set_page_config(page_title="Photo → CNC Relief", layout="wide")
st.title("Photo → 3D Relief → STL")
st.caption("حوّل صورة بورتريه إلى نموذج Relief قابل للتصدير كـ STL لاستخدامه في CAM/CNC.")

@st.cache_resource
def load_depth_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device
    )

def make_relief(depth, width_mm, relief_mm, base_mm, invert=True, smooth=1.2):
    d = np.asarray(depth, dtype=np.float32)
    d -= d.min()
    if d.max() > 1e-8:
        d /= d.max()

    # For a portrait relief, nearer/high-interest regions should normally become raised.
    if invert:
        d = 1.0 - d

    # Reduce tiny depth noise.
    if smooth > 0:
        p = Image.fromarray((d * 255).astype(np.uint8))
        p = p.filter(ImageFilter.GaussianBlur(radius=float(smooth)))
        d = np.asarray(p, dtype=np.float32) / 255.0

    h, w = d.shape
    height_mm = width_mm * h / w

    xs = np.linspace(-width_mm/2, width_mm/2, w, dtype=np.float32)
    ys = np.linspace(-height_mm/2, height_mm/2, h, dtype=np.float32)

    # z=base at the back and base+relief at the highest point.
    z = base_mm + d * relief_mm

    # vertices: top surface
    X, Y = np.meshgrid(xs, ys)
    top = np.column_stack((X.ravel(), Y.ravel(), z.ravel()))

    # bottom surface
    bottom = np.column_stack((X.ravel(), Y.ravel(), np.zeros(w*h, dtype=np.float32)))

    vertices = np.vstack([top, bottom])
    faces = []

    def idx(r, c):
        return r*w + c

    # Top and bottom grids.
    for r in range(h-1):
        for c in range(w-1):
            a, b, cc, d0 = idx(r,c), idx(r,c+1), idx(r+1,c+1), idx(r+1,c)
            faces.append((a,b,cc))
            faces.append((a,cc,d0))

            off = w*h
            a2,b2,c2,d2 = a+off,b+off,cc+off,d0+off
            faces.append((a2,c2,b2))
            faces.append((a2,d2,c2))

    # Four side walls.
    for c in range(w-1):
        a,b = idx(0,c), idx(0,c+1)
        faces.extend([(a,b,b+w*h), (a,b+w*h,a+w*h)])

        a,b = idx(h-1,c), idx(h-1,c+1)
        a += w*h; b += w*h
        faces.extend([(a,b,b-w*h), (a,a-w*h,b-w*h)])

    for r in range(h-1):
        a,b = idx(r,0), idx(r+1,0)
        faces.extend([(a,b,b+w*h), (a,b+w*h,a+w*h)])

        a,b = idx(r,w-1), idx(r+1,w-1)
        a += w*h; b += w*h
        faces.extend([(a,b,b-w*h), (a,a-w*h,b-w*h)])

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=True
    )
    mesh.remove_unreferenced_vertices()
    return mesh

uploaded = st.file_uploader("ارفع صورة الشخص", type=["jpg","jpeg","png","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="الصورة الأصلية", width=450)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        width_mm = st.number_input("العرض mm", 20.0, 1000.0, 120.0, 5.0)
    with col2:
        relief_mm = st.number_input("عمق النحت mm", 0.5, 100.0, 8.0, 0.5)
    with col3:
        base_mm = st.number_input("سمك القاعدة mm", 1.0, 100.0, 3.0, 0.5)
    with col4:
        resolution = st.selectbox("دقة الشبكة", [256, 384, 512, 768], index=1)

    invert = st.checkbox("عكس العمق (جرّب الخيارين)", value=True)
    smooth = st.slider("تنعيم Depth", 0.0, 5.0, 1.2, 0.1)

    if st.button("إنشاء STL", type="primary"):
        with st.spinner("تشغيل نموذج Depth..."):
            model = load_depth_model()
            # Limit input dimensions before inference while preserving aspect ratio.
            img = image.copy()
            max_side = 1024
            scale = min(1.0, max_side / max(img.size))
            if scale < 1:
                img = img.resize((int(img.width*scale), int(img.height*scale)), Image.Resampling.LANCZOS)

            result = model(img)
            depth_img = result["depth"].convert("L")
            depth_img = ImageOps.fit(depth_img, (resolution, int(resolution*img.height/img.width)),
                                     method=Image.Resampling.BICUBIC)

            depth = np.asarray(depth_img, dtype=np.float32) / 255.0
            mesh = make_relief(depth, width_mm, relief_mm, base_mm, invert, smooth)

        st.success(f"تم إنشاء الموديل: {len(mesh.vertices):,} vertices / {len(mesh.faces):,} triangles")
        st.write("Watertight:", mesh.is_watertight)

        stl_bytes = io.BytesIO()
        mesh.export(stl_bytes, file_type="stl")
        stl_bytes.seek(0)

        st.download_button(
            "تحميل STL",
            data=stl_bytes.getvalue(),
            file_name="portrait_relief.stl",
            mime="application/sla"
        )

        st.info(
            "مهم: هذا نموذج Depth أحادي الصورة. قبل CNC افحص الـSTL في برنامج CAM، "
            "واضبط الـtoolpath والـstock والـstepover والأداة حسب الخامة."
        )
