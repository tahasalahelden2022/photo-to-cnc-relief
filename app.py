
import io
import base64
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, ImageFilter
import trimesh
import relief as rl

st.set_page_config(page_title="Photo → CNC Relief", layout="wide")
st.title("Photo → 3D Relief → STL")
st.caption("حوّل صورة بورتريه إلى نموذج Relief قابل للتصدير كـ STL لاستخدامه في CAM/CNC.")


DEPTH_MODELS = {
    "Small (سريع – مسموح تجاريًا)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base (أدق – غير تجاري)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large (الأدق والأبطأ – غير تجاري)": "depth-anything/Depth-Anything-V2-Large-hf",
}
BG_MODELS = {
    "بدون عزل": None,
    "أشخاص (u2net_human_seg)": "u2net_human_seg",
    "عام (isnet-general-use)": "isnet-general-use",
}


@st.cache_resource
def load_depth_model(repo):
    import torch
    from transformers import AutoModelForDepthEstimation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForDepthEstimation.from_pretrained(repo).to(device).eval()
    return model, device


def estimate_depth(img, repo, size):
    """Raw float depth (larger = nearer) at the photo's resolution.
    Preprocessing done here (resize to multiple of 14 + ImageNet normalization),
    so no image-processor / torchvision dependency is needed."""
    import torch
    model, device = load_depth_model(repo)
    w, h = img.size
    sc = size / min(w, h)
    nw, nh = [max(14, int(round(v * sc / 14)) * 14) for v in (w, h)]
    x = np.asarray(img.resize((nw, nh), Image.Resampling.BICUBIC), np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(device)
    with torch.no_grad():
        pred = model(pixel_values=x).predicted_depth
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(h, w),
                                           mode="bicubic", align_corners=False)
    return pred.squeeze().float().cpu().numpy()


@st.cache_resource
def load_bg_session(name):
    from rembg import new_session
    return new_session(name)


def subject_mask(img, name):
    if not name:
        return np.ones((img.height, img.width), np.float32)
    from rembg import remove
    m = remove(img, session=load_bg_session(name), only_mask=True)
    return np.asarray(m, dtype=np.float32) / 255.0


def make_relief(z_img, width_mm):
    """Build a closed relief mesh (flat base at z=0) from a height grid."""
    # Image row 0 is the top of the photo, but mesh row 0 sits at y=-height/2 (bottom),
    # so flip vertically; otherwise the STL is a mirrored copy of the photo.
    z = np.ascontiguousarray(z_img[::-1], dtype=np.float32)
    h, w = z.shape
    height_mm = width_mm * h / w

    xs = np.linspace(-width_mm / 2, width_mm / 2, w, dtype=np.float32)
    ys = np.linspace(-height_mm / 2, height_mm / 2, h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    top = np.column_stack((X.ravel(), Y.ravel(), z.ravel()))
    bottom = np.column_stack((X.ravel(), Y.ravel(), np.zeros(w * h, dtype=np.float32)))
    vertices = np.vstack([top, bottom])
    off = w * h

    # Top and bottom grids (vectorized).
    r, c = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing="ij")
    a = (r * w + c).ravel()
    b = a + 1
    cc = a + w + 1
    d0 = a + w
    top_faces = np.concatenate([np.stack([a, b, cc], 1), np.stack([a, cc, d0], 1)])
    bot_faces = np.concatenate([np.stack([a, cc, b], 1), np.stack([a, d0, cc], 1)]) + off

    # Four side walls, wound opposite to the boundary edges of the top/bottom grids
    # so the mesh is watertight with consistent outward normals.
    def wall(p, q):
        # p -> q is the direction the wall uses along the top boundary edge.
        return np.concatenate([np.stack([p, q, q + off], 1), np.stack([p, q + off, p + off], 1)])

    cols = np.arange(w - 1)
    rows = np.arange(h - 1)
    last = (h - 1) * w
    walls = np.concatenate([
        wall(cols + 1, cols),                 # first row
        wall(last + cols, last + cols + 1),   # last row
        wall(rows * w, (rows + 1) * w),       # first column
        wall((rows + 1) * w + w - 1, rows * w + w - 1),  # last column
    ])

    faces = np.vstack([top_faces, bot_faces, walls]).astype(np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    return mesh


def viewer_html(z_img, width_mm, base_mm, max_side=400, height_px=600):
    """Interactive three.js preview of the relief (same heights as the STL)."""
    step = max(1, int(np.ceil(max(z_img.shape) / max_side)))
    z = np.ascontiguousarray(z_img[::step, ::step], dtype=np.float32)
    h, w = z.shape
    height_mm = width_mm * z_img.shape[0] / z_img.shape[1]
    z_b64 = base64.b64encode(z.tobytes()).decode("ascii")

    return f"""
<meta charset="utf-8">
<div id="wrap" style="position:relative;width:100%;height:{height_px}px;border-radius:10px;overflow:hidden;background:#1e2227">
  <div id="hint" style="position:absolute;left:12px;bottom:10px;color:#aab;font:13px sans-serif;z-index:2;direction:rtl">
    اسحب للدوران · عجلة الماوس للتكبير · زر يمين للتحريك
  </div>
  <div style="position:absolute;right:12px;top:10px;z-index:2;display:flex;gap:6px">
    <button id="wire" style="padding:4px 10px;border-radius:6px;border:0;cursor:pointer">Wireframe</button>
    <button id="reset" style="padding:4px 10px;border-radius:6px;border:0;cursor:pointer">Reset</button>
  </div>
</div>
<script type="importmap">
{{"imports": {{
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const W={w}, H={h}, WMM={width_mm}, HMM={height_mm}, BASE={base_mm};
const raw = atob("{z_b64}");
const bytes = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
const z = new Float32Array(bytes.buffer);

const wrap = document.getElementById('wrap');
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(wrap.clientWidth, wrap.clientHeight);
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1e2227);

// Relief surface: PlaneGeometry vertices go row by row from the top of the image.
const geo = new THREE.PlaneGeometry(WMM, HMM, W - 1, H - 1);
const pos = geo.attributes.position;
for (let i = 0; i < pos.count; i++) pos.setZ(i, z[i]);
geo.computeVertexNormals();
const mat = new THREE.MeshStandardMaterial({{ color: 0xd9b38c, roughness: 0.75, metalness: 0.0 }});
const surface = new THREE.Mesh(geo, mat);
scene.add(surface);

// Base block (0 .. BASE mm).
const baseGeo = new THREE.BoxGeometry(WMM, HMM, BASE);
const base = new THREE.Mesh(baseGeo, new THREE.MeshStandardMaterial({{ color: 0xb08a66, roughness: 0.85 }}));
base.position.z = BASE / 2 - 0.01;
scene.add(base);

scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 0.9));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(-WMM, HMM, WMM * 1.2);
scene.add(key);
const rim = new THREE.DirectionalLight(0xffffff, 0.4);
rim.position.set(WMM, -HMM, WMM * 0.5);
scene.add(rim);

const size = Math.max(WMM, HMM);
const camera = new THREE.PerspectiveCamera(35, wrap.clientWidth / wrap.clientHeight, size / 100, size * 20);
const home = new THREE.Vector3(0, -size * 0.35, size * 1.7);
camera.position.copy(home);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, BASE);
controls.update();

document.getElementById('wire').onclick = () => {{ mat.wireframe = !mat.wireframe; }};
document.getElementById('reset').onclick = () => {{
  camera.position.copy(home); controls.target.set(0, 0, BASE); controls.update();
}};

new ResizeObserver(() => {{
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  camera.aspect = wrap.clientWidth / wrap.clientHeight;
  camera.updateProjectionMatrix();
}}).observe(wrap);

(function loop() {{ requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); }})();
</script>
"""


# ================================================================ history
import json
import re
import shutil
import hashlib
import datetime
from pathlib import Path

HIST_DIR = Path(__file__).resolve().parent / "history"

MODES = ["صورة منحوتة جاهزة (مولّدة بالذكاء الاصطناعي)", "صورة فوتوغرافية عادية"]
PRESETS = {
    MODES[0]: dict(resolution=768, infer_size=1022, bg_label=list(BG_MODELS)[0],
                   use_bas=False, img_detail_mm=0.12, use_frame=False),
    MODES[1]: dict(resolution=512, infer_size=770, bg_label=list(BG_MODELS)[1],
                   use_bas=True, img_detail_mm=0.25, use_frame=True),
}
DEFAULTS = dict(
    mode=MODES[0], width_mm=150.0, relief_mm=8.0, base_mm=4.0,
    model_label=list(DEPTH_MODELS)[0], invert=False,
    compress=2.0, detail=1.0, pillow=0.2, frame_mm=10.0, frame_h_mm=9.0,
    crop_top=0, crop_bottom=0, crop_left=0, crop_right=0,
    **PRESETS[MODES[0]],
)
SETTING_KEYS = list(DEFAULTS)

for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("depth_cache", {})


def apply_preset():
    for k, v in PRESETS[st.session_state["mode"]].items():
        st.session_state[k] = v


def slugify(name):
    s = re.sub(r"[^\w\-]+", "_", Path(name).stem, flags=re.UNICODE).strip("_")
    return (s or "relief")[:40]


def list_entries():
    if not HIST_DIR.exists():
        return []
    out = []
    for d in sorted(HIST_DIR.iterdir(), reverse=True):
        mf = d / "meta.json"
        if d.is_dir() and mf.exists():
            try:
                meta = json.loads(mf.read_text(encoding="utf-8"))
                meta["dir"] = d
                out.append(meta)
            except Exception:
                pass
    return out


def save_entry(job_name, original, depth_key, depth, z, shade, settings, stats):
    ts = datetime.datetime.now()
    d = HIST_DIR / f"{ts:%Y%m%d_%H%M%S}_{slugify(job_name)}"
    d.mkdir(parents=True, exist_ok=True)
    original.save(d / "original.png")
    np.savez_compressed(d / "data.npz", depth=depth.astype(np.float32), z=z.astype(np.float32))
    thumb = shade.copy()
    thumb.thumbnail((360, 360))
    thumb.save(d / "thumb.png")
    meta = dict(id=d.name, name=job_name, created=f"{ts:%Y-%m-%d %H:%M}",
                depth_key=depth_key, settings=settings, stats=stats)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return d.name


def open_entry(entry_id):
    """Callback: load an entry's image, settings and result (runs before widgets are drawn)."""
    d = HIST_DIR / entry_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for k, v in meta["settings"].items():
        if k in DEFAULTS:
            if k == "model_label" and v not in DEPTH_MODELS:
                continue
            if k == "bg_label" and v not in BG_MODELS:
                continue
            st.session_state[k] = v
    st.session_state["job_name"] = meta["name"]
    data = np.load(d / "data.npz")
    st.session_state["depth_cache"][meta["depth_key"]] = data["depth"]
    image = Image.open(d / "original.png").convert("RGB")
    src_id = f"hist:{entry_id}"
    st.session_state["src"] = {"id": src_id, "name": meta["name"], "image": image}
    st.session_state["uploader_key"] += 1          # clear the uploader
    z = data["z"]
    s = meta["settings"]
    st.session_state["result"] = {
        "src_id": src_id, "entry": entry_id, "stl": None, "z": z,
        "shade": Image.fromarray(rl.hillshade(z, z_scale=z.shape[1] / s["width_mm"])),
        "depth_img": Image.fromarray((rl.robust_normalize(data["depth"]) * 255).astype(np.uint8)),
        "mask_img": None, "width_mm": s["width_mm"], "base_mm": s["base_mm"], **meta["stats"],
    }


def delete_entry(entry_id):
    shutil.rmtree(HIST_DIR / entry_id, ignore_errors=True)
    res = st.session_state.get("result")
    if res and res.get("entry") == entry_id:
        res["entry"] = None


def depth_key_for(image, model_repo, size):
    h = hashlib.sha1(image.tobytes())
    h.update(f"{image.size}|{model_repo}|{size}".encode())
    return h.hexdigest()[:16]


# ================================================================ sidebar
with st.sidebar:
    st.header("الإعدادات")
    st.radio("نوع الصورة", MODES, key="mode", on_change=apply_preset)
    st.subheader("المقاسات")
    st.number_input("العرض الكلي mm", 20.0, 1000.0, step=5.0, key="width_mm")
    st.number_input("عمق النحت mm", 0.5, 100.0, step=0.5, key="relief_mm")
    st.number_input("سمك القاعدة mm", 1.0, 100.0, step=0.5, key="base_mm")
    st.selectbox("دقة الشبكة (نقطة بالعرض)", [384, 512, 768, 1024], key="resolution",
                 help="768 أو أكثر يعطي ملف STL كبير جدًا (أكثر من 100 ميجا)")

    st.subheader("الذكاء الاصطناعي")
    st.selectbox("نموذج العمق", list(DEPTH_MODELS), key="model_label")
    st.selectbox("دقة تحليل العمق", [518, 770, 1022], key="infer_size", help="أعلى = تفاصيل أكثر وأبطأ")
    st.selectbox("عزل الخلفية", list(BG_MODELS), key="bg_label")
    st.checkbox("عكس العمق", key="invert", help="فعّله فقط لو الوجه طلع غائر بدل بارز")

    st.subheader("شكل النحت")
    st.checkbox("ضغط Bas-relief", key="use_bas")
    nb = not st.session_state["use_bas"]
    st.slider("قوة الضغط", 0.0, 10.0, step=0.5, key="compress", disabled=nb)
    st.slider("إبراز التفاصيل", 0.0, 4.0, step=0.1, key="detail", disabled=nb)
    st.slider("استدارة الحواف", 0.0, 0.6, step=0.05, key="pillow", disabled=nb)
    st.slider("تفاصيل دقيقة من الصورة mm", 0.0, 1.5, step=0.01, key="img_detail_mm",
              help="خصلات الشعر والتجاعيد وثنيات القماش")

    st.subheader("البرواز")
    st.checkbox("إضافة برواز", key="use_frame")
    nf = not st.session_state["use_frame"]
    st.number_input("عرض البرواز mm", 2.0, 100.0, step=1.0, key="frame_mm", disabled=nf)
    st.number_input("ارتفاع البرواز mm", 0.5, 50.0, step=0.5, key="frame_h_mm", disabled=nf)


# ================================================================ history panel
entries = list_entries()
with st.expander(f"📁 السجل — {len(entries)} عمل محفوظ", expanded=False):
    if not entries:
        st.caption("كل عملية «إنشاء STL» تُحفَظ هنا تلقائيًا مع إعداداتها، للرجوع إليها أو تعديلها.")
    cols_per_row = 4
    for i in range(0, len(entries), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, e in zip(cols, entries[i:i + cols_per_row]):
            with col, st.container(border=True):
                thumb = e["dir"] / "thumb.png"
                if thumb.exists():
                    st.image(str(thumb), use_container_width=True)
                st.markdown(f"**{e['name']}**")
                sx, sy, sz = e["stats"]["size_mm"]
                st.caption(f"{e['created']} — {sx:.0f}×{sy:.0f}×{sz:.1f} mm")
                b1, b2 = st.columns(2)
                b1.button("فتح / تعديل", key=f"open_{e['id']}", on_click=open_entry, args=(e["id"],),
                          use_container_width=True)
                with b2.popover("حذف", use_container_width=True):
                    st.write("حذف هذا العمل نهائيًا؟")
                    st.button("نعم، احذف", key=f"del_{e['id']}", on_click=delete_entry, args=(e["id"],),
                              type="primary")


# ================================================================ image source
uploaded = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png", "webp"],
                            key=f"uploader_{st.session_state['uploader_key']}")
if uploaded is not None:
    up_id = f"up:{uploaded.name}:{uploaded.size}"
    src = st.session_state.get("src")
    if not src or src["id"] != up_id:
        st.session_state["src"] = {"id": up_id, "name": Path(uploaded.name).stem,
                                   "image": ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")}
        st.session_state["job_name"] = Path(uploaded.name).stem

src = st.session_state.get("src")

if src:
    original = src["image"]
    if src["id"].startswith("hist:"):
        st.info(f"مفتوح من السجل: **{src['name']}** — عدّل الإعدادات واضغط «إنشاء STL» لحفظ نسخة جديدة "
                "(النسخة القديمة تبقى كما هي).")

    st.text_input("اسم العمل", key="job_name")

    with st.expander("قص الصورة (يفضَّل التركيز على الرأس والكتفين)"):
        c1, c2, c3, c4 = st.columns(4)
        c1.slider("من أعلى %", 0, 45, key="crop_top")
        c2.slider("من أسفل %", 0, 70, key="crop_bottom")
        c3.slider("من اليسار %", 0, 45, key="crop_left")
        c4.slider("من اليمين %", 0, 45, key="crop_right")
    S = {k: st.session_state[k] for k in SETTING_KEYS}
    W0, H0 = original.size
    image = original.crop((int(W0 * S["crop_left"] / 100), int(H0 * S["crop_top"] / 100),
                           int(W0 * (1 - S["crop_right"] / 100)), int(H0 * (1 - S["crop_bottom"] / 100))))
    st.image(image, caption=f"الصورة المستخدمة ({image.width}×{image.height})", width=350)
    if max(image.size) < 700:
        st.warning("الصورة صغيرة الدقة؛ النتيجة تتحسن كثيرًا مع صورة أوضح (1000 بكسل أو أكثر).")

    if st.button("إنشاء STL", type="primary"):
        repo = DEPTH_MODELS[S["model_label"]]
        dkey = depth_key_for(image, repo, S["infer_size"])
        cache = st.session_state["depth_cache"]
        if dkey in cache:
            depth = cache[dkey]
        else:
            with st.spinner("تحليل العمق..."):
                depth = estimate_depth(image, repo, S["infer_size"])
            if len(cache) >= 4:
                cache.pop(next(iter(cache)))
            cache[dkey] = depth
        d_used = -depth if S["invert"] else depth
        with st.spinner("عزل الخلفية..."):
            try:
                mask = subject_mask(image, BG_MODELS[S["bg_label"]])
            except ImportError:
                st.error("مكتبة rembg غير مثبتة — أعد تشغيل run.bat لتثبيتها. سيكمل بدون عزل.")
                mask = np.ones(depth.shape, np.float32)
        with st.spinner("بناء النحت والموديل ثلاثي الأبعاد..."):
            z, pv = rl.build_heightmap(
                np.asarray(image), d_used, mask, grid_w=S["resolution"], width_mm=S["width_mm"],
                relief_mm=S["relief_mm"], base_mm=S["base_mm"], use_bas=S["use_bas"],
                compress=S["compress"], detail=S["detail"], pillow=S["pillow"],
                img_detail_mm=S["img_detail_mm"],
                frame_mm=S["frame_mm"] if S["use_frame"] else 0.0,
                frame_h_mm=S["frame_h_mm"] if S["use_frame"] else 0.0)
            mesh = make_relief(z, S["width_mm"])
            stl_bytes = io.BytesIO()
            mesh.export(stl_bytes, file_type="stl")

        shade = Image.fromarray(rl.hillshade(z, z_scale=S["resolution"] / S["width_mm"]))
        stats = {"size_mm": [S["width_mm"], S["width_mm"] * z.shape[0] / z.shape[1], float(z.max())],
                 "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight)}
        job = (st.session_state.get("job_name") or src["name"]).strip() or "relief"
        entry_id = save_entry(job, original, dkey, depth, z, shade, S, stats)

        st.session_state["result"] = {
            "src_id": src["id"], "entry": entry_id, "stl": stl_bytes.getvalue(), "z": z, "shade": shade,
            "depth_img": Image.fromarray((rl.robust_normalize(depth) * 255).astype(np.uint8)),
            "mask_img": Image.fromarray((pv["mask"] * 255).astype(np.uint8)),
            "width_mm": S["width_mm"], "base_mm": S["base_mm"], **stats,
        }
        st.rerun()   # refresh the history panel

    res = st.session_state.get("result")
    if res and res["src_id"] == src["id"]:
        sx, sy, sz = res["size_mm"]
        st.success(f"الموديل: {res['faces']:,} مثلث — المقاس {sx:.0f}×{sy:.0f}×{sz:.1f} mm"
                   f" — Watertight: {res['watertight']}"
                   + (" — محفوظ في السجل ✓" if res.get("entry") else ""))

        view_col, side_col = st.columns([3, 1])
        with view_col:
            st.subheader("معاينة 3D")
            components.html(viewer_html(res["z"], res["width_mm"], res["base_mm"], max_side=500), height=610)
        with side_col:
            st.subheader("معاينة مظللة")
            st.image(res["shade"])
            st.caption("خريطة العمق" + (" / قناع الخلفية" if res["mask_img"] is not None else ""))
            st.image([im for im in (res["depth_img"], res["mask_img"]) if im is not None], width=120)

        fname = f"{slugify(st.session_state.get('job_name') or 'relief')}.stl"
        if res["stl"] is None:
            if st.button("تجهيز ملف STL للتحميل"):
                with st.spinner("بناء ملف STL..."):
                    mesh = make_relief(res["z"], res["width_mm"])
                    b = io.BytesIO()
                    mesh.export(b, file_type="stl")
                    res["stl"] = b.getvalue()
                st.rerun()
        else:
            st.download_button("تحميل STL", data=res["stl"], file_name=fname, mime="application/sla")

        st.info(
            "مهم: قبل CNC افحص الـSTL في برنامج CAM، واضبط الـtoolpath والـstock والـstepover والأداة حسب الخامة. "
            "للتفاصيل الدقيقة استخدم Ball Nose صغير (1 mm أو أقل) في التشطيب."
        )
