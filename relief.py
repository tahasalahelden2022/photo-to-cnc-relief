"""Height-map processing for portrait bas-reliefs (numpy / scipy / PIL only)."""
import numpy as np
from PIL import Image
from scipy import ndimage, fft


# ---------------------------------------------------------------- helpers
def robust_normalize(a, mask=None, lo_p=1.0, hi_p=99.0):
    v = a[mask > 0.5] if mask is not None and (mask > 0.5).any() else a
    lo, hi = np.percentile(v, lo_p), np.percentile(v, hi_p)
    return np.clip((a - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def resize_float(a, size_wh, resample=Image.Resampling.BICUBIC):
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(size_wh, resample),
                      dtype=np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-8), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def poisson_neumann(div):
    """Solve  laplacian(u) = div  on a grid with Neumann borders (DCT, fast)."""
    h, w = div.shape
    d = fft.dctn(div, type=2, norm="ortho")
    ky = np.cos(np.pi * np.arange(h) / h)[:, None]
    kx = np.cos(np.pi * np.arange(w) / w)[None, :]
    denom = 2 * ky + 2 * kx - 4
    denom[0, 0] = 1.0
    d /= denom
    d[0, 0] = 0.0
    return fft.idctn(d, type=2, norm="ortho").astype(np.float32)


# ---------------------------------------------------------------- bas-relief
def bas_relief(depth, mask, compress=5.0, detail=1.5, detail_sigma=4.0,
               edge_thresh=0.08, pillow=0.35, pillow_px=None):
    """
    Gradient-domain bas-relief compression.
    depth : float array, larger = nearer (any scale)
    mask  : 0..1 subject mask (1 = subject)
    Returns heights 0..1 (background = 0).
    """
    h, w = depth.shape
    m = np.clip(mask, 0, 1).astype(np.float32)
    hard = m > 0.5

    d = ndimage.gaussian_filter(robust_normalize(depth, m), 1.0) * (0.25 * w)  # pixel units

    gx = np.zeros_like(d); gy = np.zeros_like(d)
    gx[:, :-1] = d[:, 1:] - d[:, :-1]
    gy[:-1, :] = d[1:, :] - d[:-1, :]

    # Drop silhouette jumps and anything outside the subject.
    inside_x = np.zeros_like(hard); inside_x[:, :-1] = hard[:, 1:] & hard[:, :-1]
    inside_y = np.zeros_like(hard); inside_y[:-1, :] = hard[1:, :] & hard[:-1, :]
    big = edge_thresh * 0.25 * w
    gx[(~inside_x) | (np.abs(gx) > big)] = 0
    gy[(~inside_y) | (np.abs(gy) > big)] = 0

    # Non-linear attenuation: large slopes shrink a lot, fine detail stays.
    a = float(compress)
    if a > 0:
        gx = np.sign(gx) * np.log1p(a * np.abs(gx)) / a
        gy = np.sign(gy) * np.log1p(a * np.abs(gy)) / a

    # Unsharp masking in the gradient domain boosts small features.
    if detail > 0:
        gx = gx + detail * (gx - ndimage.gaussian_filter(gx, detail_sigma))
        gy = gy + detail * (gy - ndimage.gaussian_filter(gy, detail_sigma))

    div = np.zeros_like(d)
    div[:, 0] += gx[:, 0]; div[:, 1:] += gx[:, 1:] - gx[:, :-1]
    div[0, :] += gy[0, :]; div[1:, :] += gy[1:, :] - gy[:-1, :]
    u = poisson_neumann(div)

    u = robust_normalize(u, m, 0.5, 99.5)

    # "Pillow": rounded rise at the silhouette so the figure sits on the panel.
    if pillow_px is None:
        pillow_px = max(4.0, 0.06 * w)
    dist = ndimage.distance_transform_edt(np.pad(hard, 1))[1:-1, 1:-1]
    prof = 1.0 - (1.0 - np.clip(dist / pillow_px, 0, 1)) ** 2
    prof = ndimage.gaussian_filter(prof, 1.5)  # remove stair-steps from upscaled masks
    height = pillow * prof + (1.0 - pillow) * u * np.sqrt(prof)

    soft = ndimage.gaussian_filter(m, 0.8)
    return np.clip(height * soft, 0, None).astype(np.float32)


def image_detail(rgb, mask, sigma=3.0):
    """High-pass luminance detail (-1..1): hair strands, wrinkles, fabric folds."""
    g = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255.0
    g = ndimage.gaussian_filter(g, 0.7)            # kill JPEG noise
    hp = g - ndimage.gaussian_filter(g, sigma)
    s = hp[mask > 0.5].std() if (mask > 0.5).any() else hp.std()
    hp = np.tanh(hp / (2.5 * s + 1e-6))
    return (hp * ndimage.gaussian_filter(np.clip(mask, 0, 1), 1.5)).astype(np.float32)


# ---------------------------------------------------------------- frame
def frame_profile(h, w, frame_px, frame_h):
    """Moulded frame height (mm) on an h x w grid; 0 inside the frame opening."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy])
    t = dist / max(frame_px, 1)
    prof = np.where(t < 0.1, 0.5 * smoothstep(0.0, 0.1, t), 0.5)            # outer bevel + flat band
    s = np.clip((t - 0.1) / 0.6, 0, 1)
    prof = prof + np.where((t >= 0.1) & (t < 0.7), 0.5 * np.clip(np.sin(np.pi * s), 0, 1) ** 0.7, 0)  # rounded bead
    prof = np.where(t >= 0.8, 0.5 * (1 - smoothstep(0.8, 1.0, t)), prof)     # cove down to the panel
    prof[t >= 1.0] = 0.0
    return (prof * frame_h).astype(np.float32), t < 1.0


def hillshade(z, az=315, alt=45, z_scale=1.0):
    """Quick shaded preview image (uint8) of a height grid."""
    gy, gx = np.gradient(z * z_scale)
    nx, ny, nz = -gx, -gy, np.ones_like(z)
    n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    az, alt = np.radians(az), np.radians(alt)
    lx, ly, lz = np.cos(alt) * np.cos(az), -np.cos(alt) * np.sin(az), np.sin(alt)
    shade = np.clip((nx * lx + ny * ly + nz * lz) / n, 0, 1)
    return (40 + 215 * shade).astype(np.uint8)


# ---------------------------------------------------------------- full pipeline
def build_heightmap(rgb, depth, mask, grid_w, width_mm, relief_mm, base_mm,
                    use_bas=True, compress=5.0, detail=1.5, pillow=0.35,
                    img_detail_mm=0.4, frame_mm=0.0, frame_h_mm=0.0):
    """
    rgb   : HxWx3 uint8 photo
    depth : HxW float, larger = nearer
    mask  : HxW 0..1 subject mask (ones = keep everything)
    Returns (z_mm, preview) where z_mm is in image orientation and includes the base.
    """
    H0, W0 = depth.shape
    ppm = grid_w / width_mm
    frame_px = int(round(frame_mm * ppm)) if frame_mm > 0 else 0
    inner_w = max(16, grid_w - 2 * frame_px)
    inner_h = max(16, int(round(inner_w * H0 / W0)))

    rgb_r = np.asarray(Image.fromarray(rgb).resize((inner_w, inner_h), Image.Resampling.LANCZOS))
    depth_r = resize_float(depth, (inner_w, inner_h))
    mask_r = np.clip(resize_float(mask, (inner_w, inner_h), Image.Resampling.BILINEAR), 0, 1)

    if use_bas:
        hts = bas_relief(depth_r, mask_r, compress=compress, detail=detail, pillow=pillow)
    else:
        hts = robust_normalize(depth_r, mask_r) * mask_r

    z = hts * relief_mm
    if img_detail_mm > 0:
        z = z + img_detail_mm * image_detail(rgb_r, mask_r)
    z = np.clip(z, 0, None)

    H, W = inner_h + 2 * frame_px, inner_w + 2 * frame_px
    full = np.zeros((H, W), np.float32)
    full[frame_px:frame_px + inner_h, frame_px:frame_px + inner_w] = z
    if frame_px > 0 and frame_h_mm > 0:
        prof, in_frame = frame_profile(H, W, frame_px, frame_h_mm)
        full = np.where(in_frame, prof, full)

    return (base_mm + full).astype(np.float32), {"heights": hts, "mask": mask_r}
