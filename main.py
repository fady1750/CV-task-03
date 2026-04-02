"""
main.py – Streamlit application for Assignment 3

Tasks
─────
  1. Harris Corner Detection  (Harris + λ₋, with timing)
  2. SIFT Feature Descriptors (from Harris keypoints, with timing)
  3. Feature Matching         (SSD + NCC, with timing)

Run with:
    streamlit run main.py
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from src.filters import HarrisDetector, SIFTDescriptor, SSDMatcher, NCCMatcher
from src.utils   import (
    to_grayscale, normalize_image,
    fig_keypoints, fig_response_map, fig_matches,
    fig_descriptor_heatmaps, create_sample_images,
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Feature Lab – Assignment 3",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* metric cards */
      div[data-testid="metric-container"] {
          background: #1a1d2e;
          border: 1px solid #2d3150;
          border-radius: 8px;
          padding: 10px 14px;
      }
      /* section divider */
      hr { border-color: #2d3150; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR – navigation & parameters
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 CV Feature Lab")
    st.caption("Assignment 3 – Feature Detection & Matching")
    st.markdown("---")

    task = st.radio(
        "**Task**",
        [
            "🏁  Harris Corner Detection",
            "📐  SIFT Feature Descriptors",
            "🔗  Feature Matching (SSD & NCC)",
        ],
        label_visibility="collapsed",
    )

    needs_two_images = task.startswith("🔗")

    # ── Harris parameters ────────────────────────────────────
    st.markdown("---")
    with st.expander("⚙️ Harris Parameters", expanded=True):
        harris_k       = st.slider("k constant",        0.01, 0.10, 0.04, 0.005,
                                   help="Harris sensitivity constant (k=0.04 typical).")
        harris_sigma   = st.slider("σ – Gaussian",      0.5,  3.0,  1.0,  0.1,
                                   help="σ for structure-tensor smoothing.")
        harris_thresh  = st.slider("Threshold ratio", 0.001, 0.10, 0.01, 0.001,
                                   format="%.3f",
                                   help="Fraction of max(R) used as corner threshold.")
        harris_nms     = st.slider("NMS window",        3,    15,   7,    2,
                                   help="Side length of non-max suppression window.")

    # ── SIFT parameters ──────────────────────────────────────
    if not task.startswith("🏁"):
        with st.expander("⚙️ SIFT Parameters", expanded=False):
            max_kp = st.slider("Max keypoints", 50, 500, 200, 50,
                               help="Maximum number of keypoints to describe.")
    else:
        max_kp = 200

    # ── Matching parameters ──────────────────────────────────
    if needs_two_images:
        with st.expander("⚙️ Matching Parameters", expanded=False):
            ssd_ratio   = st.slider("SSD Lowe ratio",   0.50, 0.95, 0.75, 0.05,
                                    help="Ratio test threshold. Lower = stricter.")
            ncc_thresh  = st.slider("NCC threshold",    0.50, 0.99, 0.70, 0.05,
                                    help="Minimum NCC score to accept a match.")
            max_matches = st.slider("Displayed matches", 10, 100, 30, 10,
                                    help="Maximum number of match lines to draw.")
    else:
        ssd_ratio, ncc_thresh, max_matches = 0.75, 0.70, 30

    # ── Image input ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📷 Image Input")

# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLES = create_sample_images()

def _image_widget(label: str, key: str) -> np.ndarray | None:
    """Render upload + preset controls in the sidebar; return RGB uint8 array."""
    src = st.sidebar.radio(
        f"{label} source",
        ["🗂 Preset", "⬆️ Upload"],
        key=f"src_{key}",
        horizontal=True,
    )
    if src == "⬆️ Upload":
        f = st.sidebar.file_uploader(
            f"Upload {label}", type=["png", "jpg", "jpeg", "bmp"],
            key=f"up_{key}",
        )
        if f is None:
            return None
        return np.array(Image.open(f).convert("RGB"))
    else:
        name = st.sidebar.selectbox(
            label, list(SAMPLES.keys()), key=f"preset_{key}",
        )
        return SAMPLES[name]


if needs_two_images:
    img1 = _image_widget("Image 1", "1")
    img2 = _image_widget("Image 2", "2")
else:
    img1 = _image_widget("Image", "1")
    img2 = None

# ─────────────────────────────────────────────────────────────────────────────
#  GUARD: require image(s) before proceeding
# ─────────────────────────────────────────────────────────────────────────────
if img1 is None or (needs_two_images and img2 is None):
    st.info("👈  Please select or upload the required image(s) in the sidebar.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  TASK 1 – HARRIS CORNER DETECTION
# ═════════════════════════════════════════════════════════════════════════════
if task.startswith("🏁"):
    st.title("🏁 Harris Corner Detection")
    st.markdown(
        "Detecting corners using **Harris** (R = det − k·tr²) "
        "and **λ₋** (minimum eigenvalue / Shi-Tomasi) — both from scratch."
    )

    # ── Build detectors ───────────────────────────────────────
    det_h = HarrisDetector(
        method="harris",
        k=harris_k, sigma=harris_sigma,
        window_size=5, threshold_ratio=harris_thresh, nms_window=harris_nms,
    )
    det_lm = HarrisDetector(
        method="lambda_minus",
        sigma=harris_sigma,
        window_size=5, threshold_ratio=harris_thresh, nms_window=harris_nms,
    )

    # ── Run detection ─────────────────────────────────────────
    with st.spinner("Running Harris and λ₋ detectors…"):
        t0 = time.perf_counter(); kp_h,  R_h  = det_h.detect(img1);  t_h  = time.perf_counter() - t0
        t0 = time.perf_counter(); kp_lm, R_lm = det_lm.detect(img1); t_lm = time.perf_counter() - t0

    # ── Metrics row ───────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Harris – corners found", f"{len(kp_h):,}")
    m2.metric("Harris – time",          f"{t_h  * 1000:.1f} ms")
    m3.metric("λ₋ – corners found",    f"{len(kp_lm):,}")
    m4.metric("λ₋ – time",             f"{t_lm * 1000:.1f} ms")

    st.markdown("---")

    # ── Response maps ─────────────────────────────────────────
    st.subheader("Corner Response Maps")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Harris response  R = det(M) − k · tr(M)²**")
        fig = fig_response_map(R_h, title="Harris R", cmap="hot")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with c2:
        st.markdown("**λ₋ response  (min eigenvalue of M)**")
        fig = fig_response_map(R_lm, title="λ₋ Response", cmap="plasma")
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")

    # ── Detected corners overlay ──────────────────────────────
    st.subheader("Detected Corners Overlaid on Image")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Harris** — {len(kp_h):,} corners")
        fig = fig_keypoints(img1, kp_h, title="Harris Corners", color="lime")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with c2:
        st.markdown(f"**λ₋ (Shi-Tomasi)** — {len(kp_lm):,} corners")
        fig = fig_keypoints(img1, kp_lm, title="λ₋ Corners", color="cyan")
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")

    # ── Side-by-side comparison at same threshold ─────────────
    st.subheader("Original Image")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img1)
    ax.axis("off")
    ax.set_title("Input Image")
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")

    # ── Timing table ──────────────────────────────────────────
    st.subheader("📊 Computation Time Report")
    st.table({
        "Method":          ["Harris  (R = det − k·tr²)", "λ₋  (min eigenvalue)"],
        "Corners found":   [len(kp_h),                  len(kp_lm)],
        "Time (ms)":       [f"{t_h  * 1000:.2f}",       f"{t_lm * 1000:.2f}"],
        "Speedup":         ["1.00×", f"{t_h / max(t_lm, 1e-9):.2f}×"],
    })


# ═════════════════════════════════════════════════════════════════════════════
#  TASK 2 – SIFT FEATURE DESCRIPTORS
# ═════════════════════════════════════════════════════════════════════════════
elif task.startswith("📐"):
    st.title("📐 SIFT Feature Descriptors")
    st.markdown(
        "Harris keypoints are detected first; then a **128-dim SIFT descriptor** "
        "is computed for each one — entirely from scratch (pure NumPy)."
    )

    # ── Run pipeline ─────────────────────────────────────────
    detector   = HarrisDetector(
        method="harris", k=harris_k, sigma=harris_sigma,
        window_size=5, threshold_ratio=harris_thresh, nms_window=harris_nms,
    )
    descriptor = SIFTDescriptor(max_keypoints=max_kp)

    with st.spinner("Detecting keypoints and computing descriptors…"):
        t0 = time.perf_counter()
        kps, R = detector.detect(img1)
        t_detect = time.perf_counter() - t0

        t0 = time.perf_counter()
        valid_kps, descs = descriptor.describe(img1, kps)
        t_describe = time.perf_counter() - t0

    # ── Metrics ───────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Keypoints detected",    f"{len(kps):,}")
    m2.metric("Detection time",        f"{t_detect  * 1000:.1f} ms")
    m3.metric("Descriptors computed",  f"{len(valid_kps):,}")
    m4.metric("Description time",      f"{t_describe * 1000:.1f} ms")

    st.markdown("---")

    # ── Keypoints visualisation ───────────────────────────────
    st.subheader("Detected Keypoints (Harris → SIFT input)")
    fig = fig_keypoints(img1, valid_kps,
                        title="SIFT Keypoints (Harris)", color="yellow",
                        max_kp=max_kp)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")

    # ── Harris response map ───────────────────────────────────
    st.subheader("Harris Response Map (input to keypoint selection)")
    fig = fig_response_map(R, title="Harris R", cmap="hot")
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")

    # ── Descriptor heatmaps ───────────────────────────────────
    st.subheader("Sample SIFT Descriptor Heatmaps")
    st.caption(
        "Each descriptor is 128-dim, visualised as a 16×8 grid: "
        "16 rows = 4×4 spatial sub-cells, 8 columns = orientation bins (0–7)."
    )
    if len(descs) > 0:
        fig = fig_descriptor_heatmaps(descs, n=min(5, len(descs)))
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.warning(
            "No descriptors computed. "
            "Try lowering the **Threshold ratio** in the sidebar."
        )

    st.markdown("---")

    # ── Descriptor statistics ─────────────────────────────────
    if len(descs) > 0:
        st.subheader("Descriptor Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Descriptor dimension", "128")
        c2.metric("Mean L2 norm (post-norm.)", f"{np.linalg.norm(descs, axis=1).mean():.3f}")
        c3.metric("Mean descriptor entropy",
                  f"{float(np.mean([-np.sum(d * np.log(d + 1e-10)) for d in descs])):.3f}")

    st.markdown("---")

    # ── Timing table ──────────────────────────────────────────
    st.subheader("📊 Computation Time Report")
    st.table({
        "Step":       ["Harris Detection", "SIFT Description", "Total"],
        "Count":      [f"{len(kps):,} keypoints",
                       f"{len(valid_kps):,} descriptors",
                       "—"],
        "Time (ms)":  [f"{t_detect  * 1000:.2f}",
                       f"{t_describe * 1000:.2f}",
                       f"{(t_detect + t_describe) * 1000:.2f}"],
    })


# ═════════════════════════════════════════════════════════════════════════════
#  TASK 3 – FEATURE MATCHING  (SSD & NCC)
# ═════════════════════════════════════════════════════════════════════════════
elif task.startswith("🔗"):
    st.title("🔗 Feature Matching — SSD & NCC")
    st.markdown(
        "Features are extracted from both images using Harris + SIFT, "
        "then matched with **SSD** (Lowe ratio test) and **NCC** — all from scratch."
    )

    # ── Build objects ─────────────────────────────────────────
    detector    = HarrisDetector(
        method="harris", k=harris_k, sigma=harris_sigma,
        window_size=5, threshold_ratio=harris_thresh, nms_window=harris_nms,
    )
    descriptor  = SIFTDescriptor(max_keypoints=max_kp)
    ssd_matcher = SSDMatcher(ratio_threshold=ssd_ratio)
    ncc_matcher = NCCMatcher(threshold=ncc_thresh)

    # ── Run pipeline ─────────────────────────────────────────
    with st.spinner("Extracting features and matching…"):
        # Image 1
        t0 = time.perf_counter()
        kps1, _  = detector.detect(img1)
        vkps1, descs1 = descriptor.describe(img1, kps1)
        t_feat1 = time.perf_counter() - t0

        # Image 2
        t0 = time.perf_counter()
        kps2, _  = detector.detect(img2)
        vkps2, descs2 = descriptor.describe(img2, kps2)
        t_feat2 = time.perf_counter() - t0

        # SSD matching
        t0 = time.perf_counter()
        ssd_matches = ssd_matcher.match(descs1, descs2)
        t_ssd = time.perf_counter() - t0

        # NCC matching
        t0 = time.perf_counter()
        ncc_matches = ncc_matcher.match(descs1, descs2)
        t_ncc = time.perf_counter() - t0

    # ── Metrics ───────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Descriptors – Img 1", f"{len(descs1):,}")
    m2.metric("Descriptors – Img 2", f"{len(descs2):,}")
    m3.metric("SSD matches",         f"{len(ssd_matches):,}")
    m4.metric("NCC matches",         f"{len(ncc_matches):,}")

    st.markdown("---")

    # ── Input images with keypoints ───────────────────────────
    st.subheader("Input Images with Detected Keypoints")
    c1, c2 = st.columns(2)
    with c1:
        fig = fig_keypoints(img1, vkps1,
                            title=f"Image 1 — {len(vkps1):,} keypoints",
                            color="lime")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with c2:
        fig = fig_keypoints(img2, vkps2,
                            title=f"Image 2 — {len(vkps2):,} keypoints",
                            color="lime")
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")

    # ── SSD matches ───────────────────────────────────────────
    st.subheader(
        f"SSD Matches — {len(ssd_matches):,} accepted  "
        f"(Lowe ratio < {ssd_ratio:.2f})"
    )
    if ssd_matches and vkps1 and vkps2:
        fig = fig_matches(
            img1, vkps1, img2, vkps2,
            ssd_matches,
            title=f"SSD Matches  (ratio < {ssd_ratio:.2f})",
            max_matches=max_matches,
        )
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Score distribution
        ssd_scores = [m[2] for m in ssd_matches]
        fig2, ax = plt.subplots(figsize=(8, 2.5))
        ax.hist(ssd_scores, bins=30, color="#4caf50", edgecolor="black", alpha=0.85)
        ax.set_xlabel("SSD score (lower = better)")
        ax.set_ylabel("Count")
        ax.set_title("SSD Score Distribution")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)
    else:
        st.warning(
            "No SSD matches found. "
            "Try raising the **SSD Lowe ratio** or lowering the **Threshold ratio**."
        )

    st.markdown("---")

    # ── NCC matches ───────────────────────────────────────────
    st.subheader(
        f"NCC Matches — {len(ncc_matches):,} accepted  "
        f"(NCC ≥ {ncc_thresh:.2f})"
    )
    if ncc_matches and vkps1 and vkps2:
        fig = fig_matches(
            img1, vkps1, img2, vkps2,
            ncc_matches,
            title=f"NCC Matches  (score ≥ {ncc_thresh:.2f})",
            max_matches=max_matches,
        )
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Score distribution
        ncc_scores = [m[2] for m in ncc_matches]
        fig2, ax = plt.subplots(figsize=(8, 2.5))
        ax.hist(ncc_scores, bins=30, color="#2196f3", edgecolor="black", alpha=0.85)
        ax.set_xlabel("NCC score (higher = better)")
        ax.set_ylabel("Count")
        ax.set_title("NCC Score Distribution")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close(fig2)
    else:
        st.warning(
            "No NCC matches found. "
            "Try lowering the **NCC threshold** slider."
        )

    st.markdown("---")

    # ── SSD vs NCC comparison ─────────────────────────────────
    st.subheader("SSD vs NCC — Side-by-side Summary")
    if ssd_matches and ncc_matches and vkps1 and vkps2:
        c1, c2 = st.columns(2)
        with c1:
            fig = fig_matches(img1, vkps1, img2, vkps2, ssd_matches,
                              title="SSD", max_matches=20)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with c2:
            fig = fig_matches(img1, vkps1, img2, vkps2, ncc_matches,
                              title="NCC", max_matches=20)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown("---")

    # ── Timing table ──────────────────────────────────────────
    st.subheader("📊 Computation Time Report")
    st.table({
        "Step": [
            "Feature extraction – Image 1",
            "Feature extraction – Image 2",
            "SSD Matching",
            "NCC Matching",
            "Total pipeline",
        ],
        "Result": [
            f"{len(descs1):,} descriptors",
            f"{len(descs2):,} descriptors",
            f"{len(ssd_matches):,} matches",
            f"{len(ncc_matches):,} matches",
            "—",
        ],
        "Time (ms)": [
            f"{t_feat1 * 1000:.2f}",
            f"{t_feat2 * 1000:.2f}",
            f"{t_ssd   * 1000:.2f}",
            f"{t_ncc   * 1000:.2f}",
            f"{(t_feat1 + t_feat2 + t_ssd + t_ncc) * 1000:.2f}",
        ],
    })

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "All algorithms implemented from scratch using **NumPy** only.  "
    "No OpenCV, scikit-image, or scipy used for any core computation."
)
