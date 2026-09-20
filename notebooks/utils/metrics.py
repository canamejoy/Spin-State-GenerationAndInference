"""
metrics.py — Shared metric functions for generative evaluation.

All image functions assume a single-channel magnetisation map ``s_z`` in the
range ``[-1, 1]`` with shape ``(39, 39)`` — i.e. the *physical* image size,
already cropped from the 40x40 DDPM canvas.

Physical-metric contract
------------------------
The canonical set of physical observables is exactly three:

    magnetization        M        mean s_z over the nanodot disk
    spin_correlation     C_nn     nearest-neighbour s_z correlation
    peak_wave_vector     q_peak   dominant spatial frequency of the texture

Susceptibility (chi), specific heat (Cv) and exchange-energy density (E) were
removed: they require either an ensemble or a single-configuration proxy whose
validity is regime-dependent, which made them unusable as a uniform comparison
axis across models.

Mask / crop ordering (IMPORTANT)
--------------------------------
The DDPM works on a 40x40 canvas obtained by reflect-padding the 39x39 physical
image on its *right and bottom* edges. Therefore:

    1. crop   40x40 -> 39x39 with ``topleft_crop`` (exact, no interpolation)
    2. then   apply ``MASK``

``MASK`` is built at 39x39 and every masked helper asserts the shape, so a
40x40 image can no longer be masked by accident.
"""
import numpy as np
from numpy.fft import fft2, fftshift
from skimage.metrics import structural_similarity as skssim


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 39        # physical image side (pixels)
DDPM_SIZE = 40       # DDPM canvas side (pixels), = IMG_SIZE + 1 reflect pad
RD_PIXELS = 18.25    # nanodot radius R_d = 18.25 MUC (Mendez-Rondon et al. 2026)


def topleft_crop(img, size=IMG_SIZE):
    """
    Crop a (40, 40) DDPM canvas back to the (39, 39) physical image.

    The 39 -> 40 padding is applied to the RIGHT and BOTTOM edges only
    (``F.pad(x, (0, 1, 0, 1), mode='reflect')``), so the top-left crop recovers
    the original pixels exactly. This is NOT a centre crop.
    """
    return img[..., :size, :size]


# Backwards-compatible alias: older notebooks import ``center_crop``.
# The operation was always a top-left crop; only the name was wrong.
center_crop = topleft_crop


def circular_mask(h=IMG_SIZE, w=IMG_SIZE, rd=RD_PIXELS):
    """
    Binary disk mask of the nanodot: True inside radius ``rd`` of the image
    centre ``((h-1)/2, (w-1)/2)``.

    ``rd`` defaults to the physical nanodot radius R_d = 18.25 magnetic unit
    cells, NOT the inscribed circle of the pixel grid (19 px). The inscribed
    circle admits 80 background pixels that dilute every disk average.

    On the 39x39 grid, rd = 18.25 and rd = 18.3 discretise to the identical
    1049-pixel mask; 18.25 is used because it is the published value.
    """
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    Y, X = np.ogrid[:h, :w]
    return ((Y - cy) ** 2 + (X - cx) ** 2) <= rd ** 2


MASK = circular_mask()
N_MASK = int(MASK.sum())


def _check_physical_shape(img):
    """Reject an image that has not been cropped to the physical size yet."""
    if img.shape[-2:] != (IMG_SIZE, IMG_SIZE):
        raise ValueError(
            f"expected a {IMG_SIZE}x{IMG_SIZE} physical image, got {img.shape[-2:]}. "
            f"Crop the DDPM canvas first: topleft_crop(img)."
        )
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Physical metrics — the canonical set of three
# ─────────────────────────────────────────────────────────────────────────────
def magnetization(img, mask=MASK):
    """Mean s_z over the nanodot disk. Range [-1, 1]."""
    _check_physical_shape(img)
    return float(img[mask].mean())


def spin_correlation(img, mask=MASK):
    """
    Nearest-neighbour spin correlation within the disk:

        C_nn = <s_z(i) s_z(j)>  over row/column neighbour pairs i,j in MASK

    Range [-1, 1]. C_nn -> 1 is ferromagnetic alignment, C_nn -> 0 disordered,
    C_nn < 0 antiferromagnetic / short-period modulation.
    """
    _check_physical_shape(img)
    total, count = 0.0, 0
    for dy, dx in [(0, 1), (1, 0)]:
        a = img[:-dy or None, :-dx or None]
        b = img[dy:, dx:]
        valid = mask[:-dy or None, :-dx or None] & mask[dy:, dx:]
        total += float((a * b)[valid].sum())
        count += int(valid.sum())
    return total / count if count > 0 else 0.0


# Backwards-compatible alias: older notebooks import ``cnn_correlation``.
cnn_correlation = spin_correlation


def structure_factor(img, mask=None, subtract_mean=False):
    """
    2D structure factor S(q) = |FFT(field)|^2 / N, zero-frequency centred.

    Defaults reproduce the historical (raw-image) behaviour used by the FFT
    *image* metrics. For *physical* use pass ``mask=MASK, subtract_mean=True``
    so that S(q) describes spin fluctuations on the disk rather than the disk
    aperture itself — this is what ``peak_wave_vector`` does.
    """
    field = np.asarray(img, dtype=np.float64)
    if subtract_mean:
        ref = field[mask] if mask is not None else field
        field = field - ref.mean()
    if mask is not None:
        field = field * mask
    return np.abs(fftshift(fft2(field))) ** 2 / field.size


def azimuthal_average(sq_2d, n_bins=None):
    """Azimuthally average S(q) into integer radial bins. Returns (r_bins, sq_avg)."""
    h, w = sq_2d.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    r_bins, sq_avg = [], []
    for r in range(1, max_r + 1):
        ring = sq_2d[R == r]
        if len(ring) > 0:
            r_bins.append(float(r))
            sq_avg.append(float(ring.mean()))
    return np.array(r_bins), np.array(sq_avg)


def peak_wave_vector(img, mask=MASK, normalize=False):
    """
    Dominant spatial frequency of the magnetic texture.

    S(q) of the *fluctuation* field (disk mean removed, background zeroed) is
    azimuthally averaged; q_peak is the radial bin carrying the most power,
    excluding the q = 0 bin.

    Returns the wavevector in rad per lattice site::

        q_peak = 2*pi * r_peak / IMG_SIZE

    so a helical texture of wavelength ``L`` sites peaks at ``q = 2*pi / L``.
    With ``normalize=True`` the raw radial bin is returned scaled to [0, 1]
    instead (``r_peak / max_r``) — the form used by the differentiable proxy.

    Radial bins are integer, so q_peak is quantised in steps of
    ``2*pi / IMG_SIZE`` ~ 0.161 rad/site; a texture of wavelength 6 sites
    (q = 1.047) is reported at the r = 6 bin, q = 0.967.

    Returns ``nan`` for a field with no spectral power (e.g. a saturated,
    perfectly uniform image).
    """
    _check_physical_shape(img)
    sq = structure_factor(img, mask=mask, subtract_mean=True)
    r_bins, sq_avg = azimuthal_average(sq)
    if len(sq_avg) == 0 or not np.isfinite(sq_avg).any() or sq_avg.max() <= 0:
        return np.nan
    r_peak = float(r_bins[int(np.argmax(sq_avg))])
    if normalize:
        return r_peak / float(r_bins[-1])
    return 2.0 * np.pi * r_peak / float(IMG_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# Topology of the s_z level sets
# ─────────────────────────────────────────────────────────────────────────────
def topological_descriptors(img, mask=MASK, threshold=0.25):
    """
    Counts and Euler characteristic of the s_z level sets inside the disk.

    The true skyrmion number, Q = (1/4pi) * integral n . (d_x n x d_y n), needs the full
    three-component spin field. This dataset stores only the s_z projection, so Q is not
    computable from it and nothing here should be called a skyrmion number. What IS
    computable — and is what separates a skyrmion lattice from a helical stripe in the
    s_z projection — is the topology of the level sets: how many separate domains of each
    sign, and whether those domains enclose holes.

    Why this matters next to the scalar observables: these numbers are DISCRETE. The
    perturbation that latent guidance writes is about 1% of the image range and lives near
    the Nyquist frequency; it shifts every scalar average a little and changes a domain
    count not at all. A generator that reproduces M, C_nn and q_peak while producing the
    wrong number of domains is matching the statistics with the wrong structures, and only
    a discrete descriptor will say so.

    ``threshold`` is deliberately well away from zero. At threshold 0 the level set of a
    disordered configuration fragments into hundreds of single-pixel specks and the count
    measures noise; 0.25 keeps only domains with real amplitude.

    Returns a dict with:
      n_pos, n_neg   number of connected domains above +threshold and below -threshold
      euler_pos      Euler characteristic of the positive set (components minus holes)
      wall_frac      fraction of in-disk neighbour pairs that straddle zero, i.e. the
                     domain-wall density
      largest_frac   area of the largest domain over the disk area
    """
    from scipy import ndimage
    from skimage.measure import euler_number

    _check_physical_shape(img)
    a = np.asarray(img, dtype=np.float64)

    pos = (a > threshold) & mask
    neg = (a < -threshold) & mask
    # 8-connectivity: two domains touching only at a corner are one domain, which is what
    # a physical texture does.
    conn = np.ones((3, 3), dtype=bool)
    n_pos = int(ndimage.label(pos, structure=conn)[1])
    n_neg = int(ndimage.label(neg, structure=conn)[1])

    lab, _ = ndimage.label(pos, structure=conn)
    largest = int(np.bincount(lab.ravel())[1:].max()) if n_pos else 0

    # Domain walls: neighbour pairs inside the disk whose product is negative.
    walls = 0
    total = 0
    for dy, dx in ((0, 1), (1, 0)):
        u = a[:-dy or None, :-dx or None]
        v = a[dy:, dx:]
        ok = mask[:-dy or None, :-dx or None] & mask[dy:, dx:]
        walls += int(((u * v) < 0)[ok].sum())
        total += int(ok.sum())

    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "euler_pos": int(euler_number(pos, connectivity=2)),
        "wall_frac": walls / total if total else 0.0,
        "largest_frac": largest / int(mask.sum()),
    }


TOPOLOGY_NAMES = ["n_pos", "n_neg", "euler_pos", "wall_frac", "largest_frac"]


def topological_batch(imgs, mask=MASK, threshold=0.25):
    """Evaluate the topological descriptors over ``(B, 39, 39)`` -> dict of (B,) arrays."""
    rows = [topological_descriptors(im, mask=mask, threshold=threshold) for im in imgs]
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in TOPOLOGY_NAMES}


PHYSICAL_METRICS = {
    "magnetization":    magnetization,
    "spin_correlation": spin_correlation,
    "peak_wave_vector": peak_wave_vector,
}

PHYSICAL_METRIC_NAMES = list(PHYSICAL_METRICS)

PHYSICAL_METRIC_LABELS = {
    "magnetization":    r"$M$",
    "spin_correlation": r"$C_{nn}$",
    "peak_wave_vector": r"$q_{\mathrm{peak}}$",
}


def physical_metrics(img, mask=MASK):
    """Evaluate the full canonical physical set on one 39x39 image -> dict."""
    return {name: fn(img, mask=mask) for name, fn in PHYSICAL_METRICS.items()}


def physical_metrics_batch(imgs, mask=MASK):
    """Evaluate the canonical set over ``(B, 39, 39)`` -> dict of (B,) arrays."""
    rows = [physical_metrics(img, mask=mask) for img in imgs]
    return {name: np.array([r[name] for r in rows], dtype=np.float64)
            for name in PHYSICAL_METRIC_NAMES}


# ─────────────────────────────────────────────────────────────────────────────
# Image metrics
# ─────────────────────────────────────────────────────────────────────────────
def masked_mse(a, b, mask=MASK):
    _check_physical_shape(a)
    return float(((a - b) ** 2)[mask].mean())


def masked_bce(a, b, mask=MASK, eps=1e-7):
    _check_physical_shape(a)
    a_ = (a[mask] + 1) / 2
    b_ = (b[mask] + 1) / 2
    return float(-np.mean(a_ * np.log(b_ + eps) + (1 - a_) * np.log(1 - b_ + eps)))


def masked_ssim(a, b):
    return float(skssim(a, b, data_range=2.0))


def cosine_similarity_pair(z1, z2):
    n1 = np.linalg.norm(z1) + 1e-8
    n2 = np.linalg.norm(z2) + 1e-8
    return float(np.dot(z1 / n1, z2 / n2))


def cosine_similarity_batch(z1, z2):
    """Cosine similarity between paired feature vectors, shape (B, D) -> (B,)."""
    n1 = np.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8
    n2 = np.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8
    return np.sum((z1 / n1) * (z2 / n2), axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Magnetic-phase taxonomy
# ─────────────────────────────────────────────────────────────────────────────
STRUCTURE_MAP = {
    4:  "Helical",
    5:  "Helical",
    13: "Helical",
    6:  "Labyrinthine & Conical",
    14: "Labyrinthine & Conical",
    8:  "Bimeron",
    10: "Ferromagnetic",
    11: "Ferromagnetic",
    12: "Ferromagnetic",
    15: "Skyrmions",
    16: "Skyrmions",
    17: "Field-Saturated",
}

STRUCTURE_NAMES = [
    "Ferromagnetic",
    "Helical",
    "Labyrinthine & Conical",
    "Bimeron",
    "Skyrmions",
    "Field-Saturated",
]

STRUCTURE_COLORS = {
    "Ferromagnetic":          "#1f77b4",
    "Helical":                "#d62728",
    "Labyrinthine & Conical": "#2ca02c",
    "Bimeron":                "#9467bd",
    "Skyrmions":              "#8c564b",
    "Field-Saturated":        "#e377c2",
}

MODEL_COLORS = {
    "DDPM":            "#2563EB",
    "DDPM+cos":        "#7C3AED",
    "DDPM+cos (joint)": "#DB2777",
    "DDPM+guidance":   "#EA580C",
    "CVAE-Xception":   "#16A34A",
    "CVAE-ViT":        "#DC2626",
}

# Column order of ``data['params']`` — verified against README.md, docs/01_hamiltonian.md
# and the notebook that actually trained on the array
# (notebooks/inverse/XceptionFullDataBaseV3100.ipynb).
# A previous revision of this file listed these in a different order, which
# silently mislabelled every per-parameter table that imported PARAM_NAMES.
PARAM_KEYS = ["T", "Jex2", "Jex3", "Jex4", "Kan1", "KanS", "Hex", "KDM"]

PARAM_NAMES = ["T⁰", "J̃₂", "J̃₃", "J̃₄", "K̃an1", "K̃anS", "H̃ex", "K̃DM"]

PARAM_UNITS = ["K", "meV", "meV", "meV", "meV/atom", "meV/atom", "meV/atom", "meV"]

PARAM_INDEX = {k: i for i, k in enumerate(PARAM_KEYS)}


def get_structure_label(cluster_id):
    """Map a numeric cluster ID to its structure name string."""
    return STRUCTURE_MAP.get(int(cluster_id), f"Unknown({cluster_id})")


# ─────────────────────────────────────────────────────────────────────────────
# Robustness helpers
# ─────────────────────────────────────────────────────────────────────────────
def shift_image(img, px, axis=1):
    """Shift image by px pixels along axis using np.roll."""
    return np.roll(img, px, axis=axis)


def reflect_image(img):
    """Horizontal flip (left-right reflection)."""
    return img[:, ::-1]


def normalize_metrics(scores_dict, reference_key="E0", worst_key="E2"):
    """
    Normalize raw metric scores for interpretability.

    Similarity metrics (ssim, cosine): divided by mean of reference condition (E0).
        Normalized E0 = 1.0. Degraded conditions < 1.0.

    Error metrics (mse, bce): divided by mean of worst condition (E2).
        Normalized E2 = 1.0. Reference E0 ~ 0.0.

    Returns ``(norm_dict, denominators)``.
    """
    sim_metrics   = ["ssim", "cosine"]
    error_metrics = ["mse", "bce"]

    denom_sim   = {m: float(np.mean(scores_dict[reference_key][m])) for m in sim_metrics}
    denom_error = {m: float(np.mean(scores_dict[worst_key][m]))     for m in error_metrics}
    denominators = {**denom_sim, **denom_error}

    norm_dict = {}
    for cond, metric_scores in scores_dict.items():
        norm_dict[cond] = {}
        for m, vals in metric_scores.items():
            vals = np.asarray(vals)
            denom = denom_sim[m] if m in sim_metrics else denom_error[m]
            norm_dict[cond][m] = vals / denom if denom > 1e-9 else vals

    return norm_dict, denominators


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────
def save_figure(fig, path_no_ext, dpi=300):
    """Save figure in both PNG (300 dpi) and SVG formats."""
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.svg", bbox_inches="tight")


def apply_figure_style():
    """Global publication figure style — call once at the top of each notebook."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })
