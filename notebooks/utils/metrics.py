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
# Orientational order and correlation range
#
# The three canonical observables answer "how much" (M), "how locally aligned"
# (C_nn) and "at what scale" (q_peak). They do not answer "with what symmetry"
# or "over what range". A helical stripe phase, a skyrmion lattice and a
# labyrinth can share a q_peak and be told apart only by the angular content of
# S(q), which ``azimuthal_average`` discards.
# ─────────────────────────────────────────────────────────────────────────────
ORIENTATION_NAMES = ["psi2", "psi6", "aniso", "theta2"]


def _peak_annulus(sq_2d, rel_width=0.25):
    """
    Radial coordinates of ``sq_2d`` and a boolean annulus around its peak ring.

    The peak ring is located on the azimuthally averaged profile, so a strongly
    uniaxial pattern (two Bragg spots) and an isotropic one (a full ring) with
    the same characteristic wavelength select the same annulus.
    """
    h, w = sq_2d.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

    r_bins, sq_avg = azimuthal_average(sq_2d)
    # azimuthal_average starts at r = 1, so there is no DC bin to skip.
    if sq_avg.size == 0 or sq_avg.max() <= 0:
        return r, np.zeros(sq_2d.shape, dtype=bool)
    r_peak = float(r_bins[int(np.argmax(sq_avg))])
    half = max(1.0, rel_width * r_peak)
    return r, (np.abs(r - r_peak) <= half) & (r > 0)


def orientational_order(img, mask=MASK, n_sectors=36, rel_width=0.25):
    """
    Bond-orientational order parameters of the dominant spatial mode.

    ``S(q)`` is restricted to an annulus around its peak ring and its angular
    power ``P(phi)`` is accumulated over ``n_sectors`` bins spanning ``[0, pi)``
    — the half circle, because ``S(q)`` of a real field is centrosymmetric and
    ``phi`` and ``phi + pi`` are the same axis.

    Returns a dict with:

    ``psi2``   |<e^{2 i phi}>| — uniaxial order. Near 1 for helical stripes and
               for any single-axis pattern; near 0 for an isotropic ring.
    ``psi6``   |<e^{6 i phi}>| — six-fold order. **Read it together with**
               ``psi2``, never alone: angular power concentrated on a single
               axis gives |psi_n| ~ 1 for EVERY n, so a stripe pattern scores
               high on psi6 too. A triangular lattice is the case where psi6
               is high AND psi2 collapses (measured: stripes 0.98/0.91,
               hexagonal 0.00/0.89).
    ``aniso``  Angular concentration ``1 - H/H_max`` of ``P(phi)``, where ``H``
               is its Shannon entropy. Model-free: it does not assume any
               symmetry. A max/min contrast was tried first and rejected --
               it saturates at 1.0 for both clean stripes and isotropic
               noise, because some angular sector is always near-empty.
    ``theta2`` arg(<e^{2 i phi}>) / 2, in radians on ``[0, pi)`` — the
               orientation of the uniaxial axis. Meaningless when ``psi2`` is
               small, and reported regardless so the caller can gate on it.

    These are ORIENTATIONAL ORDER PARAMETERS, not topological charges. The
    topological charge of a spin texture needs the full three-component field;
    the ``s_z`` projection cannot supply it (see section 4.1 of docs/07_metrics.md).
    What links them to topology is KTHNY theory, where the orientational
    correlation length is set by the density of disclinations.
    """
    _check_physical_shape(img)
    sq = structure_factor(img, mask=mask, subtract_mean=True)
    r, ring = _peak_annulus(sq, rel_width=rel_width)

    h, w = sq.shape
    cy, cx = h // 2, w // 2
    Y, X = np.mgrid[:h, :w]
    phi = np.arctan2(Y - cy, X - cx) % np.pi          # half circle

    weights = sq[ring]
    angles = phi[ring]
    # S(q) is a power spectrum and is non-negative up to rounding; clamp so a
    # tiny negative value cannot flip the sign of a sector's weight.
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return {"psi2": 0.0, "psi6": 0.0, "aniso": 0.0, "theta2": 0.0}

    z2 = np.sum(weights * np.exp(2j * angles)) / total
    z6 = np.sum(weights * np.exp(6j * angles)) / total

    idx = np.minimum((angles / np.pi * n_sectors).astype(int), n_sectors - 1)
    power = np.bincount(idx, weights=weights, minlength=n_sectors)
    counts = np.bincount(idx, minlength=n_sectors)
    occupied = counts > 0
    if occupied.sum() < 2:
        aniso = 0.0
    else:
        dens = power[occupied] / counts[occupied]
        tot = dens.sum()
        if tot <= 0:
            aniso = 0.0
        else:
            pk = dens / tot
            nz = pk[pk > 0]
            entropy = -np.sum(nz * np.log(nz))
            aniso = float(1.0 - entropy / np.log(occupied.sum()))

    # theta2 lives on the half-open interval [0, pi). A bare modulo is not
    # enough: np.angle returns a tiny negative value for a real-positive z2,
    # and (-1e-16) % (2*pi) rounds up to exactly 2*pi, putting theta2 on pi.
    theta2 = (np.angle(z2) % (2.0 * np.pi)) / 2.0
    if not (theta2 < np.pi):
        theta2 = 0.0

    return {
        "psi2": float(np.abs(z2)),
        "psi6": float(np.abs(z6)),
        "aniso": float(aniso),
        "theta2": float(theta2),
    }


def orientational_batch(imgs, mask=MASK, n_sectors=36, rel_width=0.25):
    """``orientational_order`` over a stack, as an (N, 4) array."""
    out = [orientational_order(im, mask=mask, n_sectors=n_sectors,
                               rel_width=rel_width) for im in imgs]
    return np.array([[d[k] for k in ORIENTATION_NAMES] for d in out])


def correlation_length(img, mask=MASK, threshold=1.0 / np.e):
    """
    Range over which the ``s_z`` texture stays correlated, in pixels.

    The masked, mean-subtracted field is autocorrelated through the
    Wiener-Khinchin theorem, then divided by the autocorrelation of the mask
    itself. That division is what makes the estimate honest: without it the
    finite disk aperture alone would make ``C(r)`` decay even for a perfectly
    uniform texture.

    ``xi`` is the first radius where the azimuthally averaged ``C(r)`` crosses
    ``threshold`` (default ``1/e``), linearly interpolated between bins. A
    texture that never decays that far returns the largest measurable radius —
    a censored value, not a failure, so callers comparing distributions should
    know the scale is bounded by the disk.

    ``q_peak`` says WHICH length scale dominates; this says HOW FAR that order
    actually survives. A clean stripe pattern and a glassy one can share a
    ``q_peak`` and differ here.
    """
    _check_physical_shape(img)
    field = np.where(mask, img - img[mask].mean(), 0.0)

    f = np.abs(fft2(field)) ** 2
    ac = np.real(fftshift(np.fft.ifft2(f)))
    m = np.abs(fft2(mask.astype(float))) ** 2
    norm = np.real(fftshift(np.fft.ifft2(m)))

    with np.errstate(divide="ignore", invalid="ignore"):
        c2d = np.where(norm > 1e-9, ac / np.maximum(norm, 1e-9), 0.0)

    h, w = c2d.shape
    c0 = c2d[h // 2, w // 2]
    if not np.isfinite(c0) or c0 <= 0:
        return 0.0

    r_bins, prof = azimuthal_average(c2d)
    if prof.size == 0:
        return 0.0
    # r = 0 is the centre pixel, which azimuthal_average omits; prepend it so
    # the curve starts at C(0) = 1 and the crossing search has a left edge.
    radii = np.concatenate(([0.0], r_bins))
    prof = np.concatenate(([c0], prof)) / c0

    below = np.nonzero(prof < threshold)[0]
    if below.size == 0:
        return float(radii[-1])              # censored at the disk scale
    i = int(below[0])
    if i == 0:
        return 0.0
    hi, lo = prof[i - 1], prof[i]
    if hi == lo:
        return float(radii[i])
    frac = (hi - threshold) / (hi - lo)
    return float(radii[i - 1] + frac * (radii[i] - radii[i - 1]))


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


# Superlevel-set threshold grid for the Euler characteristic curve: 19 values,
# -0.9 .. 0.9 in steps of 0.1. This replaces the single free threshold of
# ``topological_descriptors`` with a full sweep (see docs/07_metrics.md, 4.7).
THRESHOLD_GRID = np.round(np.arange(-0.9, 0.95, 0.1), 2)


def euler_characteristic_curve(img, mask=MASK, thresholds=THRESHOLD_GRID):
    """
    Euler characteristic curve (ECC) of the s_z superlevel-set filtration.

    For every threshold u on ``thresholds`` the superlevel set

        S_u = {i in disk : s_z(i) > u}

    is formed under 8-connectivity, and three curves are evaluated over it:
    the number of connected components b0(u), the Euler characteristic
    chi(u) = euler_number(S_u), and the number of holes b1(u) = b0(u) - chi(u)
    (valid in 2D, where chi = b0 - b1 by the Betti-number decomposition of
    Hadwiger's theorem — see docs/07_metrics.md 4.2).

    Sweeping u removes the single free threshold of ``topological_descriptors``
    (u = 0.25): no one value has to be defended, and every summary below is a
    functional of the whole curve rather than a reading at one point.

    Returns a dict of four 1-D float arrays, all the same length as
    ``thresholds``: ``"u"``, ``"chi"``, ``"b0"``, ``"b1"``.
    """
    from scipy import ndimage
    from skimage.measure import euler_number

    _check_physical_shape(img)
    a = np.asarray(img, dtype=np.float64)
    conn = np.ones((3, 3), dtype=bool)

    u = np.asarray(thresholds, dtype=np.float64)
    b0 = np.empty(len(u), dtype=np.float64)
    chi = np.empty(len(u), dtype=np.float64)
    for i, thr in enumerate(u):
        s_u = (a > thr) & mask
        b0[i] = float(ndimage.label(s_u, structure=conn)[1])
        chi[i] = float(euler_number(s_u, connectivity=2))
    b1 = b0 - chi

    return {"u": u, "chi": chi, "b0": b0, "b1": b1}


ECC_SUMMARY_NAMES = ["chi_min", "chi_max", "u_chi_min", "b0_max", "ecc_l1", "b1_total"]


def ecc_summary(img, mask=MASK, thresholds=THRESHOLD_GRID):
    """
    Threshold-free scalar summaries of the Euler characteristic curve.

    Each summary is a functional of the whole curve, not a reading at one
    threshold:

      chi_min    min of chi(u) over the grid (most negative = most bubble-like,
                 i.e. a majority phase riddled with holes)
      chi_max    max of chi(u) over the grid (most positive = unbranched
                 stripes or a saturated domain)
      u_chi_min  the u at which chi attains chi_min (first occurrence)
      b0_max     max of b0(u) over the grid — the threshold-free replacement
                 for ``n_pos``: the peak domain count over ALL thresholds
                 instead of one fixed one
      ecc_l1     integral of |chi(u)| over u — total topological activity
      b1_total   integral of b1(u) over u — total loop content

    Returns a dict of six plain Python floats, keys given by
    ``ECC_SUMMARY_NAMES``.
    """
    curve = euler_characteristic_curve(img, mask=mask, thresholds=thresholds)
    u, chi, b0, b1 = curve["u"], curve["chi"], curve["b0"], curve["b1"]
    i_chi_min = int(np.argmin(chi))

    return {
        "chi_min": float(chi.min()),
        "chi_max": float(chi.max()),
        "u_chi_min": float(u[i_chi_min]),
        "b0_max": float(b0.max()),
        "ecc_l1": float(np.trapezoid(np.abs(chi), u)),
        "b1_total": float(np.trapezoid(b1, u)),
    }


def ecc_summary_batch(imgs, mask=MASK, thresholds=THRESHOLD_GRID):
    """Evaluate ``ecc_summary`` over ``(B, 39, 39)`` -> dict of (B,) arrays."""
    rows = [ecc_summary(im, mask=mask, thresholds=thresholds) for im in imgs]
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in ECC_SUMMARY_NAMES}


DEFECT_NAMES = ["n_terminals", "n_junctions", "n_isolated", "skeleton_len", "defect_density"]


def stripe_defects(img, mask=MASK, threshold=0.0, margin=2):
    """
    Junction and terminal density of the s_z domain skeleton.

    This is a descriptor of the STRIPE MORPHOLOGY, not the skyrmion number:
    it says nothing about the in-plane winding of the spin field, only about
    how the domain boundary of the s_z projection branches. It answers a
    question the Euler characteristic curve cannot: the ECC counts components
    and holes, but says nothing about whether the domain boundary branches.

    Method, grounded in Okubo, Shimizu, Shivaram & Kim, IEEE Access 12,
    92419-92430 (2024), doi:10.1109/ACCESS.2024.3422259, who detect junctions
    and terminals in magnetic labyrinthine patterns with a TM-CNN because
    their microscopy images are noisy. Our simulated fields are clean, so a
    morphological skeleton is exact and needs no learned detector:

      1. threshold the domain: ``dom = (img > threshold) & mask``
      2. skeletonize ``dom`` to a 1-pixel-wide morphological skeleton
      3. count, per skeleton pixel, its 8-neighbour skeleton neighbours
      4. classify: 1 neighbour = terminal, >= 3 = junction, 0 = isolated
         (a degenerate single-pixel bubble)

    Boundary exclusion (mandatory, and the whole correctness point): a stripe
    that is cut by the edge of the dot produces a spurious terminal at the
    cut. Counts are restricted to ``skel & inner``, where ``inner`` is
    ``mask`` eroded by ``margin`` pixels, so a domain that merely exits the
    disk does not register a false terminal there. This is a correctness
    requirement, not a tuning knob.

    Returns a dict with:
      n_terminals     skeleton pixels (inside ``inner``) with exactly 1
                       neighbour
      n_junctions      skeleton pixels (inside ``inner``) with >= 3 neighbours
      n_isolated       skeleton pixels (inside ``inner``) with 0 neighbours
      skeleton_len     number of skeleton pixels inside ``inner``
      defect_density   (n_terminals + n_junctions) / skeleton_len, or 0.0
                       when skeleton_len == 0
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    _check_physical_shape(img)
    a = np.asarray(img, dtype=np.float64)

    dom = (a > threshold) & mask
    skel = skeletonize(dom)

    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neighbor_count = ndimage.convolve(skel.astype(int), kernel, mode="constant", cval=0)

    inner = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=margin)
    region = skel & inner

    n_terminals = int(np.sum(region & (neighbor_count == 1)))
    n_junctions = int(np.sum(region & (neighbor_count >= 3)))
    n_isolated = int(np.sum(region & (neighbor_count == 0)))
    skeleton_len = int(np.sum(region))
    defect_density = (n_terminals + n_junctions) / skeleton_len if skeleton_len > 0 else 0.0

    return {
        "n_terminals": n_terminals,
        "n_junctions": n_junctions,
        "n_isolated": n_isolated,
        "skeleton_len": skeleton_len,
        "defect_density": float(defect_density),
    }


def stripe_defects_batch(imgs, mask=MASK, threshold=0.0, margin=2):
    """Evaluate ``stripe_defects`` over ``(B, 39, 39)`` -> dict of (B,) arrays."""
    rows = [stripe_defects(im, mask=mask, threshold=threshold, margin=margin) for im in imgs]
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in DEFECT_NAMES}


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
