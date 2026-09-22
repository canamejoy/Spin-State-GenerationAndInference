"""
Tests for the topological descriptors in ``notebooks/utils/metrics.py``:

  - the Euler characteristic curve (ECC) and its threshold-free summary
  - stripe defect density (junctions / terminals of the domain skeleton)

Synthetic 39x39 fields are built locally so every assertion is checked
against a geometry whose topology is known by construction.
"""
import numpy as np
import pytest

import metrics

SIZE = metrics.IMG_SIZE  # 39
CY = CX = (SIZE - 1) / 2.0  # 19.0, matches metrics.circular_mask


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic field builders
# ─────────────────────────────────────────────────────────────────────────────
def _hex_centers(k=7, ring_r=9.0, cy=CY, cx=CX):
    """One centre point plus (k - 1) points on a ring of radius ``ring_r``."""
    centers = [(cy, cx)]
    for i in range(k - 1):
        theta = 2.0 * np.pi * i / (k - 1)
        centers.append((cy + ring_r * np.sin(theta), cx + ring_r * np.cos(theta)))
    return centers


def _planted_discs(centers, radius=3.0, value_in=1.0, value_out=-1.0, size=SIZE):
    """``value_out`` background with filled discs of ``value_in`` at ``centers``."""
    y, x = np.mgrid[:size, :size]
    field = np.full((size, size), value_out, dtype=float)
    for cy, cx in centers:
        disc = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
        field[disc] = value_in
    return field


def make_bubbles(k=7):
    """
    Seven disjoint +1 discs (radius 3) on a -1 background, well inside the
    mask and spaced so no two discs touch even under 8-connectivity.

    Superlevel sets are exactly the 7 discs at every threshold, so this is
    the field used to exercise b0_max (peak domain count).
    """
    return _planted_discs(_hex_centers(k=k), radius=3.0, value_in=1.0, value_out=-1.0)


def make_bubble_holes(k=7):
    """
    The photographic negative of ``make_bubbles``: a +1 background with
    seven -1 holes. The superlevel set is now one connected region with 7
    holes cut out of it (chi = b0 - b1 = 1 - 7 = -6), matching the documented
    "-6 for seven bubbles" convention in docs/07_metrics.md section 4.5,
    where the bubble/minority-domain phase is read off the majority phase's
    excursion set.
    """
    return -make_bubbles(k=k)


def make_annulus(inner_r=5.0, outer_r=10.0):
    """A +1 ring (donut) on a -1 background: one component, one hole."""
    y, x = np.mgrid[:SIZE, :SIZE]
    dist2 = (y - CY) ** 2 + (x - CX) ** 2
    field = np.full((SIZE, SIZE), -1.0)
    ring = (dist2 <= outer_r ** 2) & (dist2 > inner_r ** 2)
    field[ring] = 1.0
    return field


def make_stripes(period=6):
    """Horizontal sign stripes of the given period, as a float field."""
    y, _x = np.mgrid[:SIZE, :SIZE]
    return np.sign(np.sin(2.0 * np.pi * y / period)).astype(float)


def make_full_width_band(row_lo=18, row_hi=21):
    """A +1 band spanning every column: both ends are cut by the disk edge."""
    field = np.full((SIZE, SIZE), -1.0)
    field[row_lo:row_hi, :] = 1.0
    return field


def make_left_edge_band(row_lo=18, row_hi=21, col_hi=21):
    """A +1 band from the left edge that stops well inside the disk."""
    field = np.full((SIZE, SIZE), -1.0)
    field[row_lo:row_hi, 0:col_hi] = 1.0
    return field


def make_t_junction():
    """A T-shaped +1 band, entirely inside the disk interior (margin-safe)."""
    field = np.full((SIZE, SIZE), -1.0)
    field[8:20, 18:21] = 1.0   # vertical arm
    field[18:21, 10:29] = 1.0  # horizontal arm
    return field


def make_canvas40():
    """A 40x40 array, which every new public function must reject."""
    return np.zeros((40, 40), dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# ECC: shape / contract
# ─────────────────────────────────────────────────────────────────────────────
def test_ecc_shapes():
    field = make_bubbles()
    curve = metrics.euler_characteristic_curve(field)
    n = len(metrics.THRESHOLD_GRID)
    assert len(curve["u"]) == n
    assert len(curve["chi"]) == n
    assert len(curve["b0"]) == n
    assert len(curve["b1"]) == n
    np.testing.assert_array_equal(curve["b1"], curve["b0"] - curve["chi"])


def test_ecc_rejects_canvas():
    with pytest.raises(ValueError):
        metrics.euler_characteristic_curve(make_canvas40())


def test_ecc_saturated_disk():
    field = np.full((SIZE, SIZE), 1.0)
    curve = metrics.euler_characteristic_curve(field)
    below_saturation = curve["u"] < 1.0
    assert np.all(curve["chi"][below_saturation] == 1)
    assert np.all(curve["b0"][below_saturation] == 1)


def test_ecc_empty_at_top():
    field = np.full((SIZE, SIZE), 0.5)
    curve = metrics.euler_characteristic_curve(field)
    at_or_above = curve["u"] >= 0.5
    assert at_or_above.any()
    assert np.all(curve["chi"][at_or_above] == 0)
    assert np.all(curve["b0"][at_or_above] == 0)


def test_ecc_counts_planted_bubbles():
    field = make_bubbles(k=7)
    summary = metrics.ecc_summary(field)
    assert summary["b0_max"] == 7


def test_ecc_annulus_has_a_hole():
    field = make_annulus(inner_r=5.0, outer_r=10.0)
    curve = metrics.euler_characteristic_curve(field)
    idx = list(np.round(curve["u"], 2)).index(0.0)
    assert curve["b0"][idx] == 1
    assert curve["b1"][idx] == 1
    assert curve["chi"][idx] == 0


def test_ecc_sign_separates_phases():
    bubbles = make_bubble_holes(k=7)
    stripes = make_stripes(period=6)
    bubble_summary = metrics.ecc_summary(bubbles)
    stripe_summary = metrics.ecc_summary(stripes)
    assert bubble_summary["chi_min"] < 0
    assert stripe_summary["chi_max"] > 0


def test_ecc_summary_survives_nyquist_perturbation():
    field = make_bubbles(k=7)
    y, x = np.mgrid[:SIZE, :SIZE]
    perturbed = field + 0.01 * (-1) ** (x + y)

    b0_max_before = metrics.ecc_summary(field)["b0_max"]
    b0_max_after = metrics.ecc_summary(perturbed)["b0_max"]
    assert b0_max_before == b0_max_after

    m_before = metrics.magnetization(field)
    m_after = metrics.magnetization(perturbed)
    assert m_before != m_after


# ─────────────────────────────────────────────────────────────────────────────
# Stripe defects
# ─────────────────────────────────────────────────────────────────────────────
def test_defects_reject_canvas():
    with pytest.raises(ValueError):
        metrics.stripe_defects(make_canvas40())


def test_defects_straight_stripes_have_no_junctions():
    field = make_stripes(period=6)
    result = metrics.stripe_defects(field)
    assert result["n_junctions"] == 0


def test_defects_edge_cut_stripe_has_no_terminals():
    field = make_full_width_band()
    result = metrics.stripe_defects(field)
    assert result["n_terminals"] == 0


def test_defects_interior_terminal_is_counted():
    field = make_left_edge_band()
    result = metrics.stripe_defects(field)
    assert result["n_terminals"] >= 1


def test_defects_y_junction_is_counted():
    field = make_t_junction()
    result = metrics.stripe_defects(field)
    assert result["n_junctions"] >= 1


def test_defects_density_is_bounded():
    fields = [
        make_stripes(period=6),
        make_full_width_band(),
        make_left_edge_band(),
        make_t_junction(),
    ]
    for field in fields:
        result = metrics.stripe_defects(field)
        assert 0.0 <= result["defect_density"] <= 1.0
