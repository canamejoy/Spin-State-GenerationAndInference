"""Tests for the orientational-order and correlation-range observables.

The point of these two observables is to separate textures that the three
canonical ones cannot tell apart, so most of these tests build a pair of
patterns that share a characteristic wavelength and assert the new observable
splits them.
"""
import numpy as np
import pytest

import metrics as M


def _disk(img):
    """Zero everything outside the nanodot disk, as the metrics do."""
    return np.where(M.MASK, img, 0.0)


def _stripes(period, angle=0.0, size=M.IMG_SIZE):
    """A uniaxial sinusoid of the given period, rotated by ``angle``."""
    y, x = np.mgrid[:size, :size]
    proj = x * np.cos(angle) + y * np.sin(angle)
    return _disk(np.sin(2 * np.pi * proj / period))


def _hexagonal(period, size=M.IMG_SIZE):
    """Three sinusoids at 60 degrees — a triangular lattice."""
    y, x = np.mgrid[:size, :size]
    out = np.zeros((size, size), dtype=float)
    for angle in (0.0, np.pi / 3, 2 * np.pi / 3):
        proj = x * np.cos(angle) + y * np.sin(angle)
        out += np.sin(2 * np.pi * proj / period)
    return _disk(out / 3.0)


# ── shape and contract ───────────────────────────────────────────────────────
def test_orientation_rejects_canvas():
    with pytest.raises(ValueError):
        M.orientational_order(np.zeros((M.DDPM_SIZE, M.DDPM_SIZE)))


def test_correlation_length_rejects_canvas():
    with pytest.raises(ValueError):
        M.correlation_length(np.zeros((M.DDPM_SIZE, M.DDPM_SIZE)))


def test_orientation_keys_and_ranges():
    out = M.orientational_order(_stripes(6.0))
    assert set(out) == set(M.ORIENTATION_NAMES)
    for k in ("psi2", "psi6", "aniso"):
        assert 0.0 <= out[k] <= 1.0 + 1e-9, (k, out[k])
    assert 0.0 <= out["theta2"] < np.pi


def test_orientation_batch_shape():
    imgs = np.stack([_stripes(6.0), _hexagonal(6.0)])
    assert M.orientational_batch(imgs).shape == (2, len(M.ORIENTATION_NAMES))


def test_flat_field_is_not_oriented():
    out = M.orientational_order(_disk(np.ones((M.IMG_SIZE, M.IMG_SIZE))))
    assert out["psi2"] < 0.5
    assert out["psi6"] < 0.5


# ── the discrimination these observables exist for ───────────────────────────
def test_stripes_are_uniaxial_and_hexagonal_is_not():
    """psi2 separates a single-axis pattern from a three-axis one."""
    assert M.orientational_order(_stripes(6.0))["psi2"] > \
           M.orientational_order(_hexagonal(6.0))["psi2"]


def test_psi6_alone_does_not_identify_a_hexagonal_lattice():
    """
    Angular power concentrated on one axis gives |psi_n| ~ 1 for every n, so a
    stripe pattern scores as high on psi6 as a triangular lattice does. This
    test pins that down so nobody reads psi6 on its own as a lattice detector.
    """
    assert M.orientational_order(_stripes(6.0))["psi6"] > 0.8
    assert M.orientational_order(_hexagonal(6.0))["psi6"] > 0.8


def test_the_pair_psi2_psi6_separates_the_two_lattices():
    """The discriminator is six-fold order IN EXCESS of two-fold order."""
    hexa = M.orientational_order(_hexagonal(6.0))
    stripe = M.orientational_order(_stripes(6.0))
    assert hexa["psi6"] - hexa["psi2"] > 0.5
    assert stripe["psi6"] - stripe["psi2"] < 0.2


def test_stripes_and_hexagonal_share_a_peak_wavevector():
    """The premise of the whole addition: q_peak cannot tell these apart."""
    q_s = M.peak_wave_vector(_stripes(6.0))
    q_h = M.peak_wave_vector(_hexagonal(6.0))
    assert abs(q_s - q_h) <= 1.0, (q_s, q_h)


def test_isotropic_noise_is_less_anisotropic_than_stripes():
    rng = np.random.default_rng(0)
    noise = _disk(rng.normal(size=(M.IMG_SIZE, M.IMG_SIZE)))
    assert M.orientational_order(noise)["aniso"] < \
           M.orientational_order(_stripes(6.0))["aniso"]


def test_theta2_tracks_the_stripe_axis():
    """Rotating the pattern rotates the reported axis by the same amount."""
    a = M.orientational_order(_stripes(6.0, angle=0.0))["theta2"]
    b = M.orientational_order(_stripes(6.0, angle=np.pi / 2))["theta2"]
    delta = abs(b - a) % np.pi
    assert min(delta, np.pi - delta) > 0.6


def test_psi2_is_invariant_under_rotation_of_the_pattern():
    """The magnitude is an order parameter; only the phase should move."""
    a = M.orientational_order(_stripes(6.0, angle=0.0))["psi2"]
    b = M.orientational_order(_stripes(6.0, angle=np.pi / 4))["psi2"]
    assert abs(a - b) < 0.25, (a, b)


# ── correlation length ───────────────────────────────────────────────────────
def test_correlation_length_is_positive_and_bounded():
    xi = M.correlation_length(_stripes(6.0))
    assert 0.0 < xi <= M.IMG_SIZE


def test_long_wavelength_stays_correlated_longer():
    assert M.correlation_length(_stripes(14.0)) > M.correlation_length(_stripes(4.0))


def test_noise_decorrelates_faster_than_structure():
    rng = np.random.default_rng(1)
    noise = _disk(rng.normal(size=(M.IMG_SIZE, M.IMG_SIZE)))
    assert M.correlation_length(noise) < M.correlation_length(_stripes(8.0))


def test_correlation_length_survives_a_nyquist_perturbation():
    """
    The documented attack on this project's cycle metric was a ~1% amplitude
    perturbation near Nyquist. A physical observable must be blind to it —
    that immunity is the reason it is trusted as an auditor.
    """
    base = _stripes(6.0)
    y, x = np.mgrid[:M.IMG_SIZE, :M.IMG_SIZE]
    checker = _disk(0.01 * (-1.0) ** (x + y))
    assert abs(M.correlation_length(base + checker) -
               M.correlation_length(base)) < 0.5


def test_orientational_order_survives_a_nyquist_perturbation():
    base = _stripes(6.0)
    y, x = np.mgrid[:M.IMG_SIZE, :M.IMG_SIZE]
    checker = _disk(0.01 * (-1.0) ** (x + y))
    assert abs(M.orientational_order(base + checker)["psi2"] -
               M.orientational_order(base)["psi2"]) < 0.05
