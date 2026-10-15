import numpy as np
from scipy.linalg import helmert
from scipy.special import sph_harm_y
from scipy.interpolate import RBFInterpolator


def basis(coordinates, modes, constant=False):
    xyz = np.asarray(coordinates, float)
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    theta = np.arccos(np.clip(xyz[:, 2], -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    values = []
    penalty = []
    if constant:
        values.append(np.ones(len(xyz)))
        penalty.append(1.0)
    degree = 1
    while len(values) < modes:
        for order in range(-degree, degree + 1):
            harmonic = sph_harm_y(degree, abs(order), theta, phi)
            value = (
                np.sqrt(2) * harmonic.imag
                if order < 0
                else harmonic.real
                if order == 0
                else np.sqrt(2) * harmonic.real
            )
            values.append(value)
            penalty.append((degree * (degree + 1)) ** 2)
            if len(values) == modes:
                break
        degree += 1
    return np.stack(values, axis=1), np.diag(penalty)


def coordinates(x, xyz, mask, c, variant="gaugescalp"):
    x = np.asarray(x, float)
    mask = np.asarray(mask, bool)
    if mask.sum() < 2:
        raise ValueError("At least two electrodes must survive")
    coords = np.asarray(xyz, float).copy()
    if variant == "wrong_coordinates":
        coords = coords[np.roll(np.arange(len(coords)), 1)]
    phi, regularizer = basis(coords, c["modes"], variant == "constant_mode")
    observed = x[mask]
    spatial = phi[mask]
    if variant in ("raw", "car", "spline"):
        canonical = np.zeros_like(x)
        values = (
            observed if variant == "raw" else observed - observed.mean(0, keepdims=True)
        )
        if variant == "spline":
            canonical = RBFInterpolator(
                coords[mask],
                values,
                kernel="thin_plate_spline",
                smoothing=c["regularization"],
                degree=0,
            )(coords)
        else:
            canonical[mask] = values
        coefficients = np.linalg.lstsq(phi, canonical, rcond=None)[0]
        return coefficients, np.full(c["modes"], c["noise_variance"])
    contrast = helmert(int(mask.sum()), full=False)
    if variant == "random_unweighted":
        rng = np.random.default_rng(c["seed"])
        contrast = rng.normal(size=contrast.shape) @ (
            np.eye(mask.sum()) - np.ones((mask.sum(), mask.sum())) / mask.sum()
        )
    operator = contrast @ spatial
    noise = contrast @ contrast.T
    precision = (
        np.eye(len(contrast))
        if variant == "random_unweighted"
        else np.linalg.inv(noise)
    )
    curvature = operator.T @ precision @ operator + c["regularization"] * regularizer
    inverse = np.linalg.inv(curvature)
    estimate = np.linalg.solve(curvature, operator.T @ precision @ contrast @ observed)
    covariance = np.maximum(np.diag(c["noise_variance"] * inverse), 1e-12)
    if variant == "no_covariance":
        covariance = np.ones_like(covariance)
    return estimate, covariance


def representation(x, xyz, mask, c, variant):
    estimate, covariance = coordinates(x, xyz, mask, c, variant)
    return np.concatenate(
        [estimate, np.repeat(np.log(covariance)[:, None], x.shape[1], axis=1)], axis=0
    ).astype("float32"), estimate
