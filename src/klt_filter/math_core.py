"""Pure-math helpers for the KL filter (instrument-agnostic)."""

import numpy as np
import logging
import scipy.linalg as la
from typing import Optional

logger = logging.getLogger(__name__)


def form_S_from_phase(fringestop_phase: np.ndarray) -> np.ndarray:
    """Signal covariance matrix from fringestop phases.

    Parameters
    ----------
    fringestop_phase : (ninputs,) complex array

    Returns
    -------
    S : (ninputs, ninputs) complex array
    """
    signal_phase = fringestop_phase.conj()
    return np.conj(signal_phase[np.newaxis, :]) * signal_phase[:, np.newaxis]


def form_F_from_vis(
    data_for_vis: np.ndarray,
    frame_start: int = 0,
    frame_stop: Optional[int] = None,
) -> np.ndarray:
    """Foreground/noise covariance from visibility data.

    Parameters
    ----------
    data_for_vis : (ninputs, ntime) complex array
    frame_start, frame_stop : int, optional
        Time samples to *exclude* (signal region) when estimating covariance.

    Returns
    -------
    F : (ninputs, ninputs) complex array
    """
    if frame_stop is not None:
        logger.info("using %d as signal frame stop", frame_stop)
        data_for_vis = np.concatenate(
            (data_for_vis[..., :frame_start], data_for_vis[..., frame_stop:]),
            axis=-1,
        )
    else:
        data_for_vis = data_for_vis[..., :]

    nn = data_for_vis.shape[-1]
    return data_for_vis @ data_for_vis.conj().T / nn


def form_M_from_V(signal_phase: np.ndarray) -> np.ndarray:
    """Outer-product visibility matrix from per-input data.

    Parameters
    ----------
    signal_phase : (ninputs, ...) complex array

    Returns
    -------
    M : (ninputs, ninputs, ...) complex array
    """
    return np.conj(signal_phase[np.newaxis, :, ...]) * signal_phase[:, np.newaxis, ...]


def KL_filter(
    S: np.ndarray,
    F: np.ndarray,
    data_to_clean: np.ndarray,
):
    """Apply the KL (Karhunen-Loève) filter.

    Solves the generalized eigenvalue problem S v = lambda F v and projects
    the data onto the eigenvector with the largest eigenvalue.

    Parameters
    ----------
    S : (N, N) signal covariance
    F : (N, N) foreground/noise covariance
    data_to_clean : (N, ntime) complex baseband data

    Returns
    -------
    cleaned_data : (N, ntime) complex array
    b : (N,) complex — the projection weights (last row of R†)
    """
    evalues, R = la.eigh(S, F)
    R_dagger = R.T.conj()
    R_dagger_inv = la.inv(R_dagger)
    a = R_dagger_inv[:, -1]
    b = R_dagger[-1]
    cleaned = a[:, np.newaxis] * np.sum(
        b[:, np.newaxis] * data_to_clean, axis=0
    )[np.newaxis, :]
    return cleaned, b


def apply_pol_covariances(
    S: np.ndarray,
    N_x: int,
    Ex_Ex: float,
    Ex_Ey: complex,
    Ey_Ex: complex,
    Ey_Ey: float,
) -> np.ndarray:
    """Scale signal covariance blocks by polarization cross-powers."""
    rows, cols = S.shape
    upper_left = np.logical_and(
        np.arange(rows)[:, None] < N_x, np.arange(cols)[None, :] < N_x
    )
    upper_right = np.logical_and(
        np.arange(rows)[:, None] < N_x, np.arange(cols)[None, :] >= N_x
    )
    lower_left = np.logical_and(
        np.arange(rows)[:, None] >= N_x, np.arange(cols)[None, :] < N_x
    )
    lower_right = np.logical_and(
        np.arange(rows)[:, None] >= N_x, np.arange(cols)[None, :] >= N_x
    )
    S[upper_left] *= Ex_Ex
    S[upper_right] *= Ex_Ey
    S[lower_left] *= Ey_Ex
    S[lower_right] *= Ey_Ey
    return S
