"""CHIME-specific data loading and preprocessing.

This module contains functions that depend on ``ch_util``,
``baseband_analysis``, and other CHIME-internal packages.  It translates
raw CHIME data into the generic inputs expected by :class:`KLTBeamformer`.
"""

import numpy as np
import h5py
import logging
from typing import List, Optional

from ch_util import ephemeris, tools
from baseband_analysis.analysis.beamform import fringestop_time_vectorized
from baseband_analysis.core.bbdata import BBData
import baseband_analysis.core.calibration as cal
from baseband_analysis.core.dedispersion import delay_across_the_band

logger = logging.getLogger(__name__)


def reorder_inputs(data, correlator_inputs, reference_feed=None):
    """Reorder correlator inputs to match BBData input ordering.

    Parameters
    ----------
    data : BBData
    correlator_inputs : list
        Output of ``tools.get_correlator_inputs()``.
    reference_feed : antenna object, optional
        Appended as the reference feed.  Defaults to a zero-position
        ``ArrayAntenna``.

    Returns
    -------
    reordered_inputs : list
        Correlator inputs reordered to match ``data.input``.
    prod_map : structured ndarray
        Product map with ``input_a`` and ``input_b`` fields.
    """
    if reference_feed is None:
        reference_feed = tools.ArrayAntenna(
            id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0], delay=0
        )

    converted = data.input.astype(
        [("chan_id", "<u2"), ("correlator_input", "U32")]
    )
    reordered = tools.reorder_correlator_inputs(converted, correlator_inputs)

    prod_map = np.empty(
        len(data.input), dtype=[("input_a", "u2"), ("input_b", "u2")]
    )
    prod_map["input_a"] = np.arange(len(data.input))
    reordered.append(reference_feed)
    prod_map["input_b"] = len(data.input)

    return reordered, prod_map


def load_gains(gain_file: str, data) -> np.ndarray:
    """Load and reorder gains to match the BBData input ordering.

    Parameters
    ----------
    gain_file : str
        Path to the HDF5 gain file.
    data : BBData

    Returns
    -------
    gain_reordered : (nfreq, ninput) complex array
    """
    freq_id_list = data.index_map["freq"]["id"]
    gain = h5py.File(gain_file, "r")["gain"][:][freq_id_list]
    return gain[:, data.input["chan_id"]]


def apply_gains(data, gain_file: str, correlator_inputs):
    """Read gains and apply calibration in-place.

    Parameters
    ----------
    data : BBData
    gain_file : str
    correlator_inputs : list
    """
    gains = cal.read_gains(gain_file)
    cal.apply_calibration(data, gains, inputs=correlator_inputs)


def compute_fringestop_phases(
    data,
    freq_index: int,
    ras: np.ndarray,
    decs: np.ndarray,
    reordered_inputs,
    obs,
    static_delays: bool,
    fringestop_delays: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute fringestop phases for one frequency channel.

    Parameters
    ----------
    data : BBData
    freq_index : int
    ras, decs : (npointing,) float arrays
    reordered_inputs : list
    obs : observatory object
    static_delays : bool
    fringestop_delays : (ninputs,) array, optional
        If provided, uses ``exp(2j pi freq * delays)`` directly instead
        of computing geometric delays via ``fringestop_time_vectorized``.

    Returns
    -------
    fringestop_phase : (npointing, ninput) complex array
    """
    if fringestop_delays is not None:
        phase = np.exp(2j * np.pi * data.freq[freq_index] * fringestop_delays)
        if phase.shape[0] != len(ras):
            phase = phase.reshape(1, len(phase))
        return phase

    time = (
        data["time0"]["ctime"][freq_index]
        + data["time0"]["ctime_offset"][freq_index]
    )
    time = time + np.mean(data.index_map["time"]["offset_s"])

    ra_from_src = np.empty(len(ras))
    dec_from_src = np.empty(len(decs))
    for ii in range(len(ras)):
        src = ephemeris.skyfield_star_from_ra_dec(ras[ii], decs[ii])
        ra_from_src[ii], dec_from_src[ii] = ephemeris.object_coords(
            src, time, obs=obs
        )

    prod_map = np.empty(
        len(data.input), dtype=[("input_a", "u2"), ("input_b", "u2")]
    )
    prod_map["input_a"] = np.arange(len(data.input))
    prod_map["input_b"] = len(data.input)

    phase = fringestop_time_vectorized(
        time,
        data.freq[freq_index],
        reordered_inputs,
        ra_from_src,
        dec_from_src,
        prod_map=prod_map,
        obs=obs,
        static_delays=static_delays,
    ).T.astype(np.complex64)  # (npointing, ninput)

    return phase


def get_good_inputs(
    gains_freq: np.ndarray,
    reordered_inputs,
) -> List[np.ndarray]:
    """Identify good (non-zero gain) inputs, split by polarisation.

    Parameters
    ----------
    gains_freq : (ninput,) complex array
        Gains for a single frequency.
    reordered_inputs : list
        Correlator input objects (must have ``.pol`` attribute).

    Returns
    -------
    [x_indices, y_indices] : list of two int arrays
    """
    assert np.nanmax(np.abs(gains_freq.flatten())) > 0, "all gains are zero!"
    good = np.where(np.abs(gains_freq) > 0.0)[0]
    x_idx = np.array([i for i in good if reordered_inputs[i].pol == "S"])
    y_idx = np.array([i for i in good if reordered_inputs[i].pol == "E"])
    return [x_idx, y_idx]


def incoherent_dedisp_raw(
    data,
    DM: float,
    t_ref: float,
    f_ref: float = 400.390625,
    key: str = "baseband",
):
    """Apply incoherent dedispersion to raw (pre-beamformed) data in-place.

    Parameters
    ----------
    data : BBData
    DM : float
        Dispersion measure (pc cm^-3).
    t_ref : float
        Reference time at ``f_ref``.
    f_ref : float
        Reference frequency (MHz).
    key : str
        Dataset key in ``data``.
    """
    if "DM" in data[key].attrs.keys():
        undo_incoherent_dedisp_raw(data, key=key)

    matrix_in = data[key][:]
    freq = data.freq
    dt = data.attrs["delta_time"]

    for i in range(matrix_in.shape[0]):
        desired_shift = delay_across_the_band(DM, f_ref, freq[i])
        existing_shift = t_ref - (
            data["time0"]["ctime"][i] + data["time0"]["ctime_offset"][i]
        )
        bins_shift = np.round((desired_shift - existing_shift) / dt).astype(int)
        matrix_in[i] = np.roll(matrix_in[i], bins_shift, axis=-1)

    data[key].attrs["DM"] = DM
    data[key].attrs["DM_t_ref"] = t_ref
    data[key].attrs["DM_f_ref"] = f_ref


def undo_incoherent_dedisp_raw(data, key: str = "baseband"):
    """Undo a previously applied incoherent dedispersion in-place."""
    if "DM" not in data[key].attrs.keys():
        return

    matrix_in = data[key][:]
    freq = data.freq
    dt = data.attrs["delta_time"]
    DM = data[key].attrs["DM"]
    t_ref = data[key].attrs["DM_t_ref"]
    f_ref = data[key].attrs["DM_f_ref"]

    for i in range(matrix_in.shape[0]):
        desired_shift = delay_across_the_band(DM, f_ref, freq[i])
        existing_shift = t_ref - (
            data["time0"]["ctime"][i] + data["time0"]["ctime_offset"][i]
        )
        bins_shift = -np.round((desired_shift - existing_shift) / dt).astype(int)
        matrix_in[i] = np.roll(matrix_in[i], bins_shift, axis=-1)

    del data[key].attrs["DM"]
    del data[key].attrs["DM_t_ref"]
    del data[key].attrs["DM_f_ref"]


def get_position_from_equatorial(ras, decs, ctime):
    """Wrapper around beam_model for use with ``write_tiedbeam_locations``.

    Returns
    -------
    x, y : arrays
    """
    from beam_model.utils import get_position_from_equatorial as _get_pos
    return _get_pos(ras, decs, ctime, telescope_rotation_angle=None)
