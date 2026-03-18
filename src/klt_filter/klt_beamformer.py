import numpy as np
import logging
from typing import List, Callable
from scipy.linalg import cho_factor, cho_solve
from concurrent.futures import ThreadPoolExecutor

from .math_core import form_S_from_phase, form_F_from_vis, form_M_from_V


logger = logging.getLogger(__name__)


POLS = [0, 1]


class KLTBeamformer:
    """Instrument-agnostic KL-filter beamformer.

    Parameters
    ----------
    data : h5py-like dataset
        Must support ``data["baseband"][freq_index]`` -> (ninput, ntime),
        ``data.freq``, ``data.input``, ``data.index_map``, and
        ``data.create_dataset(...)``.
    ras, decs : (npointing,) float arrays
        Sky coordinates of the pointings.
    source_names : list of str or None
        Names for each pointing.
    obs : observatory object
        Telescope position / observer (passed through, not used directly).
    frame_starts, frame_stops : array-like or None
        Per-frequency start/stop sample indices that bracket the signal.
    """

    def __init__(
        self,
        data,
        ras: np.ndarray,
        decs: np.ndarray,
        source_names,
        obs,
        frame_starts=None,
        frame_stops=None,
    ):
        self.data = data
        self.ras = np.asarray(ras)
        self.decs = np.asarray(decs)
        self.source_names = source_names
        self.obs = obs
        self.frame_starts = frame_starts
        self.frame_stops = frame_stops

        self.npointing = len(self.ras)
        self.npol = 2

    def __call__(
        self,
        freq_indices: List[int],
        gains: np.ndarray,
        fringestop_func: Callable[[int], np.ndarray],
        good_inputs_func: Callable[[int], List[np.ndarray]],
        output: str = "tiedbeam_baseband",
        clean: bool = True,
        n_workers: int = 1,
    ):
        """Process a list of frequency channels.

        Parameters
        ----------
        freq_indices : list of int
            Which frequency channels to process.
        gains : (nfreq, ninput) complex array
            Calibration gains (already reordered to match data).
        fringestop_func : callable
            ``fringestop_func(freq_index)`` -> (npointing, ninput) complex phases.
            Must be thread-safe if ``n_workers > 1``.
        good_inputs_func : callable
            ``good_inputs_func(freq_index)`` -> [x_indices, y_indices].
            Must be thread-safe if ``n_workers > 1``.
        output : str
            ``"tiedbeam_baseband"`` or ``"tiedbeam_visibilities"``.
        clean : bool
            If True, apply the KL filter before beamforming.
        n_workers : int
            Number of threads for parallel frequency processing.
            Default 1 (sequential).
        """
        self._ensure_output_dataset(output)

        def _process_one(freq_index):
            frame_start, frame_stop = self._get_frame_bounds(freq_index)
            fringestop_phase = fringestop_func(freq_index)
            good_inputs_index = good_inputs_func(freq_index)

            dispatch = {
                "tiedbeam_baseband": self._process_baseband_channel,
                "tiedbeam_visibilities": self._process_visibility_channel,
            }
            func = dispatch[output]

            try:
                func(
                    freq_index, gains, fringestop_phase,
                    good_inputs_index, frame_start, frame_stop, clean,
                )
                logger.info("finished channel %s", self.data.freq[freq_index])
            except Exception as e:
                logger.warning(
                    "channel %s failed with clean=%s: %s — retrying without cleaning",
                    self.data.freq[freq_index], clean, e,
                )
                func(
                    freq_index, gains, fringestop_phase,
                    good_inputs_index, frame_start, frame_stop, clean=False,
                )

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(_process_one, freq_indices))
        else:
            for freq_index in freq_indices:
                _process_one(freq_index)

    def _process_baseband_channel(
        self, freq_index, gains, fringestop_phase,
        good_inputs_index, frame_start, frame_stop, clean,
    ):
        """Beamform one frequency channel -> tiedbeam_baseband."""
        fringestop_by_pol = self._split_fringestop_by_pol(
            fringestop_phase, good_inputs_index
        )
        data_to_beamform = np.array(self.data["baseband"][freq_index])

        if clean:
            nan_mask = ~np.isnan(data_to_beamform[-1])
            data_for_vis = [
                data_to_beamform[good_inputs_index[pp]][:, nan_mask]
                for pp in POLS
            ]
            for pp in POLS:
                F = form_F_from_vis(data_for_vis[pp], frame_start, frame_stop)
                try:
                    c, low = cho_factor(F)
                except Exception as e:
                    logger.warning("Cholesky failed, adding regularisation: %s", e)
                    F += 1e-1 * np.eye(F.shape[0]) * np.min(np.abs(F))
                    c, low = cho_factor(F)

                # Batched solve: all pointings at once.
                # S = phase.conj() (x) phase is rank-1, so cho_solve(F, S)[:,-1]
                # is proportional to cho_solve(F, phase.conj()). The scalar
                # factor is absorbed by normalization below.
                phases = fringestop_by_pol[pp]  # (npointing, ninput)
                W = cho_solve((c, low), phases.T.conj())  # (ninput, npointing)
                V = np.conj(W)  # (ninput, npointing)

                # Normalize phase: <v, s*> = <s, s*>
                z = np.nansum(V * phases.T.conj(), axis=0)  # (npointing,)
                V /= z[np.newaxis, :]

                # Normalize amplitude: <|v s*|^2> = <|s s*|^2>
                v_dot_s = np.nansum(np.abs(V * phases.T.conj()) ** 2, axis=0)
                s_dot_s = np.nansum(np.abs(phases.T) ** 4, axis=0)
                V *= np.sqrt(s_dot_s / v_dot_s)[np.newaxis, :]

                # Beamform all pointings via matrix multiply
                data_good = np.nan_to_num(data_to_beamform[good_inputs_index[pp]])
                beamformed = V.T @ data_good  # (npointing, ntime)

                for pointing in range(self.npointing):
                    self.data["tiedbeam_baseband"][
                        freq_index, 2 * pointing + pp, :
                    ] = beamformed[pointing]

            self.data["tiedbeam_baseband"].attrs["cleaned"] = "True"
        else:
            for pp in POLS:
                phases = fringestop_by_pol[pp]  # (npointing, ninput)
                data_good = data_to_beamform[good_inputs_index[pp]]
                beamformed = phases @ data_good  # (npointing, ntime)
                for pointing in range(self.npointing):
                    self.data["tiedbeam_baseband"][
                        freq_index, 2 * pointing + pp, :
                    ] = beamformed[pointing]

    def _process_visibility_channel(
        self, freq_index, gains, fringestop_phase,
        good_inputs_index, frame_start, frame_stop, clean,
    ):
        """Beamform one frequency channel -> tiedbeam_visibilities."""
        fringestop_by_pol = self._split_fringestop_by_pol(
            fringestop_phase, good_inputs_index
        )
        data_to_beamform = np.array(self.data["baseband"][freq_index])

        if clean:
            nan_mask = ~np.isnan(data_to_beamform[-1])
            data_for_vis = [
                data_to_beamform[good_inputs_index[pp]][:, nan_mask]
                for pp in POLS
            ]
            for pp in POLS:
                data_for_vis_pp = data_for_vis[pp]
                N = data_for_vis_pp.shape[0]
                Ntimes = data_for_vis_pp.shape[1]

                # Build visibility covariance (compute once, outside pointing loop)
                C_for_compute = form_M_from_V(data_for_vis_pp).reshape(N ** 2, Ntimes)
                C_for_compute /= np.nanmean(np.abs(C_for_compute))
                F_vis = form_F_from_vis(C_for_compute)
                c, low = cho_factor(F_vis)

                # Also compute data_to_clean once (was recomputed every pointing)
                data_to_clean = form_M_from_V(data_for_vis_pp).reshape(N ** 2, Ntimes)

                # Build all s_arrays at once via batched outer product.
                # s_array_p = outer(phase_p, conj(phase_p)).flatten()
                phases = fringestop_by_pol[pp]  # (npointing, ninput)
                s_arrays = np.einsum(
                    'pi,pj->pij', phases, phases.conj()
                ).reshape(self.npointing, N ** 2).T  # (N^2, npointing)

                # Batched solve (rank-1 optimisation: solve with vector, not matrix)
                W = cho_solve((c, low), s_arrays.conj())  # (N^2, npointing)
                V = np.conj(W)  # (N^2, npointing)

                # Normalize phase
                z = np.nansum(V * s_arrays.conj(), axis=0)
                V /= z[np.newaxis, :]

                # Normalize amplitude
                v_dot_s = np.nansum(np.abs(V * s_arrays.conj()) ** 2, axis=0)
                s_dot_s = np.nansum(np.abs(s_arrays) ** 4, axis=0)
                V *= np.sqrt(s_dot_s / v_dot_s)[np.newaxis, :]

                # Beamform all pointings via matrix multiply
                beamformed = V.T @ data_to_clean  # (npointing, Ntimes)

                for pointing in range(self.npointing):
                    self.data["tiedbeam_visibilities"][
                        freq_index, 2 * pointing + pp, :
                    ] = beamformed[pointing]

            self.data["tiedbeam_visibilities"].attrs["cleaned"] = "True"
        else:
            # |phase @ baseband|^2 — no need to build S explicitly
            for pp in POLS:
                phases = fringestop_by_pol[pp]  # (npointing, ninput)
                data_good = data_to_beamform[good_inputs_index[pp]]
                beamformed = phases @ data_good  # (npointing, ntime)
                result = np.abs(beamformed) ** 2

                for pointing in range(self.npointing):
                    self.data["tiedbeam_visibilities"][
                        freq_index, 2 * pointing + pp, :
                    ] = result[pointing]

    def _get_frame_bounds(self, freq_index):
        if self.frame_starts is not None:
            return self.frame_starts[freq_index], self.frame_stops[freq_index]
        return None, None

    def _split_fringestop_by_pol(self, fringestop_phase, good_inputs_index):
        """Split (npointing, ninput) phases into per-pol arrays.

        Returns
        -------
        list of two (npointing, n_good_inputs_pol) arrays
        """
        return [
            fringestop_phase[:, good_inputs_index[0]],
            fringestop_phase[:, good_inputs_index[1]],
        ]

    def _ensure_output_dataset(self, output):
        """Create the output dataset if it doesn't exist yet."""
        if output in self.data.keys():
            return
        nfreq = self.data["baseband"].shape[0]
        ntime = self.data["baseband"].shape[-1]
        shape = (nfreq, self.npointing * self.npol, ntime)
        ds = self.data.create_dataset(output, shape=shape, dtype=self.data["baseband"].dtype)
        ds.attrs["axis"] = ["freq", "beam", "time"]
        if output == "tiedbeam_baseband":
            ds.attrs["conjugate_beamform"] = int(1)

    def write_tiedbeam_locations(self, get_position_func=None):
        """Write pointing metadata to the output dataset.

        Parameters
        ----------
        get_position_func : callable, optional
            ``get_position_func(ras, decs, ctime)`` -> (x, y).
            If None, x_400MHz and y_400MHz are set to NaN.
        """
        ib_dtype = [
            ("ra", float),
            ("dec", float),
            ("x_400MHz", float),
            ("y_400MHz", float),
            ("pol", "S1"),
        ]
        if self.source_names is not None:
            ib_dtype.append(("source_name", "<S50"))

        ib = np.empty(self.npointing * self.npol, dtype=ib_dtype)
        ib["ra"] = (self.ras[:, np.newaxis] * np.ones(2, dtype=self.ras.dtype)).flat
        ib["dec"] = (self.decs[:, np.newaxis] * np.ones(2, dtype=self.decs.dtype)).flat

        if get_position_func is not None:
            ctime = self.data["time0"]["ctime"][-1] + self.data["time0"]["ctime_offset"][-1]
            ctime = ctime + np.mean(self.data.index_map["time"]["offset_s"])
            x, y = get_position_func(self.ras, self.decs, ctime)
            ib["x_400MHz"] = (x[:, np.newaxis] * np.ones(2, dtype=x.dtype)).flat
            ib["y_400MHz"] = (y[:, np.newaxis] * np.ones(2, dtype=y.dtype)).flat
        else:
            ib["x_400MHz"] = np.nan
            ib["y_400MHz"] = np.nan

        ib["pol"] = ["S", "E"] * self.npointing
        if self.source_names is not None:
            ib["source_name"] = [y for x in self.source_names for y in (x,) * 2]

        loc = self.data.create_dataset("tiedbeam_locations", data=ib)
        loc.attrs["axis"] = ["beam"]
