import librosa
import numpy as np
import scipy


def sdct(signal, frame_length, frame_step, window="hamming"):
    """Compute Short-Time Discrete Cosine Transform of `signal`.

    Parameters
    ----------
    signal : Time-domain input signal of shape `(n_samples,)`.

    frame_length : Window length and DCT frame length in samples.

    frame_step : Number of samples between adjacent DCT columns.

    window : Window specification passed to ``librosa.filters.get_window``.
        Default: "hamming".  Window to use for DCT.

    Returns
    -------
    dct : Real-valued F-T domain DCT matrix of shape `(frame_length, n_frames)`
    """
    framed = librosa.util.frame(signal, frame_length, frame_step)
    if window is not None:
        window = librosa.filters.get_window(window, frame_length, fftbins=True).astype(
            signal.dtype
        )
        framed = framed * window[:, np.newaxis]
    return scipy.fft.dct(framed, norm="ortho")


def isdct(dct, *, frame_step, frame_length=None, window="hamming"):
    """Compute Inverse Short-Time Discrete Cosine Transform of `dct`.

    Parameters other than `dct` are keyword-only.

    Parameters
    ----------
    dct : DCT matrix from `sdct`.

    frame_step : Number of samples between adjacent DCT columns (should be the
        same value that was passed to `sdct`).

    frame_length : Ignored. Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct`.

    window : Window specification passed to ``librosa.filters.get_window``.
        Default: "hamming".  Window to use for IDCT.

    Returns
    -------
    signal : Time-domain signal reconstructed from `dct` of shape `(n_samples,)`.
        Note that `n_samples` may be different from the original signal's length as passed to `sdct`.
    """
    frame_length2, n_frames = dct.shape
    assert frame_length in {None, frame_length2}
    signal = overlap_add(scipy.fft.idct(dct, norm="ortho"), frame_step)
    if window is not None:
        window = librosa.filters.get_window(window, frame_length2, fftbins=True).astype(
            dct.dtype
        )
        window_frames = np.tile(window[:, np.newaxis], (1, n_frames))
        window_signal = overlap_add(window_frames, frame_step)
        signal = signal / window_signal
    return signal


def overlap_add(framed, frame_step):
    """Overlap-add ("deframe") a framed signal.

    Parameters
    ----------
    framed : array_like, frames of shape `(..., frame_length, n_frames)`.

    frame_step : Overlap to use when adding frames.

    Returns
    -------
    deframed : Overlap-add ("deframed") signal.
        np.ndarray of shape `(..., (n_frames - 1) * frame_step + frame_length)`.
    """
    *shape_rest, frame_length, n_frames = framed.shape
    deframed_size = (n_frames - 1) * frame_step + frame_length
    deframed = np.zeros((*shape_rest, deframed_size), dtype=framed.dtype)
    for i in range(n_frames):
        pos = i * frame_step
        deframed[..., pos : pos + frame_length] += framed[..., i]
    return deframed
