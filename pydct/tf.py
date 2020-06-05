import tensorflow as tf


def sdct_tf(signals, frame_length, frame_step, window_fn=tf.signal.hamming_window):
    """Compute Short-Time Discrete Cosine Transform of `signals`.

    Parameters
    ----------
    signal : Time-domain input signal(s), a `[..., n_samples]` tensor.

    frame_length : Window length and DCT frame length in samples.

    frame_step : Number of samples between adjacent DCT columns.

    window_fn : See documentation for `tf.signal.stft`.
        Default: hamming window.  Window to use for DCT.

    Returns
    -------
    dct : Real-valued T-F domain DCT matrix/matrixes, a `[..., n_frames, frame_length]` tensor.
    """
    framed = tf.signal.frame(signals, frame_length, frame_step, pad_end=False)
    if window_fn is not None:
        window = window_fn(frame_length, dtype=framed.dtype)
        framed = framed * window[tf.newaxis, :]
    return tf.signal.dct(framed, norm="ortho")


def isdct_tf(
    dcts, *, frame_step, frame_length=None, window_fn=tf.signal.hamming_window
):
    """Compute Inverse Short-Time Discrete Cosine Transform of `dct`.

    Parameters other than `dcts` are keyword-only.

    Parameters
    ----------
    dcts : DCT matrix/matrices from `sdct_tf`

    frame_step : Number of samples between adjacent DCT columns (should be the
        same value that was passed to `sdct_tf`).

    frame_length : Ignored.  Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct_tf`.

    window_fn : See documentation for `tf.signal.istft`.
        Default: hamming window.  Window to use for DCT.

    Returns
    -------
    signals : Time-domain signal(s) reconstructed from `dcts`, a `[..., n_samples]` tensor.
        Note that `n_samples` may be different from the original signals' lengths as passed to `sdct_tf`.
    """
    *_, n_frames, frame_length2 = dcts.shape
    assert frame_length in {None, frame_length2}
    signals = tf.signal.overlap_and_add(tf.signal.idct(dcts, norm="ortho"), frame_step)
    if window_fn is not None:
        window = window_fn(frame_length2, dtype=signals.dtype)
        window_frames = tf.tile(window[tf.newaxis, :], (n_frames, 1))
        window_signal = tf.signal.overlap_and_add(window_frames, frame_step)
        signals = signals / window_signal
    return signals
