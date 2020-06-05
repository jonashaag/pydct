import random

import librosa
import numpy as np
import pytest
import tensorflow as tf

import pydct.scipy
import pydct.tf

DURATION = 5
EXAMPLE_AUDIO, SR = librosa.core.load(
    librosa.util.example_audio_file(), offset=random.randint(0, 50), duration=DURATION
)


@pytest.mark.parametrize("backend", ["scipy", "tf"])
@pytest.mark.parametrize("frame_length", [2048, 1024, 512, 256])
@pytest.mark.parametrize("frame_step_ratio", [4, 2, 1.1, 1.001])
def test_single(backend, frame_length, frame_step_ratio):
    frame_step = int(frame_length // frame_step_ratio)
    if backend == "scipy":
        scipy_sdct = pydct.scipy.sdct(EXAMPLE_AUDIO, frame_length, frame_step)
        assert scipy_sdct.shape[0] == frame_length
        reconstructed = pydct.scipy.isdct(scipy_sdct, frame_step=frame_step)
    else:
        tf_sdct = pydct.tf.sdct_tf(EXAMPLE_AUDIO, frame_length, frame_step)
        assert tf_sdct.shape[1] == frame_length
        reconstructed = pydct.tf.isdct_tf(tf_sdct, frame_step=frame_step)
    assert mse(EXAMPLE_AUDIO[: len(reconstructed)], reconstructed) <= 1e-11


def test_batch_tf():
    audio_samples = [EXAMPLE_AUDIO[i * SR : (i + 1) * SR] for i in range(DURATION)]

    batched_sdct = pydct.tf.sdct_tf(audio_samples, 1024, 512)
    nonbatched_sdct = [pydct.tf.sdct_tf(sample, 1024, 512) for sample in audio_samples]

    assert tf.math.reduce_all(batched_sdct == nonbatched_sdct)

    batched_isdct = pydct.tf.isdct_tf(batched_sdct, frame_step=512,)
    nonbatched_isdct = [
        pydct.tf.isdct_tf(sdct, frame_step=512,) for sdct in batched_sdct
    ]

    assert tf.math.reduce_all(batched_isdct == nonbatched_isdct)


def mse(a, b):
    return np.mean((a - b) ** 2)
