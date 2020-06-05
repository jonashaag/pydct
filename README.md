# PyDCT

Short-Time Discrete Cosine Transform (DCT) for Python. SciPy and TensorFlow implementations.
This similar to `librosa.core.{stft,istft}` and `tf.signal.{stft,inverse_stft}` but using DCT.

## Usage

```py
# Short-Time DCT
spectrogram = pydct.scipy.sdct(example_audio, frame_length=1024, frame_step=256)
spectrogram_tf = pydct.tf.sdct_tf(example_audio, frame_length=1024, frame_step=256)

# Inverse Short-Time DCT
example_audio_2 = pydct.scipy.isdct(spectrogram, frame_step=256)
example_audio_2_tf = pydct.tf.isdct_tf(spectrogram_tf, frame_step=256)

# Plot with librosa
librosa.display.specshow(
    librosa.core.amplitude_to_db(spectrogram),
    y_axis='log',
)
```

## Differences between SciPy and TensorFlow implementations

### Batching

#### SciPy

No batch support.

#### TensorFlow

Supports batching:

```py
example_audio_batch.shape  # (32, ...)
spectrogram_tf_batch = pydct.tf.sdct_tf(example_audio_batch, ...)
spectrogram_tf_batch.shape  # TensorShape([32, ..., ...])
```

### Order of dimensions

#### SciPy

Dimension order is "F-T", identical to `librosa.core.stft`: `pydct.scipy.sdct(...) -> (frequencies, time)`

#### TensorFlow

Dimension order is "T-F", identical to `tf.signal.stft`: `pydct.tf.sdct_tf(...) -> (batch, time, frequencies)`
