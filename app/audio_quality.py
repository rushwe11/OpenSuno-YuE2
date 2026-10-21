"""Cheap catastrophic-output checks shared by MLX and PyTorch renderers."""

import numpy as np


def analyze(audio, sample_rate=48000):
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if signal.ndim != 1 or not len(signal):
        return {'passed': False, 'reasons': ['empty audio']}
    window_size, hop = 4096, 2048
    window = np.hanning(window_size).astype(np.float32)
    spectrum = np.zeros(window_size // 2 + 1, dtype=np.float64)
    centroids = []
    frequencies = np.fft.rfftfreq(window_size, 1 / sample_rate)
    for start in range(0, max(1, len(signal) - window_size + 1), hop):
        frame = signal[start:start + window_size]
        if len(frame) < window_size:
            frame = np.pad(frame, (0, window_size - len(frame)))
        power = np.abs(np.fft.rfft(frame * window)) ** 2
        spectrum += power
        centroids.append(float((frequencies * power).sum() / (power.sum() + 1e-12)))
    audible = spectrum[(frequencies >= 20) & (frequencies < 12000)].sum() + 1e-12
    sub_ratio = float(spectrum[(frequencies >= 20) & (frequencies < 80)].sum() / audible)
    musical_ratio = float(spectrum[(frequencies >= 250) & (frequencies < 12000)].sum() / audible)
    rms = float(np.sqrt(np.mean(signal * signal)))
    reasons = []
    if not np.isfinite(signal).all():
        reasons.append('non-finite audio')
    if rms < 1e-4:
        reasons.append('near-silent output')
    if sub_ratio > .75 and musical_ratio < .02:
        reasons.append('degenerate sub-bass-only output')
    return {
        'passed': not reasons, 'reasons': reasons, 'rms': rms,
        'sub_20_80_percent': sub_ratio * 100,
        'energy_above_250_percent': musical_ratio * 100,
        'median_spectral_centroid_hz': float(np.median(centroids)) if centroids else 0.,
    }


def check_and_save(destination, audio, write_audio):
    """Save metrics; preserve rejected candidates without promoting them to the library."""
    import json
    destination = destination
    quality = analyze(audio)
    (destination / 'quality.json').write_text(json.dumps(quality, indent=2))
    if quality['passed']:
        return quality
    write_audio(destination / 'audio-rejected.wav', audio)
    raise RuntimeError('Audio failed the quality gate: ' + ', '.join(quality['reasons']) +
                       '. The rejected candidate was preserved; use another seed or mode.')
