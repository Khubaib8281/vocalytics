import io, os, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from scipy import signal
from IPython.display import Audio, display

# Try import parselmouth (Praat bindings) for jitter/shimmer/HNR
try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAS_PRAAT = True
except Exception:
    HAS_PRAAT = False
    st.warning("Detailed vocal stability analysis (jitter, shimmer, HNR) is not available in this version.")

def load_audio(path_or_bytes, sr=16000):
    """Load audio and resample to sr. Accept path or raw bytes-like object."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data, file_sr = sf.read(io.BytesIO(path_or_bytes))
    else:
        data, file_sr = sf.read(path_or_bytes)
    # if multi-channel, convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if file_sr != sr:
        data = librosa.resample(data.astype(float), orig_sr=file_sr, target_sr=sr)
    return data.astype(np.float32), sr

def frame_rms(y, frame_length=1024, hop_length=512):
    return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

def compute_pitch(y, sr, fmin=50, fmax=500, frame_length=2048, hop_length=256):
    # returns f0 (Hz) array (NaN when unvoiced), voiced_flags
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=fmin, fmax=fmax,
                                                     sr=sr, frame_length=frame_length, hop_length=hop_length)
        # librosa.pyin returns np.ndarray with np.nan for unvoiced frames
        return f0, ~np.isnan(f0)
    except Exception as e:
        # fallback: use piptrack
        S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
        pitches, mags = librosa.piptrack(S=S, sr=sr, fmin=fmin, fmax=fmax)
        f0 = np.zeros(pitches.shape[1]) * np.nan
        for i in range(pitches.shape[1]):
            idx = mags[:, i].argmax()
            f0[i] = pitches[idx, i] if mags[idx, i] > 0 else np.nan
        return f0, ~np.isnan(f0)

def spectral_features(y, sr, n_fft=2048, hop_length=512):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    return dict(centroid=centroid, bandwidth=bandwidth, rolloff=rolloff, flatness=flatness)

def harmonic_to_noise_ratio_parselmouth(path_or_bytes):
    if not HAS_PRAAT:
        return None
    if isinstance(path_or_bytes, (bytes, bytearray)):
        snd = parselmouth.Sound(io.BytesIO(path_or_bytes))
    else:
        snd = parselmouth.Sound(path_or_bytes)
    # compute HNR via praat
    try:
        hnr = praat_call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        # HNR in dB over the signal: we'll request mean
        mean_hnr = praat_call(hnr, "Get mean", 0, 0)
        return float(mean_hnr)
    except Exception:
        return None

def jitter_shimmer_parselmouth(path_or_bytes):
    if not HAS_PRAAT:
        return None, None
    if isinstance(path_or_bytes, (bytes, bytearray)):
        snd = parselmouth.Sound(io.BytesIO(path_or_bytes))
    else:
        snd = parselmouth.Sound(path_or_bytes)
    try:
        point_process = praat_call(snd, "To PointProcess (periodic, cc)", 75, 500)
        local_jitter = praat_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        local_shimmer = praat_call(snd, point_process, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        return float(local_jitter), float(local_shimmer)
    except Exception:
        return None, None

def approximate_hnr(y, sr):
    # approximate HNR via harmonic-percussive separation energy ratio
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        pow_harm = np.sum(y_harm**2)
        pow_perc = np.sum((y - y_harm)**2)
        if pow_perc <= 1e-10:
            return 60.0  # very harmonic
        hnr_db = 10 * np.log10(pow_harm / (pow_perc + 1e-12))
        return float(hnr_db)
    except Exception:
        return None

def compute_metrics(y, sr, path_or_bytes=None,
                    frame_length=2048, hop_length=256):
    # Core features
    rms = frame_rms(y, frame_length=frame_length, hop_length=hop_length)
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))
    peak_energy = float(np.max(rms))
    duration = len(y) / sr

    # pitch
    f0, voiced = compute_pitch(y, sr, frame_length=frame_length, hop_length=hop_length)
    f0_voiced = f0[~np.isnan(f0)]
    pitch_mean = float(np.nanmean(f0)) if f0_voiced.size>0 else None
    pitch_median = float(np.nanmedian(f0)) if f0_voiced.size>0 else None
    pitch_std = float(np.nanstd(f0)) if f0_voiced.size>0 else None
    pitch_min = float(np.nanmin(f0)) if f0_voiced.size>0 else None
    pitch_max = float(np.nanmax(f0)) if f0_voiced.size>0 else None
    voiced_ratio = float(np.sum(~np.isnan(f0)) / len(f0))

    # spectral
    spec = spectral_features(y, sr, n_fft=frame_length, hop_length=hop_length)
    centroid_mean = float(np.mean(spec['centroid']))
    bandwidth_mean = float(np.mean(spec['bandwidth']))
    rolloff_mean = float(np.mean(spec['rolloff']))
    flatness_mean = float(np.mean(spec['flatness']))

    # HNR and jitter/shimmer (parselmouth if available)
    hnr = None
    jitter = None
    shimmer = None
    if path_or_bytes is not None and HAS_PRAAT:
        try:
            hnr = harmonic_to_noise_ratio_parselmouth(path_or_bytes)
            jitter, shimmer = jitter_shimmer_parselmouth(path_or_bytes)
        except Exception:
            hnr = None

    if hnr is None:
        # approximate
        hnr = approximate_hnr(y, sr)

    # Derived perceptual proxies
    # Softness: low RMS and low centroid => value in [0,1], 1 is very soft
    soft_score = 1 - np.clip((energy_mean / (energy_mean + 1e-9)) * 1.0 + (centroid_mean/ (centroid_mean+1e-9)) * 0.0005, 0, 1)
    # Tone / brightness: normalized spectral centroid
    tone_score = np.tanh(centroid_mean / (sr/2))  # 0..~1
    # Dryness/Wetness: flatness high -> more noise-like/dry; low flatness -> tonal/wet proxy
    # We'll report "wetness_score" where higher = more "wet" (reverb/tonal)
    wetness_score = 1 - np.clip(flatness_mean, 0.0, 1.0)

    # Vocal strain proxy: combine high mean pitch, high pitch std, high energy variance, high jitter
    components = []
    if pitch_mean is not None:
        components.append(np.clip((pitch_mean - 100) / 300.0, 0, 1))  # normalize voice pitch >100 Hz
    components.append(np.clip(energy_std / (energy_mean + 1e-9), 0, 3))
    if jitter is not None:
        components.append(np.clip(jitter / 1.0, 0, 3))  # jitter in absolute ratio
    # normalize components and average
    strain_score = float(np.clip(np.mean(components), 0, 1))

    metrics = dict(
        duration=duration,
        energy_mean=energy_mean,
        energy_std=energy_std,
        peak_energy=peak_energy,
        voiced_ratio=voiced_ratio,
        pitch_mean=pitch_mean,
        pitch_median=pitch_median,
        pitch_std=pitch_std,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        centroid_mean=centroid_mean,
        bandwidth_mean=bandwidth_mean,
        rolloff_mean=rolloff_mean,
        flatness_mean=flatness_mean,
        hnr_db=hnr,
        jitter=jitter,
        shimmer=shimmer,
        soft_score=float(np.clip(soft_score, 0, 1)),
        tone_score=float(np.clip(tone_score, 0, 1)),
        wetness_score=float(np.clip(wetness_score, 0, 1)),
        strain_score=float(np.clip(strain_score, 0, 1))
    )
    time = np.arange(len(rms)) * (hop_length / sr)
    return metrics, dict(rms=rms, time=time, f0=f0, f0_voiced=~np.isnan(f0), spec=spec)

def plot_all(y, sr, outputs, figsize=(12,9), title="Voice analysis"):
    rms = outputs['rms']; time = outputs['time']; f0 = outputs['f0']
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(4,1,1)
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    ax2 = plt.subplot(4,1,2, sharex=ax1)
    plt.plot(time, rms, label='RMS')
    plt.ylabel('RMS')
    plt.legend()
    ax3 = plt.subplot(4,1,3, sharex=ax1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format="%+2.0f dB")
    plt.title('Spectrogram (dB)')
    ax4 = plt.subplot(4,1,4, sharex=ax1)
    t_f = np.linspace(0, len(y)/sr, len(f0))
    plt.plot(t_f, f0, label='f0 (Hz)')
    plt.ylabel('Hz')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()    

def analyze_bytes_and_show(bytes_audio, sr=16000):
    y, sr = load_audio(bytes_audio, sr=sr)
    metrics, outputs = compute_metrics(y, sr, path_or_bytes=bytes_audio)
    fig = plot_all(y, sr, outputs, title="Analysis")
    display(Audio(y, rate=sr))
    print("Metrics summary:")   
    print(json.dumps(metrics, indent=2))

    return metrics, outputs, fig

