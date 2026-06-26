import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def dynamic_filter(eeg_signal, fs, segment_len=128):
    filtered_segments = []
    for i in range(0, len(eeg_signal), segment_len):
        segment = eeg_signal[i:i + segment_len]
        b, a = butter_bandpass(1, 40, fs)  
        filtered_segment = filtfilt(b, a, segment)
        filtered_segments.extend(filtered_segment)
    return np.array(filtered_segments)


def normalize_features(features):
    mean_val = np.mean(features, axis=0)
    std_val = np.std(features, axis=0)
    return (features - mean_val) / (std_val + 1e-8)  


def acquire_eeg(emotion_state, fs=256, duration=10):
    t = np.linspace(0, duration, fs * duration)
    if emotion_state == "happy":
        eeg_signal = np.sin(10 * 2 * np.pi * t) + 0.5 * np.random.randn(len(t)) 
    elif emotion_state == "sad":
        eeg_signal = np.sin(5 * 2 * np.pi * t) + 0.5 * np.random.randn(len(t))  
    else:
        eeg_signal = 0.5 * np.random.randn(fs * duration) 
    filtered_signal = dynamic_filter(eeg_signal, fs)
    features = np.array([np.mean(filtered_signal[i:i + 32]) for i in range(0, len(filtered_signal), 32)]) 
    normalized_features = normalize_features(features.reshape(-1, 1)).flatten()
    return eeg_signal, filtered_signal, normalized_features


if __name__ == "__main__":
    eeg, filtered, features = acquire_eeg("happy")
    print("原始脑电信号长度:", len(eeg))
    print("滤波后脑电信号长度:", len(filtered))
    print("标准化特征集长度:", len(features))