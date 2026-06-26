import numpy as np
import librosa as lb

def carregar_audio(caminho_arquivo, sample_rate=16000):
    y, sr = lb.load(caminho_arquivo, sr=sample_rate)
    if len(y) != sr:
        y = lb.util.fix_length(y, size=sr)
    return y

def morlet(mu, sigma, t):
    a = 1/(2*sigma**2)
    b = np.exp(-sigma**2/(4*a))
    c = np.sqrt(1/(np.sqrt(np.pi/2/a)*(1/2 + b**2 + np.exp(-sigma**2/a) - 2*b*np.exp(-sigma**2/8/a))))

    real = c * np.exp(-a * (t - mu)**2) * np.cos(sigma * (t - mu))
    imag = c * np.exp(-a * (t - mu)**2) * np.sin(sigma * (t - mu))

    return real, imag

def extract_features(audio,sigma_list,sample_rate=16000):
    features = []

    t = int(sample_rate * 0.2)  # 0.5 seconds
    t_wavelet = np.linspace(-t / 2, t / 2, t)

    for sigma in sigma_list:

        real, imag = morlet(0, sigma, t_wavelet)

        convolved_real = np.convolve(audio, real, mode='same')
        convolved_imag = np.convolve(audio, imag, mode='same')

        magnitude = np.sqrt(convolved_real**2 + convolved_imag**2)
        features.append(magnitude)
    return np.array(features)