import numpy as np
import librosa as lb

def carregar_audio(caminho_arquivo, sample_rate=16000):
    y, sr = lb.load(caminho_arquivo, sr=sample_rate)
    if len(y) != sr:
        y = lb.util.fix_length(y, size=sr)
    return y

def morlet(mu, sigma,freq, t): #t = np.linspace(0,250*dt,step=dt) dt do audio ste
    a = 1/(2*sigma**2)
    b = np.exp(-sigma**2/(4*a))
    c = np.sqrt(1/(np.sqrt(np.pi/2/a)*(1/2 + b**2 + np.exp(-sigma**2/a) - 2*b*np.exp(-sigma**2/8/a))))

    real = c * np.exp(-a * (t - mu)**2) * np.cos(freq * (t - mu))
    imag = c * np.exp(-a * (t - mu)**2) * np.sin(freq * (t - mu))

    return real, imag

def extract_features(audio,freq_list,sample_rate=16000):
    features = []
    dt = 1 / sample_rate
    sigma = dt/2
    t_wavelet = np.linspace(0, 250 * dt, 250, endpoint=False)  # 250 amostras de tempo

    for freq in freq_list:

        real, imag = morlet(125*dt,sigma,freq, t_wavelet)

        convolved_real = np.convolve(audio, real, mode='same')
        convolved_imag = np.convolve(audio, imag, mode='same')
        #subsampling  vec[0:end:step] #step varia de 250 em 250

        magnitude = np.sqrt(convolved_real**2 + convolved_imag**2)
        features.append(magnitude)
    return np.array(features)

#lista de frequencias em PA e PG no range de frequencias audiveis 0 a 8kHz teste com (64 e 128)