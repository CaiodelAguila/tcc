import numpy as np
import matplotlib.pyplot as plt
def morlet(mu, sigma, freq, t):
    # se freq==0 -> usar distribuição normal!
    # Tamanho da janela gaussiana
    a = 1/(2*sigma**2)
    # Parametro de y-shift
    b = np.exp(-sigma**2/(4*a))
    # Constante de normalizacao
    c = np.sqrt(1/(np.sqrt(np.pi/2/a)*(1/2 + b**2 + np.exp(-sigma**2/a) - 2*b*np.exp(-sigma**2/8/a))))

    real = c * np.exp(-a * (t - mu)**2) * np.cos(sigma * (t - mu))
    imag = c * np.exp(-a * (t - mu)**2) * np.sin(sigma * (t - mu))

    return real, imag

def signal_cos_transit(fa, fb, t0, t):
  s = np.zeros_like(t)

  for sample in range(len(t)):
    f = fa + (fb-fa)/(1+np.exp(t[sample]-t0))
    if t[sample] > t0:
      s[sample] = np.cos(f*t[sample])
    else:
      s[sample] = np.cos(f*t[sample])
  return s

f_low = 1.0
f_high = 6.0
t0 = 4.0

sample_rate = 0.01
ti = -4.0
tf = 14.0
tvec = np.linspace(ti,tf,num=int(np.ceil((tf-ti)/sample_rate)))
svec = signal_cos_transit(f_low, f_high, t0, tvec)
plt.plot(tvec, svec)

# Filtros
f1 = [2.0, 1.0, 0.5]
f2 = [8.0, 1.0, 0.5]
f3 = [2.0, 1.0, 5.0]
f4 = [8.0, 1.0, 5.0]

f1vec = morlet(f1[0], f1[1], f1[2], tvec)
f2vec = morlet(f2[0], f2[1], f2[2], tvec)
f3vec = morlet(f3[0], f3[1], f3[2], tvec)
f4vec = morlet(f4[0], f4[1], f4[2], tvec)

plt.plot(tvec, svec)
plt.plot(tvec, f1vec)
plt.plot(tvec, f2vec)
plt.plot(tvec, f3vec)
plt.plot(tvec, f4vec)

# prod internos:
I1 = np.dot(svec, f1vec)
I2 = np.dot(svec, f2vec)
I3 = np.dot(svec, f3vec)
I4 = np.dot(svec, f4vec)
print("I1 ", I1)
print("I2 ", I2)
print("I3 ", I3)
print("I4 ", I4)

muvec = [0, 2, 4, 6, 8, 10]
I1vec = [ ]
I3vec = [ ]
for idx in range(len(muvec)):
   m = morlet(muvec[idx], f1[1], f1[2], tvec)
   I1vec.append(np.dot(svec, m))
   m = morlet(muvec[idx], f3[1], f3[2], tvec)
   I3vec.append(np.dot(svec, m))

print(I1vec)
print(I3vec)