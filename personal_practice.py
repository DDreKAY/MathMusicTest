import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd

df = pd.read_csv('ga.txt', sep=r'\s+', engine='python')
print(df.info())
print(df.describe())
print(df.head(10))
plt.plot(df['time'],df['amplitude'],linewidth=0.1)
plt.show()

# 2. FFT 수행 (실수 신호는 rfft 사용 가능)
#fft_result = np.fft.fft()
fft_val = np.fft.rfft(df['amplitude'])
freqs = np.fft.rfftfreq(len(df['time']), d=1/48000)

# 3. 진폭 크기 구하기 (정규화: 2/N)
# k=0 및 Nyquist 주파수를 제외하고 양수 주파수 성분은 * 2를 해줌
amplitude = np.abs(fft_val) * 2 / len(df['time'])
amplitude[0] = amplitude[0] / 2  # DC 성분

# 4. 시각화
plt.figure(figsize=(8, 4))
plt.plot(freqs, amplitude)
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
'''
df = pd.read_csv('./Ga.csv', header=24)
df.info()

print(df.describe())
dfDivided=df.groupby(df.index // 1).mean()
dfDivided.info()
dfDivided.plot(x='Frequency (Hz)',y='Level (dBFS)',figsize=(16, 9))
plt.show()

X=np.linspace(0,1,48000)
amplitudes = 10 ** (dfDivided['Level (dBFS)'].values / 20)
frequencies = dfDivided['Frequency (Hz)'].values
Y = np.sum(amplitudes[:, np.newaxis] * np.sin(2 * np.pi * frequencies[:, np.newaxis] * X), axis=0)

plt.plot(X[:1000], Y[:1000])
plt.title("Synthesized Signal from Frequency Components")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
'''
original_indices = np.linspace(0, 1, len(df['amplitude']))
target_indices = np.linspace(0, 1, 27360)

extended_audio = np.interp(target_indices, original_indices, df['amplitude'])

sd.play(extended_audio, 48000)
sd.wait()