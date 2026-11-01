import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd

df = pd.read_csv('korean/jja.txt', sep=r'\s+', engine='python')
print(df.info())

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

amplitudes = np.abs(fft_val)
cutoff = np.sort(amplitudes)[::-1][1599]
filtered_fft = fft_val.copy()
filtered_fft[amplitudes < cutoff] = 0

Y=np.fft.irfft(filtered_fft,n=len(df['time']))
plt.plot(df['time'],Y,linewidth=0.1)
plt.show()

origin=np.array(df['amplitude'])
approx=np.array(Y)
similarity = np.dot(origin,approx)/(np.linalg.norm(origin)*np.linalg.norm(approx))
print(similarity)

original_indices = np.linspace(0, 1, len(Y))
target_indices = np.linspace(0, 1, 27360)
extended_audio = np.interp(target_indices, original_indices, Y)

sd.play(extended_audio, 48000)
sd.wait()

original_indices = np.linspace(0, 1, len(df['amplitude']))
target_indices = np.linspace(0, 1, 27360)

extended_audio = np.interp(target_indices, original_indices, df['amplitude'])

sd.play(extended_audio, 48000)
sd.wait()
