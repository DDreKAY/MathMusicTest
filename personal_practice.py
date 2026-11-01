import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd

# 1. 파일 받고 시각화
df1 = pd.read_csv('korean/ga.txt', sep=r'\s+', engine='python')
df2 = pd.read_csv('korean/ka.txt', sep=r'\s+', engine='python')
df3 = pd.read_csv('korean/aa.txt', sep=r'\s+', engine='python')

fig, axes = plt.subplots(3, 1, figsize=(6, 8))
plt.xlabel("Time")
plt.ylabel("Amplitude")

axes[0].plot(df1['time'],df1['amplitude'],linewidth=0.1, color='red')
axes[0].set_title("plain")
axes[0].grid(True)
axes[1].plot(df2['time'],df2['amplitude'],linewidth=0.1, color='blue')
axes[1].set_title("fortis")
axes[1].grid(True)
axes[2].plot(df3['time'],df3['amplitude'],linewidth=0.1, color='green')
axes[2].set_title("aspirated")
axes[2].grid(True)
plt.tight_layout()
plt.show()

# 2. FFT 수행
fft_val1 = np.fft.rfft(df1['amplitude'])
fft_val2 = np.fft.rfft(df2['amplitude'])
fft_val3 = np.fft.rfft(df3['amplitude'])

freqs1 = np.fft.rfftfreq(len(df1['time']), d=1/48000)
freqs2 = np.fft.rfftfreq(len(df2['time']), d=1/48000)
freqs3 = np.fft.rfftfreq(len(df3['time']), d=1/48000)

# 3. 진폭 크기 구하기 (정규화: 2/N)
# k=0 및 Nyquist 주파수를 제외하고 양수 주파수 성분은 * 2를 해줌
amplitude1 = np.abs(fft_val1) * 2 / len(df1['time'])
amplitude2 = np.abs(fft_val2) * 2 / len(df2['time'])
amplitude3 = np.abs(fft_val3) * 2 / len(df3['time'])

amplitude1[0] = amplitude1[0] / 2  # DC 성분
amplitude2[0] = amplitude2[0] / 2  # DC 성분
amplitude3[0] = amplitude3[0] / 2  # DC 성분

# 4. FFT 시각화

fig, axes = plt.subplots(3, 1, figsize=(6, 8))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

axes[0].plot(freqs1, amplitude1, color='red', linewidth=0.1)
axes[0].set_title("plain")
axes[0].grid(True)
axes[1].plot(freqs2, amplitude2, color='blue', linewidth=0.1)
axes[1].set_title("fortis")
axes[1].grid(True)
axes[2].plot(freqs3, amplitude3, color='green', linewidth=0.1)
axes[2].set_title("aspirated")
axes[2].grid(True)
plt.tight_layout()
plt.show()

# 5. 진폭 상위 n개 선정
n=1599 #1599

amplitudes1 = np.abs(fft_val1)
cutoff1 = np.sort(amplitudes1)[::-1][n]
filtered_fft1 = fft_val1.copy()
filtered_fft1[amplitudes1 < cutoff1] = 0

amplitudes2 = np.abs(fft_val2)
cutoff2 = np.sort(amplitudes2)[::-1][n]
filtered_fft2 = fft_val2.copy()
filtered_fft2[amplitudes2 < cutoff2] = 0

amplitudes3 = np.abs(fft_val3)
cutoff3 = np.sort(amplitudes3)[::-1][n]
filtered_fft3 = fft_val3.copy()
filtered_fft3[amplitudes3 < cutoff3] = 0

#5.5 근사 FFT 시각화
fig, axes = plt.subplots(3, 1, figsize=(6, 8))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")

axes[0].plot(freqs1, np.abs(filtered_fft1), color='red', linewidth=0.1)
axes[0].set_title("plain")
axes[0].grid(True)
axes[1].plot(freqs2, np.abs(filtered_fft2), color='blue', linewidth=0.1)
axes[1].set_title("fortis")
axes[1].grid(True)
axes[2].plot(freqs3, np.abs(filtered_fft3), color='green', linewidth=0.1)
axes[2].set_title("aspirated")
axes[2].grid(True)
plt.tight_layout()
plt.show()


# 6. 근사 소리 시각화
Y1=np.fft.irfft(filtered_fft1,n=len(df1['time']))
Y2=np.fft.irfft(filtered_fft2,n=len(df2['time']))
Y3=np.fft.irfft(filtered_fft3,n=len(df3['time']))

fig, axes = plt.subplots(3, 1, figsize=(6, 8))
plt.xlabel("Time")
plt.ylabel("Amplitude")

axes[0].plot(df1['time'],Y1,linewidth=0.1, color='red')
axes[0].set_title("plain")
axes[0].grid(True)
axes[1].plot(df2['time'],Y2,linewidth=0.1, color='blue')
axes[1].set_title("fortis")
axes[1].grid(True)
axes[2].plot(df3['time'],Y3,linewidth=0.1, color='green')
axes[2].set_title("aspirated")
axes[2].grid(True)
plt.tight_layout()
plt.show()

# 7. 유사도 분석
origin1=np.array(df1['amplitude'])
approx1=np.array(Y1)
similarity1 = np.dot(origin1,approx1)/(np.linalg.norm(origin1)*np.linalg.norm(approx1))

origin2=np.array(df2['amplitude'])
approx2=np.array(Y2)
similarity2 = np.dot(origin2,approx2)/(np.linalg.norm(origin2)*np.linalg.norm(approx2))

origin3=np.array(df3['amplitude'])
approx3=np.array(Y3)
similarity3 = np.dot(origin3,approx3)/(np.linalg.norm(origin3)*np.linalg.norm(approx3))
print(similarity1,similarity2,similarity3)

# 8. 근사 음성 재생
original_indices = np.linspace(0, 1, len(Y1))
target_indices = np.linspace(0, 1, 27360)
extended_audio = np.interp(target_indices, original_indices, Y1)
sd.play(extended_audio, 48000)
sd.wait()
original_indices = np.linspace(0, 1, len(Y2))
target_indices = np.linspace(0, 1, 27360)
extended_audio = np.interp(target_indices, original_indices, Y2)
sd.play(extended_audio, 48000)
sd.wait()
original_indices = np.linspace(0, 1, len(Y3))
target_indices = np.linspace(0, 1, 27360)
extended_audio = np.interp(target_indices, original_indices, Y3)
sd.play(extended_audio, 48000)
sd.wait()


# 9. 원본 음성 재생
original_indices = np.linspace(0, 1, len(df1['amplitude']))
target_indices = np.linspace(0, 1, 27360)
extended_audio = np.interp(target_indices, original_indices, df1['amplitude'])
sd.play(extended_audio, 48000)
sd.wait()
original_indices = np.linspace(0, 1, len(df2['amplitude']))
target_indices = np.linspace(0, 1, 27360)
extended_audio = np.interp(target_indices, original_indices, df2['amplitude'])
sd.play(extended_audio, 48000)
sd.wait()
original_indices = np.linspace(0, 1, len(df3['amplitude']))
target_indices = np.linspace(0, 1, 27360)
extended_audio = np.interp(target_indices, original_indices, df3['amplitude'])
sd.play(extended_audio, 48000)
sd.wait()