import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd

df = pd.read_csv('./Ga.csv', header=24)
df.info()

print(df.describe())
dfDivided=df.groupby(df.index // 1).mean()
dfDivided.info()
'''dfDivided.plot(x='Frequency (Hz)',y='Level (dBFS)',figsize=(16, 9))
plt.show()'''

X=np.linspace(0,1,48000)
amplitudes = 10 ** (dfDivided['Level (dBFS)'].values / 20)
frequencies = dfDivided['Frequency (Hz)'].values
Y = np.sum(amplitudes[:, np.newaxis] * np.sin(2 * np.pi * frequencies[:, np.newaxis] * X), axis=0)

plt.plot(X[:1000], Y[:1000]) # 100,000개를 다 그리면 뭉개지므로 앞의 1000개 샘플만 확인
plt.title("Synthesized Signal from Frequency Components")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

sd.play(Y, 48000)
sd.wait()