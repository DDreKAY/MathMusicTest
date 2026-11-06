import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd

# consonants name and txt file position dictionary
consonants = {
    '가': ('ga', 'korean/ga.txt'),
    '나': ('na', 'korean/na.txt'),
    '다': ('da', 'korean/da.txt'),
    '라': ('la', 'korean/la.txt'),
    '마': ('ma', 'korean/ma.txt'),
    '바': ('ba', 'korean/ba.txt'),
    '사': ('sa', 'korean/sa.txt'),
    '아': ('aa', 'korean/aa.txt'),
    '자': ('ja', 'korean/ja.txt'),
    '차': ('cha', 'korean/cha.txt'),
    '카': ('ka', 'korean/ka.txt'),
    '타': ('ta', 'korean/ta.txt'),
    '파': ('pa', 'korean/pa.txt'),
    '하': ('ha', 'korean/ha.txt'),
    '까': ('gga', 'korean/gga.txt'),
    '따': ('dda', 'korean/dda.txt'),
    '빠': ('bba', 'korean/bba.txt'),
    '싸': ('ssa', 'korean/ssa.txt'),
    '짜': ('jja', 'korean/jja.txt'),
    ' ': ('blank', 'korean/blank.txt')
}

# read files
files={}
amplitudes={}
times={}
for i in consonants.values():
    files[i[0]]=pd.read_csv(i[1],sep=r'\s+',engine='python')
for key, value in files.items():
    amplitudes[key]=value['amplitude']
    times[key]=value['time']

# do fft
fft_vals={}
freqs={}
sampling_rates=48000
for key, value in files.items():
    fft_vals[key]=np.fft.rfft(value['amplitude'])
    freqs[key]=np.fft.rfftfreq(len(value['time']),d=1.0/sampling_rates)

# normalization
fft_amplitudes={}
for key, value in files.items():
    fft_amplitudes[key]=np.abs(fft_vals[key]) * 2 / len(times[key])

# cut off top N
N = 100 # 1600
cutoffs = {}
filtered_ffts = {}
filtered_amplitudes = {}
for key, value in fft_amplitudes.items():
    cutoffs[key]=np.partition(value,-N)[-N]
    filtered_ffts[key] = fft_vals[key].copy()
    filtered_ffts[key][value < cutoffs[key]] = 0
    filtered_amplitudes[key] = np.abs(filtered_ffts[key]) * 2 / len(times[key])

# filtered fft to sound
filtered_sounds = {}
for key, value in filtered_ffts.items():
    filtered_sounds[key] =np.fft.irfft(value,n=len(times[key]))

# calculate similarity
similarities = {}
for key, value in files.items():
    similarities[key] = np.dot(amplitudes[key],filtered_sounds[key])/(np.linalg.norm(filtered_sounds[key])*np.linalg.norm(amplitudes[key]))

def play(isOrigin, key):
    global sampling_rates
    if isOrigin:
        original_indices = np.linspace(0, 1, len(amplitudes[key]))
        target_indices = np.linspace(0, 1, int(times[key].iloc[-1]*sampling_rates))
        extended_audio = np.interp(target_indices, original_indices, amplitudes[key])
    else:
        original_indices = np.linspace(0,1,len(filtered_sounds[key]))
        target_indices = np.linspace(0,1,int(times[key].iloc[-1]*sampling_rates))
        extended_audio = np.interp(target_indices,original_indices,filtered_sounds[key])
    sd.play(extended_audio,sampling_rates)
    sd.wait()
    return

print("\n--- 1. 유사도 측정 결과 ---")
for key, sim in similarities.items():
    print(f"자음 '{key}': 유사도 = {sim:.4f}")

print("\n--- 2. 음성을 재생합니다. ---")
'''for key, value in files.items():
    play(True, key)'''
print("\n--- 2. 재생 완료 ---")

n,m = 7,5
line_width = 0.1

# (key,isOrigin,isFFT)
graphs = [('ba',False,True), ('da',False,True), ('blank',False,True), ('ga',False,True), ('blank',False,True),
          ('pa',False,True), ('ta',False,True), ('blank',False,True), ('ka',False,True), ('blank',False,True),
          ('blank',False,True), ('blank',False,True), ('ja',False,True), ('blank',False,True), ('blank',False,True),
          ('blank',False,True), ('blank',False,True), ('cha',False,True), ('blank',False,True), ('blank',False,True),
          ('blank',False,True), ('sa',False,True), ('blank',False,True), ('blank',False,True), ('ha',False,True),
          ('ma',False,True), ('na',False,True), ('blank',False,True), ('aa',False,True), ('blank',False,True),
          ('blank',False,True), ('la',False,True), ('blank',False,True), ('blank',False,True), ('blank',False,True)]
print("\n--- 3. 그래프 그리기 ---")
'''
fig, axes = plt.subplots(n,m, figsize=(2*m,1*n))
axes = axes.flatten()
#fig.suptitle('Graphs')
for idx, (key,isOrigin,isFFT) in enumerate(graphs):
    if isOrigin:
        if isFFT:
            axes[idx].plot(freqs[key], fft_amplitudes[key], linewidth=line_width)
            #axes[idx].set_xlabel("Frequency (Hz)")
            #axes[idx].set_title(f"{key} Original FFT")
        else:
            axes[idx].plot(times[key],amplitudes[key],linewidth=line_width)
            #axes[idx].set_xlabel("Time (second)")
            #axes[idx].set_title(f"{key} Original Sound")
    else:
        if isFFT:
            axes[idx].plot(freqs[key],filtered_amplitudes[key],linewidth=line_width)
            #axes[idx].set_xlabel("Frequency (Hz)")
            #axes[idx].set_title(f"{key} Filtered FFT")
        else:
            axes[idx].plot(times[key], filtered_sounds[key], linewidth=line_width)
            #axes[idx].set_xlabel("Time (second)")
            #axes[idx].set_title(f"{key} Filtered Sound")
    #axes[idx].set_ylabel("Amplitude")
    axes[idx].grid(True)
'''
# consonants bundle
consonants_bundle = {
    'Voiced':('ma','na','la','aa'),
    'Voiceless':('ba','da','sa','ja','ga','bba','dda','ssa','jja','gga','pa','ta','cha','ka','ha'),

    'Plosive':('ba','da','ga','pa','ta','ka','bba','dda','gga'),
    'Affricate':('ja','jja','cha'),
    'Fricative':('sa','ha','ssa'),
    'Liquid':('la',),
    'Nasal':('ma','na','aa'),

    'Bilabial':('ba','bba','pa','ma'),
    'Alveolar':('da','dda','ta','sa','ssa','na','la'),
    'Palatal':('ja','jja','cha'),
    'Velar':('ga','gga','ka','aa'),
    'Glottal':('ha',),

    'Plain':('ba','da','sa','ja','ga','ha'),
    'Fortis':('gga','dda','bba','ssa','jja'),
    'Aspirated':('ka','ta','pa','cha')
}

#plt.tight_layout()
#plt.show()