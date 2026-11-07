import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd
from itertools import zip_longest

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
    '무음': ('blank', 'korean/blank.txt')
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
fft_db={}
A_ref = 1.0
for key, value in files.items():
    fft_amplitudes[key]=np.abs(fft_vals[key]) * 2 / len(times[key])
    fft_db[key]=20*np.log10((fft_amplitudes[key]/A_ref) + 1e-10)

# cut off top N
N = 2000
cutoffs = {}
filtered_ffts = {}
filtered_amplitudes = {}
filtered_db = {}
for key, value in fft_amplitudes.items():
    current_N = min(N, len(value) - 1)

    cutoffs[key] = np.partition(value, -current_N)[-current_N]
    filtered_ffts[key] = fft_vals[key].copy()
    filtered_ffts[key][value < cutoffs[key]] = 0
    filtered_amplitudes[key] = np.abs(filtered_ffts[key]) * 2 / len(times[key])
    filtered_db[key] = 20*np.log10((filtered_amplitudes[key]/A_ref) + 1e-10)

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

print(f"자음별 근사 유사도 (N = {N})")
for idx, (key, sim) in enumerate(similarities.items()):
    print(f"{list(consonants.keys())[idx]}: {sim:.4f}")

n,m = 4,5
width = 0.1

# (key,isOrigin,isFFT)
'''graphs = [
    ('ba',False,True), ('da',False,True), ('blank',False,True), ('ga',False,True), ('blank',False,True),
    ('pa',False,True), ('ta',False,True), ('blank',False,True), ('ka',False,True), ('blank',False,True),
    ('blank',False,True), ('blank',False,True), ('ja',False,True), ('blank',False,True), ('blank',False,True),
    ('blank',False,True), ('blank',False,True), ('cha',False,True), ('blank',False,True), ('blank',False,True),
    ('blank',False,True), ('sa',False,True), ('blank',False,True), ('blank',False,True), ('ha',False,True),
    ('ma',False,True), ('na',False,True), ('blank',False,True), ('aa',False,True), ('blank',False,True),
    ('blank',False,True), ('la',False,True), ('blank',False,True), ('blank',False,True), ('blank',False,True)
]'''
graphs_TT = [
    ('ga', True, True), ('na', True, True), ('da', True, True), ('la', True, True), ('ma', True, True),
    ('ba', True, True), ('sa', True, True), ('aa', True, True), ('ja', True, True), ('cha', True, True),
    ('ka', True, True), ('ta', True, True), ('pa', True, True), ('ha', True, True), ('gga', True, True),
    ('dda', True, True), ('bba', True, True), ('ssa', True, True), ('jja', True, True), ('blank', True, True)
]
graphs_TF = [
    ('ga', True, False), ('na', True, False), ('da', True, False), ('la', True, False), ('ma', True, False),
    ('ba', True, False), ('sa', True, False), ('aa', True, False), ('ja', True, False), ('cha', True, False),
    ('ka', True, False), ('ta', True, False), ('pa', True, False), ('ha', True, False), ('gga', True, False),
    ('dda', True, False), ('bba', True, False), ('ssa', True, False), ('jja', True, False), ('blank', True, False)
]
graphs_FT = [
    ('ga', False, True), ('na', False, True), ('da', False, True), ('la', False, True), ('ma', False, True),
    ('ba', False, True), ('sa', False, True), ('aa', False, True), ('ja', False, True), ('cha', False, True),
    ('ka', False, True), ('ta', False, True), ('pa', False, True), ('ha', False, True), ('gga', False, True),
    ('dda', False, True), ('bba', False, True), ('ssa', False, True), ('jja', False, True), ('blank', False, True)
]
graphs_FF = [
    ('ga', False, False), ('na', False, False), ('da', False, False), ('la', False, False), ('ma', False, False),
    ('ba', False, False), ('sa', False, False), ('aa', False, False), ('ja', False, False), ('cha', False, False),
    ('ka', False, False), ('ta', False, False), ('pa', False, False), ('ha', False, False), ('gga', False, False),
    ('dda', False, False), ('bba', False, False), ('ssa', False, False), ('jja', False, False), ('blank', False, False)
]

seen = set()
for key in graphs_FF:
    if (key[0],key[1]) not in seen:
        play(key[1], key[0])
        seen.add((key[0],key[1]))

time_left_limit = 0.0
time_right_limit = 0.6
frequency_left_limit = 0.0
frequency_right_limit = 25000.0
dB_top_limit = 0.0
dB_bottom_limit = -200.0
amplitude_top_limit = 1.0
amplitude_bottom_limit = -1.0

# from 0 to NaN
for key, value in fft_amplitudes.items():
    filtered_ffts[key][filtered_ffts[key] == 0] = np.nan
    filtered_amplitudes[key] = np.abs(filtered_ffts[key]) * 2 / len(times[key])
    filtered_db[key] = 20*np.log10((filtered_amplitudes[key]/A_ref) + 1e-10)


fig, axes = plt.subplots(n,m, figsize=(4*m,2*n))
axes = axes.flatten()
fig.suptitle('Graphs')
for idx, (key,isOrigin,isFFT) in enumerate(graphs_FF):
    if isOrigin:
        if isFFT:
            axes[idx].plot(freqs[key],fft_db[key], linewidth=width)
            axes[idx].set_title(f"{key} Original FFT")
            axes[idx].set_xlabel("Frequency (Hz)")
            axes[idx].set_ylabel("dB")
            axes[idx].set_xlim(frequency_left_limit, frequency_right_limit)
            axes[idx].set_ylim(dB_bottom_limit, dB_top_limit)

        else:
            axes[idx].plot(times[key],amplitudes[key],linewidth=width)
            axes[idx].set_title(f"{key} Original Sound")
            axes[idx].set_xlabel("Time (second)")
            axes[idx].set_ylabel("Amplitude")
            axes[idx].set_xlim(time_left_limit, time_right_limit)
            axes[idx].set_ylim(amplitude_bottom_limit, amplitude_top_limit)

    else:
        if isFFT:
            axes[idx].plot(freqs[key],filtered_db[key], linewidth=width)
            axes[idx].set_title(f"{key} Filtered FFT")
            axes[idx].set_xlabel("Frequency (Hz)")
            axes[idx].set_ylabel("dB")
            axes[idx].set_xlim(frequency_left_limit, frequency_right_limit)
            axes[idx].set_ylim(dB_bottom_limit, dB_top_limit)
        else:
            axes[idx].plot(times[key], filtered_sounds[key], linewidth=width)
            axes[idx].set_title(f"{key} Filtered Sound")
            axes[idx].set_xlabel("Time (second)")
            axes[idx].set_ylabel("Amplitude")
            axes[idx].set_xlim(time_left_limit, time_right_limit)
            axes[idx].set_ylim(amplitude_bottom_limit, amplitude_top_limit)
    axes[idx].grid(True)

plt.tight_layout()
plt.show()

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

