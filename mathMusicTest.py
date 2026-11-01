import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd

from personal_practice import original_indices, extended_audio

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
    '짜': ('jja', 'korean/jja.txt')
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
    fft_amplitudes[key]=np.abs(fft_vals[key]) * 2 / len(files[key]['time'])

# cut off top N
N = 1600
cutoffs = {}
filtered_fft = {}
filtered_ffts = {}
for key, value in fft_amplitudes.items():
    cutoffs[key]=np.partition(value,-N)[-N]
    filtered_ffts[key] = fft_vals[key].copy()
    filtered_ffts[key][value < cutoffs[key]] = 0

# filtered fft to graph
filtered_amplitudes = {}
for key, value in filtered_ffts.items():
    filtered_amplitudes[key] =np.fft.irfft(value,n=len(times[key]))

# calculate similarity
similarities = {}
for key, value in files.items():
    similarities[key] = np.dot(amplitudes[key],filtered_amplitudes[key])/(np.linalg.norm(filtered_amplitudes[key])*np.linalg.norm(amplitudes[key]))

def play(isOrigin, key):
    global sampling_rates
    if isOrigin:
        sd.play(amplitudes[key], sampling_rates)
        sd.wait()
        return
    else:
        original_indices = np.linspace(0,1,len(filtered_amplitudes[key]))
        target_indices = np.linspace(0,1,len(amplitudes[key]))
        extended_audio = np.interp(target_indices,original_indices,filtered_amplitudes[key])
        sd.play(extended_audio,sampling_rates)
        sd.wait()