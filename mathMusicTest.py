import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd
import wave
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
N = 100
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


# --- WAV 파일로 저장하는 함수 (내장 wave 모듈 사용) ---
def save_filtered_audio_to_wav(key, filename="output.wav"):
    global sampling_rates, filtered_sounds, times

    original_indices = np.linspace(0, 1, len(filtered_sounds[key]))
    target_indices = np.linspace(0, 1, int(times[key].iloc[-1] * sampling_rates))
    extended_audio = np.interp(target_indices, original_indices, filtered_sounds[key])

    # -1.0 ~ 1.0 범위를 16비트 PCM 정수형(-32768 ~ 32767)으로 변환
    audio_data = np.clip(extended_audio, -1.0, 1.0)
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # wave 모듈을 이용해 파일 저장 (추가 라이브러리 설치 불필요)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)  # 모노 채널
        wf.setsampwidth(2)  # 16-bit (2 bytes)
        wf.setframerate(sampling_rates)
        wf.writeframes(audio_int16.tobytes())

    print(f"자음 '{key}' 필터링 음원이 '{filename}' 파일로 저장되었습니다.")

print(f"자음별 근사 유사도 (N = {N})")
for idx, (key, sim) in enumerate(similarities.items()):
    print(f"{list(consonants.keys())[idx]}: {sim:.4f}")

# 무음을 제외한 모든 자음 음원을 WAV 파일로 자동 저장
print("\n--- 2. 음원 파일 저장 중 ---")
for key in filtered_sounds.keys():
    if key != 'blank':
        save_filtered_audio_to_wav(key, filename=f"{key}_filtered.wav")

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

'''seen = set()
for key in graphs_FF:
    if (key[0],key[1]) not in seen:
        play(key[1], key[0])
        seen.add((key[0],key[1]))'''

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

'''# 1. '가' 소리의 진폭을 절반으로 줄이고 그래프 시각화 및 재생
half_amplitude = amplitudes['ga'] * 0.5

plt.figure(figsize=(10, 3))
plt.plot(times['ga'], half_amplitude, linewidth=width)
plt.title("'가' Original Sound - Half Amplitude")
plt.xlabel("Time (second)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

extended_audio_half_amp = np.interp(
    np.linspace(0, 1, int(times['ga'].iloc[-1] * sampling_rates)),
    np.linspace(0, 1, len(half_amplitude)),
    half_amplitude
)
sd.play(extended_audio_half_amp, sampling_rates)
sd.wait()


# 주파수 스펙트럼 스케일링 함수 (2배 및 절반 변환용)
def modify_frequency_scale(fft_val, scale):
    n_bins = len(fft_val)
    x_old = np.arange(n_bins)
    x_new = x_old * scale
    real_interp = np.interp(x_new, x_old, fft_val.real, left=0, right=0)
    imag_interp = np.interp(x_new, x_old, fft_val.imag, left=0, right=0)
    return real_interp + 1j * imag_interp


# 2-1. '가' 소리 주파수 2배 변환 및 재생/시각화
fft_double = modify_frequency_scale(fft_vals['ga'], 2.0)
sound_double = np.fft.irfft(fft_double, n=len(times['ga']))

plt.figure(figsize=(10, 3))
plt.plot(times['ga'], sound_double, linewidth=width)
plt.title("'가' Frequency Doubled Sound (x2)")
plt.xlabel("Time (second)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

extended_audio_double = np.interp(
    np.linspace(0, 1, int(times['ga'].iloc[-1] * sampling_rates)),
    np.linspace(0, 1, len(sound_double)),
    sound_double
)
sd.play(extended_audio_double, sampling_rates)
sd.wait()


# 2-2. '가' 소리 주파수 절반(0.5배) 변환 및 재생/시각화
fft_half = modify_frequency_scale(fft_vals['ga'], 0.5)
sound_half = np.fft.irfft(fft_half, n=len(times['ga']))

plt.figure(figsize=(10, 3))
plt.plot(times['ga'], sound_half, linewidth=width)
plt.title("'가' Frequency Halved Sound (x0.5)")
plt.xlabel("Time (second)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

extended_audio_half = np.interp(
    np.linspace(0, 1, int(times['ga'].iloc[-1] * sampling_rates)),
    np.linspace(0, 1, len(sound_half)),
    sound_half
)
sd.play(extended_audio_half, sampling_rates)
sd.wait()'''


'''# 가장 큰 진폭을 가지는 주파수 성분 주위의 성분들까지 함께 위상차를 옮겨주는 반복문 ('q' 입력 시 종료)
ga_fft = fft_vals['ga'].copy()
max_idx = np.argmax(np.abs(ga_fft))

print("\n--- 최대 진폭 주파수 및 주변 성분 위상차 조절 반복문 ('q' 입력시 종료) ---")
while True:
    user_input = input("위상차(라디안) 및 주변 범위(반경) 입력 (예: 1.57 5 또는 1.57, 종료: q): ")
    if user_input.strip().lower() == 'q':
        break

    parts = user_input.split()
    try:
        phase_shift = float(parts[0])
        # 기본 반경을 5로 설정하고, 사용자가 입력했다면 해당 값 사용
        radius = int(parts[1]) if len(parts) > 1 else 5
    except ValueError:
        print("유효한 숫자를 입력하거나 종료하려면 'q'를 입력하세요.")
        continue

    # 주파수 성분 복사
    modified_fft = ga_fft.copy()

    # 최대 주파수 인덱스 주변 (max_idx - radius부터 max_idx + radius까지) 위상차 적용
    start_idx = max(0, max_idx - radius)
    end_idx = min(len(modified_fft) - 1, max_idx + radius)

    for idx in range(start_idx, end_idx + 1):
        modified_fft[idx] = modified_fft[idx] * np.exp(1j * phase_shift)

    sound_phase_shifted = np.fft.irfft(modified_fft, n=len(times['ga']))

    # 그래프 시각화
    plt.figure(figsize=(10, 3))
    plt.plot(times['ga'], sound_phase_shifted, linewidth=width)
    plt.title(f"'가' Neighbor Phase Shifted (Center Index: {max_idx}, Radius: ±{radius}, Shift: {phase_shift})")
    plt.xlabel("Time (second)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # 오디오 재생
    extended_audio_phase = np.interp(
        np.linspace(0, 1, int(times['ga'].iloc[-1] * sampling_rates)),
        np.linspace(0, 1, len(sound_phase_shifted)),
        sound_phase_shifted
    )
    sd.play(extended_audio_phase, sampling_rates)
    sd.wait()'''