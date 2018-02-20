import aifc, math, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt 

# Populate Dict with key numbers to notes (440Hz = A4)
# http://www.sengpielaudio.com/calculator-notenames.htm
N = 88 # Number of keys on a standard piano
octv = ['A','As','B','C','Cs','D','Ds','E','F','Fs','G','Gs']
note_dict = {}
note_dict_rev = {}
for i in range(N) :
    octave = int((i+9)/12) 
    note = octv[i - octave*12]
    note_dict.update({i+1:note+str(octave)})
    note_dict_rev.update({note+str(octave):i+1})

# Function to produce piano key number from given frequency in Hz
def freq2n(f_Hz) :
    # https://en.wikipedia.org/wiki/Piano_key_frequencies
    return 49 + int(round(12 * math.log(f_Hz/440,2)))

# Function which opens a .aif file, pulls out the time series from the bytestring
# performs an FFT and returns the spectrum and frequencies in Hz
def get_spec(fname) :
    audio = aifc.open(fname)
    nchannels = audio.getnchannels()
    sampwidth = audio.getsampwidth()
    framerate = audio.getframerate()
    nframes   = audio.getnframes()
    audio_arr = np.fromstring(audio.readframes(int(nframes)),np.int32).byteswap()
    #Combine Channels
    audio_arr_ch1 = audio_arr[::2]
    audio_arr_ch2 = audio_arr[1::2]
    audio_arr = audio_arr_ch1 + audio_arr_ch2
    audio_spec = np.abs(np.fft.rfft(audio_arr))
    # For frame time cadence should use 2/framerate because every other
    # frame switches channel.
    freqs = np.fft.rfftfreq(audio_arr.size,2/framerate)
    # Remove any f=0 signal from spectra
    return freqs[100:],audio_spec[100:]

# Return peak values and locations of 1D Spectrum
def get_peaks(freqs,spectrum) :
    peak_freqs = []
    peak_vals = []
    notes=[]    
    # Find reference maximum of spectrum
    max_val = spectrum[np.where(spectrum == max(spectrum))[0][0]]  
    # Where spectrum is maximized
    ind_max = np.where(spectrum == max(spectrum))[0][0]
    # Store frequency corresponding to current peak in spectrum
    peak_freqs.append(freqs[ind_max])
    peak_vals.append(spectrum[ind_max])
    freqs_ = freqs
    spectrum_ = spectrum
    while True :
        # If spectrum max peak is now < 1/2 the initial peak, exit loop
        if spectrum_[ind_max] <= 0.01*max_val :
            print(str(spectrum_[ind_max]/max_val))
            break
        ind_max = np.where(spectrum_ == max(spectrum_))[0][0]
        peak_freqs.append(freqs_[ind_max])
        peak_vals.append(spectrum_[ind_max])
        # Remove current peak from spectrum and loop to find next biggest peak
        freqs_=np.delete(freqs_,range(ind_max-100,ind_max+100))
        spectrum_=np.delete(spectrum_,range(ind_max-100,ind_max+100))  
        plt.plot(freqs_,spectrum_)
        plt.pause(0.001)
        input()
    return peak_freqs,peak_vals
        
# Convert each peak frequency into a note
def get_notes(freqs) :
    notes = []
    for f in freqs : 
        n = freq2n(f)
        if n >= 1 : notes.append(note_dict.get(freq2n(n)))        
    # Search list of notes and remove harmonics, and repeats
    for note in notes : 
        print(note)
        n = note_dict_rev.get(note)
        if note_dict.get(n+12) in notes : notes = [x for x in notes if x != note_dict.get(n+12)]
    notes = set(notes)        
    return notes
