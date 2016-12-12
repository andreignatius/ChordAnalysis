## Author: Andre Lim

#import urllib
import numpy as np
from scipy.io import wavfile
from scipy.signal import hamming
from scipy.fftpack import fft
import pylab

###########################################################
##### reading the original wav file #######################
music_file = 'bean.wav'
sampling_rate, array = wavfile.read(music_file)

#########
#Fs = 22050.0
Fs = 44100.0
window_size = 2048.0
window_size_int = 2048
num_frames = 0

def buffer_split(time_domain_data):
    global num_frames
    spectogram_mat = []
    for i in range(len(time_domain_data)):
        start = i * window_size_int
        end = start + window_size_int
        if (end <= len(time_domain_data)):
            spectogram = fft(time_domain_data[start:end] * hamming(window_size_int))
            spectogram = abs(spectogram[:len(spectogram)/2+1])
            spectogram_mat.append(spectogram)
            num_frames = num_frames + 1
    spectogram_mat = np.array(spectogram_mat)
    return spectogram_mat

def audio_reconstruction(spectogram_mat):
    # freq_amp = np.zeros((num_frames, 2))
    # M = np.max(spectogram_mat, axis=1)
    # low_values_M = M < 5
    # M[low_values_M] = 0
    # M_index = np.argmax(spectogram_mat, axis=1)
    # print spectogram_mat.shape
    # print M_index.shape
    # f = M_index / window_size * Fs
    # f[low_values_M] = 0
    # freq_amp[:,0] = f
    # freq_amp[:,1] = M

    freq_amp = np.zeros((num_frames, 2))
    M_index = np.argpartition(spectogram_mat, -4, axis=1)[:,-4]
    print M_index.shape
    #M = np.take(spectogram_mat, M_index.transpose(), axis=1)
    M = spectogram_mat[np.arange(len(spectogram_mat)), M_index]
    f = M_index / window_size * Fs
    low_values_M = M < 5
    M[low_values_M] = 0
    f[low_values_M] = 0
    freq_amp[:,0] = f
    freq_amp[:,1] = M
    np.savetxt('freq_amp_2048_4.csv', freq_amp, fmt='%.6g', delimiter=',')

    recon_wav = np.array([])
    for i in range(num_frames):
        sine_wav = M[i] * np.sin(2.0 * np.pi * f[i] / Fs * np.arange(window_size))
        recon_wav = np.concatenate((recon_wav, sine_wav))
    recon_wav_norm = recon_wav / recon_wav.max() * 32767
    return recon_wav_norm.astype(np.int16)
    
def plot_spectogram_graph(spectogram, recon_spectogram, filename):
    pylab.subplot(2,1,1)
    pylab.imshow(spectogram.T/spectogram.max(), origin='lower', aspect='auto')
    pylab.title('Spectogram for original(above) and reconstructed(below):\nfrequency bins vs time frames')
    pylab.xlabel('time frames (original wav)', fontsize=10)
    pylab.ylabel('frequency bins', fontsize=10)
    
    pylab.subplot(2,1,2)
    pylab.imshow(recon_spectogram.T/recon_spectogram.max(), origin='lower', aspect='auto')
    pylab.xlabel('time frames (reconstructed wav)', fontsize=10)
    pylab.ylabel('frequency bins', fontsize=10)
    
    pylab.savefig(filename)

                          
time_domain_data    = np.divide(array, 32768.0)
spectogram_mat      = buffer_split(time_domain_data)

recon_file          = 'reconstructed_bean_2048_4.wav'
recon_wav_norm      = audio_reconstruction(spectogram_mat)
wavfile.write(recon_file, Fs, recon_wav_norm)

sampling_rate, array_recon  = wavfile.read(recon_file)
recon_data                  = np.divide(array_recon, 32768.0)
recon_spectogram_mat        = buffer_split(recon_data)
plot_spectogram_graph(spectogram_mat, recon_spectogram_mat, 'spectogram_4.png')
