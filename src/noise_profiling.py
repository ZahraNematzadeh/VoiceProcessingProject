import os 
import scipy
import time
import librosa
import numpy as np
from datetime import timedelta as td
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import correlate

#======================================================================================
def noise_profiling(audio_file, start_time, end_time):
    y, sr = librosa.load(audio_file, sr=None)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    noise_clip = y[start_sample:end_sample]
    return noise_clip, sr

def noise_preparing(fixed_part, var_train, var_test):
#sr=44100
#fixed_part = 'C:/Users/zahra/VoiceColab/dataset/e/test_train/ClusteredData/big_mass_wav'
    all_noise_clips = []
    if var_train:
        audio_files = [
            (os.path.join(fixed_part,'Positive/3-U4983590-64.wav'), 2,3),
            (os.path.join(fixed_part,'Positive/3-U2646818-89.wav'),2,3),
            (os.path.join(fixed_part,'Negative/1-U2141017-31.wav'),0,1),
            (os.path.join(fixed_part,'Negative/1-U2394528-44.wav'),1,2),
            (os.path.join(fixed_part,'Negative/1-U2608977-30.wav'),0,1.2),
            (os.path.join(fixed_part,'Negative/1-U3595713-25.wav'),0,1.35),
            (os.path.join(fixed_part,'Negative/1-U4912672-59.wav'),0,0.6),
            (os.path.join(fixed_part,'Negative/1-UWSPHC022420AC-46.wav'),4.5,5),
        ]
    if var_test:
        audio_files = [
            (os.path.join(fixed_part,'Negative/1-U2017561-23.wav'),0,1),
            (os.path.join(fixed_part,'Negative/1-U3374755-60.wav'),3.1,5),
            (os.path.join(fixed_part,'Negative/1-U4968863-96.wav'),4,5),
            (os.path.join(fixed_part,'Negative/2-U3631924-38.wav'),3,4),
            (os.path.join(fixed_part,'Negative/3-U2762753-74.wav'),0,1.1),
            (os.path.join(fixed_part,'Negative/3-U3635622-44.wav'),0,1.1),
            (os.path.join(fixed_part,'Negative/3-U8829227-19.wav'),0,1)
        ]
    
    for audio_file, start_time, end_time in audio_files:
        noise_clip, sr = noise_profiling(audio_file, start_time, end_time)
        all_noise_clips.append(noise_clip)

    #print("Number of Noise Clips:", len(all_noise_clips))
    #write('noise.wav',sr, noise_clip.astype(np.float32))
    return all_noise_clips

#==========================================================================
def find_matching_noise(input_audio, noise_clips):
    max_similarity = -np.inf
    #matched_noise = None
    matched_index = None
    matching_start = None
    
    for i, noise_clip in enumerate(noise_clips):
        similarity = np.max(correlate(input_audio, noise_clip))
        
        if similarity > max_similarity:
            max_similarity = similarity
            #matched_noise = noise_clip
            matched_index = i
            matching_start = np.argmax(correlate(input_audio, noise_clip))
    
    return matched_index, matching_start, max_similarity
#==============================================================================
#==============================================================================
def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram(signal, title, save_path = None):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter, save_path = None
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise", save_path='noise.png')
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter, save_path='thereshold and filter.png'
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal", save_path='signal.png')
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied", save_path='Mask Applied.png')
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal", save_path='Masked signal.png')
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram", save_path='Recovered spectrogram.png')
    return recovered_signal
#------------------------------------------------------------------------------
def noise_removal(input_audio, all_noise_clips):
    final_output =[]
    for data in input_audio:
        signal = data[0]
        filename = data[1]
        label = data[2]
        matched_index, _ , _ = find_matching_noise(signal, all_noise_clips)
        matched_noise_clip = all_noise_clips[matched_index]
        output = removeNoise(audio_clip= signal, noise_clip = matched_noise_clip,verbose=True,visual=False)
   
        final_output.append((output, filename, label))
   
    return final_output






#input_audio, _ = librosa.load(os.path.join(fixed_part,'train/Positive/3-U2646818-89.wav'), sr=None)  
#matched_index, matching_start, max_similarity = find_matching_noise(input_audio, all_noise_clips)

#matched_noise_path = f'matched_noise_{matched_index}.wav'
#write(matched_noise_path, sr, all_noise_clips[matched_index].astype(np.float32))



#write("reduced_noise.wav", sr, output)

