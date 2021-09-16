from scipy.fftpack import fft
from scipy import signal
import soundfile as sf
import wave as w
import numpy as np
import matplotlib.pyplot as plt

#Getting original wave form
data, fs = sf.read('Vijaya Mawatha 2.wav', dtype='float32')
#.......................................................................................................................

# open the audio file and extract some information
spf = w.open('Vijaya Mawatha 2.wav','r')
(nChannels, sampWidth, sampleRate, nFrames, compType, compName) = spf.getparams()

# extract audio from wav file
input_signal = spf.readframes(-1)
input_signal = np.fromstring(input_signal, 'Int16')
spf.close()

# create the filter
N = 4
nyq = 0.5 * sampleRate
low = 100 / nyq
high = 500 / nyq
b, a = signal.butter(N, [low, high], btype='band')

# apply filter
output_signal = signal.filtfilt(b, a, input_signal)

# ceate output file
wav_out = w.open("output.wav", "w")
wav_out.setparams((nChannels, sampWidth, sampleRate, nFrames, compType, compName))

# write to output file
wav_out.writeframes(output_signal.tobytes())
wav_out.close()
#.......................................................................................................................

#Frequency Analysis of Original Signal

#fast fourier transform of original signal
FT_Y = fft(data)
Shift_FT_Y=np.fft.fftshift(FT_Y)

#.......................................................................................................................
#Frequency Analysis of Butterworth filtred  Signal
BFT_Y = fft(output_signal)
BShift_FT_Y=np.fft.fftshift(BFT_Y)

# plot the Original signals
plt.subplot(2,1,1)
plt.plot(data)
plt.title("Original Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

# plot the filtered signals
plt.subplot(2,1,2)
plt.plot(output_signal)
plt.title("Original Signal after applying Butterworth Filter")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

#.......................................................................................................................

#plot the original time domain signal
plt.subplot(2,2,1)
plt.plot(data)
plt.title("Original Signal in Time Domain")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

#Plot the Original Signal in Frequency Domain
plt.subplot(2,2,2)
plt.plot(abs(FT_Y))
plt.title("Original Signal in Frequency Domain")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid()

#Plot the Shifted Frequency Domain Signal
plt.subplot(2,2,3)
plt.plot(abs(Shift_FT_Y))
plt.title("Shifted Frequency Domain Signal")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude(V)")
plt.grid()

#Plot the Shifted Positive Side Frequency Domain Signal
N=len(data)
f=np.arange(-fs/2,fs/2,fs/N)
plt.subplot(2,2,4)
plt.plot(f[int(N/2):N],abs(Shift_FT_Y)[int(N/2):N]/(N/2))
plt.title("Shifted Positive Side Frequency Domain Signal")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude(V)")
plt.grid()
plt.show()
#.......................................................................................................................

#plot the Butterworth filtered time domain signal
plt.subplot(2,2,1)
plt.plot(output_signal)
plt.title("Butterworth filtered Signal in Time Domain")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

#Plot the Butterworth filtered Signal in Frequency Domain
plt.subplot(2,2,2)
plt.plot(abs(BFT_Y))
plt.title("Butterworth filtered Signal in Frequency Domain")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.grid()

#Plot the Butterworth filtered Shifted Frequency Domain Signal
plt.subplot(2,2,3)
plt.plot(abs(BShift_FT_Y))
plt.title("Shifted Frequency Domain Signal of Butterworth filtered")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude(V)")
plt.grid()

#Plot the Butterworth filtered Shifted Positive Side Frequency Domain Signal
Nb=len(output_signal)
fb=np.arange(-sampleRate/2,sampleRate/2,sampleRate/Nb)
plt.subplot(2,2,4)
plt.plot(fb[int(Nb/2):Nb],abs(BShift_FT_Y)[int(Nb/2):Nb]/(Nb/2))
plt.title("Shifted Positive Side Frequency Domain Signal of Butterworth filtered")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude(V)")
plt.grid()

plt.show()
