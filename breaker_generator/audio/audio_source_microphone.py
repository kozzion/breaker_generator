
import pyaudio
import wave
 

class AudioSource:

    def start():
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "file.wav"
        
        audio = pyaudio.PyAudio()
        
        # start Recording
        stream = audio.open(
            format=FORMAT, 
            channels=CHANNELS,
            rate=RATE, 
            input=True,
            frames_per_buffer=CHUNK)
 
        frames = []
        
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()