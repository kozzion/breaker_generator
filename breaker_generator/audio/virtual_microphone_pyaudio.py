import os
import pyaudio
import numpy as np

class VirualMicrophonePyaudio:

    # install vb-audio for virual microphone
    # https://vb-audio.com/Cable/
    
    # also pyaudio requires build tools or use the precompiled wheel (latter works better)
    # https://visualstudio.microsoft.com/visual-cpp-build-tools/
    
    def __init__(self, factor_boost) -> None:
       self.pa = pyaudio.PyAudio()
       self.factor_boost = factor_boost
    
    def get_list_device(self):
        for i in range(self.pa.get_device_count()):
            print(self.pa.get_device_info_by_index(i))

    def get_default_host_api_info(self):
        print(self.pa.get_default_host_api_info())

    def get_device_info_by_name(self, name_device):
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['name'] == name_device:
                return info
        raise Exception('No device with name: ' + name_device)


    def start(self, name_device_source, name_device_target):
        info_source = self.get_device_info_by_name(name_device_source)
        info_target = self.get_device_info_by_name(name_device_target)
        print(info_source)
        print(info_target)

        stream_source = self.pa.open(
            input_device_index=info_source['index'], 
            rate=44100,
            channels=info_source['maxInputChannels'],
            format=pyaudio.paInt16,
            input=True,                   
            frames_per_buffer=1024
        )

        print(info_target['maxOutputChannels'])
        stream_target = self.pa.open(
            output_device_index=info_target['index'], 
            rate=44100,
            channels=2,
            format=pyaudio.paInt16,
            output=True,
            frames_per_buffer=1024, 
        )
  
        while True:
            buffer = stream_source.read(1024)           
            array_values = np.frombuffer(buffer, dtype=np.int16)
            array_values = array_values.copy()
            # array_values.setflags(write=1)
            array_values *= self.factor_boost
            # import matplotlib.pyplot as plt
            # plt.plot(amplitude)
            # plt.show()
            buffer = array_values.astype(np.int16).tostring()

            stream_target.write(buffer)
            #stream_input.read(44100)


# # read 5 seconds of the input stream
# input_audio = stream_in.read(5 * 48000)

#         stream_output = self.pa.open(
#             format              = self.pa.get_format_from_width(wf.getsampwidth()),
#             channels            = info,
#             rate                = wf.getframerate(),
#             output              = True,
#             output_device_index = 16,
#             frames_per_buffer   =1024
#         )


#     {'index': 2, 
#     'structVersion': 2, 
#     'name': 'CABLE Output (VB-Audio Virtual ', 
#     'hostApi': 0, 
#     'maxInputChannels': 8, 
#     'maxOutputChannels': 0, 
#     'defaultLowInputLatency': 0.09, 
#     'defaultLowOutputLatency': 0.09, 
#     'defaultHighInputLatency': 0.18, 
#     'defaultHighOutputLatency': 0.18, 
#     'defaultSampleRate': 44100.0}
# :
# stream_out = pa.open(
#     rate=wav_file.getframerate(),     # sampling rate
#     channels=wav_file.getnchannels(), # number of output channels
#     format=pa.get_format_from_width(wav_file.getsampwidth()),  # sample format and length
#     output=True,             # output stream flag
#     output_device_index=4,   # output device index
#     frames_per_buffer=1024,  # buffer length
# )

# p = pyaudio.PyAudio()
# 