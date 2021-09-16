import sys
import os
import json
import numpy as np
import time


sys.path.append('../')
from breaker_generator.audio.virtual_microphone_pyaudio import VirualMicrophonePyaudio



# virtual_microphone = VirualMicrophonePygame()

virtual_microphone = VirualMicrophonePyaudio(10)
print(virtual_microphone.get_list_device())
name_device_source='Microphone Array (Realtek(R) Au'
# name_device_target='Speakers (Realtek(R) Audio)'
# name_device_target='CABLE Output (VB-Audio Virtual '
name_device_target='CABLE Input (VB-Audio Virtual C'

virtual_microphone.start(name_device_source, name_device_target)
# time.sleep(5)
#generator.generate()