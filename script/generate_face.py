import sys
import os
import json

sys.path.append('../')
from breaker_generator.face.generator_face import GeneratorFace

with open('config.cfg', 'r') as file:
    config = json.load(file)

generator = GeneratorFace()
generator.run_train()
#generator.generate()