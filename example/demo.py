from MK_MMD import MKMMD
import numpy as np
import random

random.seed(42)  

source = []
while len(source) < 20:
    random_number = random.uniform(16, 18)
    if round(random_number,2) not in source:
        formatted_number = round(random_number,2)
        source.append(formatted_number)
    source.sort()

target = []
while len(target) < 20:
    random_number = random.uniform(14, 16)
    if round(random_number,2) not in target:
        formatted_number = round(random_number,2)
        target.append(formatted_number)

random.shuffle(source) 
random.shuffle(target)  

MK_MMD,weights = MKMMD().predict(source, target)

print(MK_MMD)
print(weights)