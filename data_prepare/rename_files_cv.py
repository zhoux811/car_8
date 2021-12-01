import os
import re, shutil, tqdm
from os import listdir, mkdir
from os.path import isfile, join, isdir

# print(re.match('^images \(\d+\)$', ts))
# print(re.match('^images \(\d+\)$', ts)[0])
# print(type(re.match('^images \(\d+\)$', ts)[0]))
# print(re.match('^images \(\d+\)$', ts)[0]+'.jpg')

# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!
# DONT RUN!!!

BASE_dir = r'C:\Users\Shio\Desktop\New_folder\ForTrain\02.honda_civic_16_21'

f_list = []
for f in listdir(BASE_dir):
    # os.rename(join(BASE_dir, f), join(BASE_dir, str(index)+'.jpg'))
    f_list.append(f)

print(len(f_list))
import random

sampled = random.sample(f_list, 1250)
print(len(sampled))
print((sampled))

for f in listdir(BASE_dir):
    if f not in sampled:
        print(f)
        os.remove(join(BASE_dir, f))
