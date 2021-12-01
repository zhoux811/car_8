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
BASE_dir = r'C:\Users\Shio\Desktop\New_folder\ForTrain\tesla model '
model = ['3', 'x', 'y', 's']

for m in model:
    index = 1
    for f in listdir(BASE_dir + m):
        os.rename(join(BASE_dir+m, f), join(BASE_dir+m, str(index)+ '.jpg'))
        index += 1