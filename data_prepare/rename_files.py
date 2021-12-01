import os
import re, shutil, tqdm
from os import listdir, mkdir
from os.path import isfile, join, isdir

# print(re.match('^images \(\d+\)$', ts))
# print(re.match('^images \(\d+\)$', ts)[0])
# print(type(re.match('^images \(\d+\)$', ts)[0]))
# print(re.match('^images \(\d+\)$', ts)[0]+'.jpg')
BASE_dir = r'C:\Users\Shio\Desktop\honda'
index = 0

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

def rename_basic(workingDirName, sensitiveWord, fNamePattern, targetDirName, counter):
    folderNameList = []
    for folderName in listdir(workingDirName):
        if isdir(join(workingDirName, folderName)) \
                and sensitiveWord in folderName \
                and folderName.rstrip(sensitiveWord) not in listdir(join(workingDirName, targetDirName)):
            folderNameList.append(folderName)
    targetSubDirName = folderNameList[0].rstrip(sensitiveWord)[5:] \
                       + ' ' \
                       + str(min([int(year[2:]) for year in  [fn.split(' ')[0] for fn in folderNameList]])) \
                       + '-' \
                       + str(max([int(year[2:]) for year in [fn.split(' ')[0] for fn in folderNameList]]))

    print(folderNameList)
    print(targetSubDirName)
    mkdir(join(workingDirName, targetDirName, targetSubDirName))

    for folderName in listdir(workingDirName):
        # print(type(folderName))
        if isdir(join(workingDirName, folderName)) \
                and sensitiveWord in folderName:
            # and folderName.rstrip(sensitiveWord) not in listdir(join(workingDirName, targetDirName)):
            print('\n' + folderName)

            for f in tqdm.tqdm(listdir(join(workingDirName, folderName))):
                try:
                    if isfile(join(workingDirName, folderName, f)) and re.match(fNamePattern, f):
                        shutil.copy(
                            join(workingDirName, folderName, f),
                            join(workingDirName, targetDirName, targetSubDirName, str(counter) + '.jpg'))
                        counter += 1
                except Exception as e:
                    print(e)

            shutil.rmtree(join(join(workingDirName, folderName)))
            os.remove(join(workingDirName, folderName.rstrip('_files')) + '.html')

    print(counter)


rename_basic(
    workingDirName=BASE_dir,
    sensitiveWord='- Google Search_files',
    fNamePattern=r'(^images \(\d+\)$)|(^images\(\d+\)$)',
    targetDirName='CarTrain',
    counter=index)
