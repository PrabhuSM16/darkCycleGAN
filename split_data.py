# -*- coding: utf-8 -*-
# Split data into training and testing sets for rain and clear
from os import listdir, makedirs
from os.path import join, exists
from shutil import copyfile

src = 'JPG_RAIN2CLEAR'
dst = 'data'
subf = ['REAL_DROPLETS','CLEAN']
set_ = ['train', 'test']
type_ = ['rain', 'clear']
split = .8

for s in set_:
    for t in type_:
        if not exists(join(dst,s,t)):
            makedirs(join(dst,s,t))
            print('Created directory: ', join(dst,s,t))
            
for j in range(len(subf)):
    datalist = listdir(join(src, subf[j]))
    print('root: ', join(src, subf[j]))
    trainlen = int(split*len(datalist))
    print('{} training samples, {} test samples'.format(trainlen, len(datalist)-trainlen))

    for i in range(len(datalist)):
        if i<int(split*(len(datalist))):
            out = join(dst, set_[0], type_[j], datalist[i])
            copyfile(join(src, subf[j], datalist[i]), out)
        else:
            out = join(dst, set_[1], type_[j], datalist[i])
            copyfile(join(src, subf[j], datalist[i]), out)