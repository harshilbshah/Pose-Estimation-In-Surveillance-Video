import numpy as np
from time import sleep
import glob
import os


f= open("train.txt","w+")
count=0
for file_name in sorted(os.listdir("train/")):
    print (file_name)
    if 'Male' in file_name:
        string = "train/"+file_name+str(" ")+str(0)+"\n"
    if 'Female' in file_name:
        string = "train/"+file_name+str(" ")+str(1)+"\n"
    if 'FEMALE' in file_name:
        string = "train/"+file_name+str(" ")+str(0)+"\n"
    if 'MALE' in file_name:
        string = "train/"+file_name+str(" ")+str(1)+"\n"
    f.write(string)
    print(string)
f.close()

g= open("val.txt","w+")
count=0
for file_name in sorted(os.listdir("test/")):
    print (file_name)
    if 'Male' in file_name:
        string = "test/"+file_name+str(" ")+str(0)+"\n"
    if 'Female' in file_name:
        string = "test/"+file_name+str(" ")+str(1)+"\n"
    if 'MALE' in file_name:
        string = "test/"+file_name+str(" ")+str(1)+"\n"
    if 'FEMALE' in file_name:
        string = "test/"+file_name+str(" ")+str(0)+"\n"

    g.write(string)
    print(string)
g.close()

