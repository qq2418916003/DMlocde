#1067-1306
import numpy as np
import  random

f=open('data/test2.txt')
lines=f.readlines()
arr=np.loadtxt('data/truearr')
f2=open('data/test3.txt','w')
for index,line in enumerate(lines):
    if index<1234 and arr[index]<0.5 and random.random()>0.7:
        continue
    f2.writelines(line)
