#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 17:46:27 2022

@author: saanchi
"""

"""
    Rewriting Rudy's Pupil precosessing code: 
    https://github.com/rudyvdbrink/Pupil_code/blob/main/scripts/S3_preprocess_pupil.m 
    in python 

"""

#import libraries define functions

import os
import pandas as pd
from pathlib import Path
import numpy as np
import numpy.ma as ma
import glob
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy import linalg
import skimage

#defining interpolation to improve numpy's built in interp function which does not check for x2=x1
#def interpolate(x,temp,points):
#    yi = np.interp(points, x, temp) #this performs the interpolation
#    same = np.where(points == x)[0]
#    not_same = np.where(temp[same] != yi[same])[0]
#    yi[not_same] = temp[not_same]
#    return(yi)

def hp_filter(X, fs, highpass):
    nyq = fs/2
    order = np.int_(3*np.fix(fs/highpass))
    length= order+1
    kernel= scipy.signal.firwin(length, highpass/nyq, pass_zero='highpass')
    print(len(kernel))
    filtered = scipy.signal.filtfilt(kernel,1, X, padlen= 3*(order), method='pad', axis=0)
    
    return filtered

#accessing files
"""


homedir = Path('C:/DATA/Pupil_code/')
rawdir = homedir /  'data/converted' #this is where the raw data (EEGLAB format) gets read from
wrtdir = homedir / 'data/processed' #this is where the processed data are stored
funqdir = homedir / 'functions' #folder that contains a filter function

print(rawdir)

if wrtdir.exists()== False:
    os.mkdir('wrtdir')
    
os.chdir(rawdir) #cd
filz = glob.glob(".mat") #each participant has its separate file,glob.glob creates list of files ending with .mat
#defining conditionals for features of data processing

makefig=1 #want figures for interpolation algo for each file? 1 yes 0 nah
dc=1 #want to use deconvolution to filter time locked data around start and end 
#of each blink(mechanical adjustment to light after blink) ? 1 yes 0 nah
"""



rawfile = pd.read_csv('/Users/saanchi/Desktop/rudycode/4saanchi/data/raw/P1S2B1_raw.csv',delimiter=',', skiprows=[1])
data = rawfile.to_numpy()[:,[1,2,3]] 
data= np.transpose(data)



#variables for interpolation

ninterps = 100 #the number of iterations for the interpolation algorithm
delta = 25  #the slope (in pixels) that we consider to be artifactually large 
#both the above variables would need to be changed if the sample rate is
# 1000 Hz, or if the units of the pupil aren't in pixels. Also, it's
#best to tailor this for each participant so that all artifacts are
#accuratlely identified. 

#variables for deconv
dc=1
ndc = 1000 #num of samples around the start/end of blinks to remove with deconvolution
fs= 1000

#start looping

"""

outputfile=[] #to store processed data

for i,j in enumerate(filz): #i is index, j is value(of items in the list as we index over each item)
   
    os.chdir(rawdir)
    newfile= filz[i][:-8] + '.mat'
    outputfile.append(newfile) #assign name
    
    if newfile.exists():  #check if it already exists
        print('skipping file:' + filz[i])
    else:
        print('working on file:' + filz[i])
    
    #scipy.io.loadmat(filz[i])   LOAD DATA
    
    if makefig==1: #plot
          fig, ax = plt.subplots(1,3)
          x,y = newfile.data()
          for k in range(3):
                if k==1:
                  ax[0][k].plot(x[:], y[k])
                  ax[0][k].set_title( newfile + ':diameter')
                  ax[0][k].set_ylabel('Diameter (pixels)')
                elif k==2:                   
                  ax[0][k].plot(x[:], y[k])
                  ax[0][k].set_title( newfile + ':x gaze')
                  ax[0][k].set_ylabel('Gaze x-position (pixels)')
                elif k==3:                   
                  ax[0][k].plot(x[:], y[k])
                  ax[0][k].set_title( newfile + ':diameter')
                  ax[0][k].set_ylabel('Gaze y-position (pixels)')  
          fig.tight_layout()
          plt.show()
"""
"""
 find sections of bad data in diameter
    
    first,find bad data sections by slope and set them to zero. make
    multiple passes through the data to find peaks that occur over
    multiple time-points. note that 'ninterps' assumes a sampling rate
    of 1000Hz, modify if the sampling rate differs (e.g. at 60 Hz, the
    equivalent would be 6 iterations).
    
"""


#set= np.array([[1,0,0,1],[1,1,0,0],[0,1,1,0]])
y = data[0, :]

fig = plt.figure(figsize=[15,2])
ax = plt.subplot(111)
#ax.grid('on')


ax.margins(x=0)

plt.plot(y, c= 'darkslateblue', lw=1)

#ax.yaxis.grid(True, which="both", color="#cccccc", alpha=0.8, lw=0.5)

#plt.fill_between(x, cpp  - secpp, cpp  + secpp ,color= "darkslateblue", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
    ax.spines[spine].set_color('grey')

plt.ylim(0,10000)
plt.show()

points = []
for iter in range(ninterps):
    
    
    # for pointi in range(1, np.size(y)):
        
    #     if (y[pointi] - y[pointi - 1]) > delta:      #check slope/artifactual derivative : downward artefact is indicative of blink
    #         y[pointi-1]=0    
    #         points.append(pointi-1)
        
    # points= np.where(y==0)[0]        
            

    diff= np.ediff1d(y)
    points= np.where((diff)>delta)[0]
    y[points]=0
    points= np.where(y==0)[0]
   

    
    
    if points.size == 0:        #set the subsequent and preceding points to zero to get rid of peaky data
        pointalt = np.where(min(y))[0]
        points = np.array(pointalt)
        y[points]==0 


        
    if points[-1] == (y.size - 1) :   #remove the first and last point of the 'bad section' variable in case it's bad data
        points = points[:-1]
    if points[0] == 0:
        points = points[1:]
        

    y[points + 1] = 0
    y[points - 1] = 0



    for pointi in range(1, np.size(points)):     #set any single point flanked by bad data to zero
            if y[pointi - 1] == 0 and  y[pointi + 1] == 0:
                y[pointi] = 0
                            

points = np.where(y == 0)[0]         #in case the very end of the recording is bad data, we cannot

print(np.isnan(data).any())
print(data.sum(1))

if y[-1] == 0:   
    avgofnonzeros= np.true_divide(data.sum(1),(data!=0).sum(1))                      #interpolate across it, so we manually set it to the mean of the recording
    for k in range(3):
        
       data[k, -ninterps:] = avgofnonzeros[k]
       y[-1] = np.true_divide(y.sum(),(y!=0).sum())
       points = np.delete(points, -1) 
                    
"""


 interpolate sections of bad data: blinks
 
"""
 
temp = y

temp = np.delete(temp, points)

print(len(points))
interpfirstsample = 0  
if len(points)>0 and points[0]== 0:
    temp = np.concatenate((np.zeros(ninterps)+ np.mean(temp), temp), axis=None)
    points = np.delete(points, range(ninterps))
    interpfirstsample = 1 
    
x = np.arange(0, np.size(y))
x= np.delete(x, points)
print(len(x))
print(len(temp))
yi = np.interp(points, x, temp) #this performs the interpolation
same = np.where(np.in1d(points,x))[0]
sameidx=np.where(same<len(temp))[0]
same=same[sameidx]
not_same = np.where(temp[same] != yi[same])[0]
yi[not_same] = temp[not_same]
y[points] = yi

if interpfirstsample == 1:  #if the first data point needs to be interpolated, we replace it with
   y[:ninterps] = np.mean(y[ninterps:]) #the mean of the rest of the data and then interpolate
   
baddata = np.zeros(np.size(y))
baddata[points] = 1

data[0,:] = y  


#for gazex

y = data[1,:]


if interpfirstsample == 1:
    y[:ninterps] = np.mean(y[ninterps:])
     
y[np.where(baddata == 1)[0]] = 0
points = np.where(y == 0)[0]    
temp = y
temp = np.delete(temp, points)
x = np.arange(1, np.size(y)+1)
x= np.delete(x, points)
yi = np.interp(points, x, temp) #this performs the interpolation
same = np.where(np.in1d(points,x))[0]
sameidx=np.where(same<len(temp))[0]
same=same[sameidx]
not_same = np.where(temp[same] != yi[same])[0]
yi[not_same] = temp[not_same]
y[points] = yi   
data[1,:] = y    

#for gazey 

y = data[2,:]


if interpfirstsample == 1:
    y[:ninterps] = np.mean(y[ninterps:])
     
y[np.where(baddata == 1)[0]] = 0
points = np.where(y == 0)[0]   
temp = y
temp = np.delete(temp, points)
x = np.arange(1, np.size(y)+1)
x= np.delete(x, points)
yi = np.interp(points, x, temp) #this performs the interpolation
same = np.where(np.in1d(points,x))[0]
sameidx=np.where(same<len(temp))[0]
same=same[sameidx]
not_same = np.where(temp[same] != yi[same])[0]
yi[not_same] = temp[not_same]
y[points] = yi
data[2,:] = y



# fig, ax = plt.subplots(3,1)
# for k in range(3):
#       if k==0:
#         ax[k].plot(data[k,:])
#         ax[k].set_title('diameter')
#       elif k==1:                   
#         ax[k].plot(data[k,:])
#         ax[k].set_title('Gaze x-position (pixels)')
#       elif k==2:                   
#         ax[k].plot(data[k,:])
#         ax[k].set_title('Gaze y-position (pixels)')  
# fig.tight_layout()
# plt.show()

#comparing rudy and my results

# rudydata =np.transpose(np.loadtxt('/Users/saanchi/Desktop/rudycode/4saanchi/data/processed/P1S2B1_it.csv',delimiter=',',skiprows=1, usecols=(1,2,3))) 

# diamcorr= np.corrcoef(rudydata[0,:], data[0, :])
# gazexcorr=  np.corrcoef(rudydata[1,:], data[1, :])
# gazeycorr= np.corrcoef(rudydata[2,:], data[2, :])



"""
now deconvolution
"""



if dc==1:
    startpoints = np.zeros(np.size(baddata))
    endpoints   = np.zeros(np.size(baddata))
        
    #find all individual sections of interpolated data
    badsecs = skimage.measure.label(baddata)
    #plt.plot(badsecs)
    #plt.show()
    
      
     #loop over sections, and find the start and end points
    for si in range(1, np.max(badsecs)+1):
             sidx =(np.where((badsecs==si))[0])[0] - (ndc-1) #index of the start of a bad section, minus the number of points to remove
             eidx = ((np.where(badsecs==si))[0])[-1]   #index of the end of a bad section
            
             if sidx >= 0:
                 startpoints[sidx] = 1
                
             if eidx <= (np.size(baddata)-1):
                 endpoints[eidx] = 1
                 
    
                
    #now make design matrix
                
    print(1)
    

    s = startpoints #stick function
    m =np.size(s)  #length of event sequence
    n = ndc #length of data section to deconvolve            
    X = np.zeros((m,n)) #the design matrix
    temp = s
    for i in range(n):
        X[:,i] = temp
        temp = np.concatenate(([0], temp[:-1]) , axis=0)
    
    s = endpoints #stick function
    m =np.size(s)  #length of event sequence
    n = ndc #length of data section to deconvolve            
    Y = np.zeros((m,n)) #the design matrix
    temp = s
    for i in range(n):
        Y[:,i] = temp
        temp = np.concatenate(([0], temp[:-1]) , axis=0)
    
    XY = np.concatenate((X,Y), axis = 1) #this is the design matrix  
    #np.savetxt('/Users/saanchi/Desktop/p5s2b2xy.txt', XY, delimiter=',')
    #AB= scipy.linalg.pinv(XY)
    
    print(2)
    column= np.where(~XY.any(axis=0))[0]
    print(column)
    row = np.where(~XY.any(axis=1))[0]
    print(row)
    print(len(row))
    
    if len(row)>0:
        XY = np.transpose(XY)
        PXY= np.linalg.pinv(XY)   
        PXY = np.transpose(PXY)
        XY= np.transpose(XY)
    else:
        PXY= np.linalg.pinv(XY)    
   
    for di in range(3):
        y = data[di,:]
        if di == 0:
            y = hp_filter(y, fs, 0.5)
        y = signal.detrend(y)
        y = y-np.mean(y) #mean center
        es= np.matmul(np.transpose(PXY),np.matmul(np.transpose(XY), y ))
        data[di,:] = data[di,:] - (es)
        print(len(data[di,:]))
    
    print(3)
    # rudydata =(np.loadtxt('/Users/saanchi/Desktop/rudycode/4saanchi/data/processed/dc.csv', delimiter=','))
    # diamcorr= np.corrcoef(rudydata[0,:], data[0, :])
    # gazexcorr=  np.corrcoef(rudydata[1,:], data[1, :])
    # gazeycorr= np.corrcoef(rudydata[2,:], data[2, :])
    
    for ci in range(3):
        y = data[ci,:]
        y[np.where(baddata == 1)[0]] = 0
        points = np.where(y == 0)[0]   
        temp = y
        temp = np.delete(temp, points)
        x = np.arange(1, np.size(y)+1)
        x= np.delete(x, points)
        yi = np.interp(points, x, temp) #this performs the interpolation
        same = np.where(np.in1d(points,x))[0]
        sameidx=np.where(same<len(temp))[0]
        same=same[sameidx]
        not_same = np.where(temp[same] != yi[same])[0]
        yi[not_same] = temp[not_same]
        y[points] = yi
        data[ci,:] = y

fig = plt.figure(figsize=[15,2])
ax = plt.subplot(111)
#ax.grid('on')


ax.margins(x=0)

plt.plot(data[0,:], c= 'darkslateblue', lw=1)

#ax.yaxis.grid(True, which="both", color="#cccccc", alpha=0.8, lw=0.5)

#plt.fill_between(x, cpp  - secpp, cpp  + secpp ,color= "darkslateblue", alpha=0.3)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
    ax.spines[spine].set_color('grey')

plt.ylim(0,10000)

plt.show()

"""
dcitdata = np.transpose(data)
rawfile.iloc[:, [1,2,3]]= dcitdata
rawfile['time']=rawfile['time'].div(fs)
rawfile['baddata']= badsecs.tolist()
rawfile.to_csv('/Users/saanchi/Servers/mountpoint1/stressdata/processed/P24S2B1R_processed.csv', index=False)
"""

