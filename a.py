from __future__ import print_function
__author__ = 'silva'

import csv
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
import os
import math

def maxLog(dir):
    maxBase = []
    minBase = []
    meanBase = []
    stdBase = []
    maxStart = []
    minStart = []
    meanStart = []
    stdStart = []
    maxEnd = []
    minEnd = []
    meanEnd = []
    stdEnd = []
    maxMaxP = []
    minmaxP = []
    meanMaxP = []
    stdMaxP = []
    maxMinP = []
    minMinP = []
    meanMinP = []
    stdMinP = []
    maxVol = []
    minVol = []
    meanVol = []
    stdVol = []
    for dirName, subdirList, fileList in os.walk(dir):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if not ".csv" in fname:
                continue
            numline = len(file(fname, 'rb').readlines())
            print("%s lines : %d" % (fname, numline))
            reader = csv.reader(file(fname, 'rb'))
            base = [0]#*len(file("mtgoxPLN.csv", 'rb').readlines())
            vol = []
            maxV  = []
            minV = []
            start = []
            end = []
            stamp = []
            i = 0
            isFirst = True

            for l, line in enumerate(reader):
               if l % 1000000 == 0:
                    print("%s, %d line, %d total, %f\n" % (fname, l, numline, float(l)/numline))
               if (int(line[0]) - base[i] < 24*60*60):
                    vol[i] += float(line[2])
                    end[i] = float(line[1])
                    if minV[i] > float(line[1]):
                        minV[i] = float(line[1])
                    if maxV[i] < float(line[1]):
                        maxV[i] = float(line[1])
               else:
                   if isFirst == False:
                       i += 1
                       base.append(int(line[0]))
                   else:
                       isFirst = False
                       base[0] = int(line[0])
                   stamp.append(int(line[0]))
                   start.append(float(line[1]))
                   minV.append(float(line[1]))
                   maxV.append(float(line[1]))
                   end.append(float(line[1]))
                   vol.append(float(line[2]))
            base = [math.log(float(v)) for v in base]
            start = [math.log(float(v)) for v in start]
            end = [math.log(float(v)) for v in end]
            minV = [math.log(float(v)) for v in minV]
            maxV = [math.log(float(v)) for v in maxV]
            vol = [math.log(float(v)) for v in vol]
            for i in range(0, len(base)-2):
                base[i] = base[i+1] - base[i]
            for i in range(0, len(start)-2):
                start[i] = start[i+1] - start[i]
            for i in range(0, len(end)-2):
                end[i] = end[i+1] - end[i]
            for i in range(0, len(minV)-2):
                minV[i] = minV[i+1] - minV[i]
            for i in range(0, len(maxV)-2):
                maxV[i] = maxV[i+1] - maxV[i]
            for i in range(0, len(vol)-2):
                vol[i] = vol[i+1] - vol[i]
            maxBase.append(max(base))
            minBase.append(min(base))
            meanBase.append(sum(base)/len(base))
            #stdBase.append(math.sqrt( sum([math.pow(v - meanBase, 2) for v in base])/(len(base) - 1)))
            maxStart.append(max(start))
            minStart.append(min(start))
            meanStart.append(sum(start)/len(start))
            #stdStart.append(math.sqrt( sum([math.pow(v - meanStart, 2) for v in start])/(len(start) - 1)))
            maxEnd.append(max(end))
            minEnd.append(min(end))
            meanEnd.append(sum(end)/len(end))
            #stdEnd.append(math.sqrt(sum([math.pow(v - meanEnd, 2) for v in end])/(len(end) - 1)))
            maxMaxP.append(max(maxV))
            minmaxP.append(min(minV))
            meanMaxP.append(sum(maxV)/len(maxV))
            maxMinP.append(max(minV))
            minMinP.append(min(minV))
            meanMinP.append(sum(minV)/len(minV))
            maxVol.append(max(vol))
            minVol.append((min(vol)))
            meanVol.append(sum(vol)/len(vol))
    return max(maxBase), min(maxBase), max(minBase), min(minBase),sum(meanBase)/len(meanBase), max(maxStart), min(maxStart), max(minStart), min(minStart), sum(meanStart)/len(meanStart), max(maxEnd), min(maxEnd), \
           max(minEnd), min(minEnd), sum(meanEnd)/len(meanEnd), max(maxMaxP), min(maxMaxP), max(minmaxP), min(minmaxP), sum(meanMaxP)/len(meanMaxP), max(maxMinP), min(maxMinP), max(minMinP), min(minMinP), sum(meanMinP)/len(meanMinP), \
           max(maxVol), min(maxVol), max(minVol), min(minVol), sum(meanVol)/len(meanVol)

maxMaxBase, minMaxBase, maxMinBase, minMinBase, meanBase, maxMaxStart , minMaxStart, maxMinStart, minMinStart, meanStart, maxMaxEnd, minMaxEnd,\
maxMinEnd, minMinEnd, meanMinEnd, maxMaxMaxP, minMaxMaxP, maxMinMaxP,minMinMaxP, meanMaxP, maxMaxMinP, minMaxMinP, maxMinMinP,minMinMinP, meanMinP, maxMaxVol, minMaxVol, maxMinVol, minMinVol, meanVol = maxLog('.')
writer = csv.writer(file('okcoinK.txt', 'wb'))
writer.writerow([maxMaxBase, minMaxBase, maxMinBase, minMinBase, meanBase, maxMaxStart , minMaxStart, maxMinStart, minMinStart, meanStart, maxMaxEnd, minMaxEnd,\
                maxMinEnd, minMinEnd, meanMinEnd, maxMaxMaxP, minMaxMaxP, maxMinMaxP,minMinMaxP, meanMaxP, maxMaxMinP, minMaxMinP, maxMinMinP,minMinMinP, meanMinP, maxMaxVol, minMaxVol, maxMinVol, minMinVol, meanVol])

def stdandizeData(fname):
    reader = csv.reader(file(fname, 'rb'));
    base = [0]#*len(file("mtgoxPLN.csv", 'rb').readlines())
    vol = []
    maxV  = []
    minV = []
    start = []
    end = []
    stamp = []
    i = 0
    isFirst = True
    for line in reader:
       if (int(line[0]) - base[i] < 24*60*60):
            vol[i] += float(line[2])
            end[i] = float(line[1])
            if minV[i] > float(line[1]):
                minV[i] = float(line[1])
            if maxV[i] < float(line[1]):
                maxV[i] = float(line[1])

       else:
           if isFirst == False:
               i += 1
               base.append(int(line[0]))
           else:
               isFirst = False
               base[0] = int(line[0])
           stamp.append(int(line[0]))
           start.append(float(line[1]))
           minV.append(float(line[1]))
           maxV.append(float(line[1]))
           end.append(float(line[1]))
           vol.append(float(line[2]))

    #writer = csv.writer(file('mtgoxPLN1Hour.csv', 'wb'))
    #for i in range(len(base)):
    #    writer.writerow([ time.ctime(base[i]), start[i], end[i], minV[i], maxV[i], vol[i]])

    base = [math.log(float(v)) for v in base]
    start = [math.log(float(v)) for v in start]
    end = [math.log(float(v)) for v in end]
    minV = [math.log(float(v)) for v in minV]
    maxV = [math.log(float(v)) for v in maxV]
    vol = [math.log(float(v)) for v in vol]

    for i in range(0, len(base)-2):
        base[i] = base[i+1] - base[i]
    for i in range(0, len(start)-2):
        start[i] = start[i+1] - start[i]
    for i in range(0, len(end)-2):
        end[i] = end[i+1] - end[i]
    for i in range(0, len(minV)-2):
        minV[i] = minV[i+1] - minV[i]
    for i in range(0, len(maxV)-2):
        maxV[i] = maxV[i+1] - maxV[i]
    for i in range(0, len(vol)-2):
        vol[i] = vol[i+1] - vol[i]

    #base = [(float(v) - min(minBase)) / (max(base) - min(base)) for i, v in enumerate(base)]
    start = [(float(v) - minMinStart) / (maxMaxStart - minMinStart) for i, v in enumerate(start)]
    end = [(float(v) - minMinEnd) / (maxMaxEnd - minMinEnd) for i, v in enumerate(end)]
    minV = [(float(v) - minMinMinP) / (maxMaxMinP - minMinMinP) for i, v in enumerate(minV)]
    maxV = [(float(v) - minMinMaxP) / (maxMaxMaxP - minMinMaxP) for i, v in enumerate(maxV)]
    vol = [(float(v) - minMinVol) / (maxMaxVol - minMinVol) for i, v in enumerate(vol)]

    base = (np.array(base) - meanBase)/np.std(base)
    start = (np.array(start) - meanStart)/np.std(start)
    end = (np.array(end) - meanMinEnd)/np.std(end)
    minV = (np.array(minV) - meanMinP)/np.std(minV)
    maxV = (np.array(maxV) - meanMaxP)/np.std(maxV)
    vol = (np.array(vol) - meanVol)/np.std(vol)

    X = np.zeros((len(base), 1, 5), dtype=np.float)
    y = np.zeros((len(base), 3), dtype=np.float)#minV, max, vol
    for i in range(len(base)):
        X[i][0] = [start[i], end[i], minV[i], maxV[i], vol[i]]
        if i + 1 < len(base):
            y[i] = [ minV[i+1], maxV[i+1], vol[i+1] ]
        else:
            y[i] = [ minV[i], maxV[i], vol[i] ]
    return X, y

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(1,5)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')

X, y = stdandizeData('./zyadoEUR.csv')
model.fit(X[0:(len(X)*0.8)], y[0:(len(X)*0.8)], batch_size=128, nb_epoch=1,
              show_accuracy=True, verbose=1, validation_data=(X[(len(X)*0.8):], y[(len(X)*0.8):]))
model.save_weights("./weights.w")
model.predict([0.1, 0.1, 0.11, 0.11, 0.02], 1)
X, y = stdandizeData('btcnCNY.csv')
model.fit(X[0:(len(base)-100)], y[0:(len(base)-100)], batch_size=128, nb_epoch=1,
              show_accuracy=True, verbose=1, validation_data=(X[(len(base)-100):], y[(len(base)-100):]))




