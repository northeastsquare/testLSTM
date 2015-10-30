import csv

def getConstant(fname):
    reader = csv.reader(file(fname, 'rb'))
    return reader.next()

def standizeData(inputK):
    for i in range(0, len(inputK)-2):
        base[i] = base[i+1] - base[i]
    [maxMaxBase, minMaxBase, maxMinBase, minMinBase, meanBase, maxMaxStart , minMaxStart, maxMinStart, minMinStart, meanStart, maxMaxEnd, minMaxEnd,\
        maxMinEnd, minMinEnd, meanMinEnd, maxMaxMaxP, minMaxMaxP, maxMinMaxP,minMinMaxP, meanMaxP, maxMaxMinP, minMaxMinP, maxMinMinP,minMinMinP, meanMinP, maxMaxVol, minMaxVol, maxMinVol, minMinVol, meanVol] = getConstant('okcoinK.txt')


print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(1,5)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')
model.load_weights("./weights.w")

model.predict([0.1, 0.1, 0.11, 0.11, 0.02], 1)