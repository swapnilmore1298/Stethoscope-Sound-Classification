import pandas as pd
from api import *
import wave
'''
import numpy as np

import os
from keras.models import load_model 
import math
import scipy.io.wavfile as wf
import scipy.signal

model = load_model('models/model.h5')

from keras.utils import plot_model
plot_model(model, to_file='model.png')
'''
rec_annotations = []
rec_annotations_dict = {}
a = pd.read_csv('sampletext.txt', names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
rec_annotations.append(a)
rec_annotations_dict['sampleaudio'] = a
data = get_sound_samples(rec_annotations_dict['sampleaudio'],'sampleaudio','/', 22000)
cycles_with_labels = [(d[0], d[3], d[4]) for d in data[1:]]
c_only = cycles_with_labels
desired_length=5
test_w = split_and_pad_and_apply_mel_spect(cycles_with_labels, desired_length, 22000)
#print(len(test_w[0][0]))
sample_height = test_w[0][0].shape[0]
print(sample_height)
sample_width = test_w[0][0].shape[1]
ind = 1
plt.figure(figsize = (10,10))
plt.subplot(4,1,1)
plt.imshow(test_w[0][0].reshape(sample_height, sample_width))
plt.savefig('sampleplot.png')
test_gen= feed_all([test_w])
#print(len(test_w))
#print(test_gen.n_available_samples())
test_set=test_gen.generate_keras(24)
'''
test_set1=test_set.__next__()

predictions = model.predict(test_set1[0])
op=[]
for i in predictions:
  for j,e in enumerate(i):
    if(max(i)==e):
      op.append(j)
print(op)
no=0
c=0
w=0
cw=0
sum=0
for e,i in enumerate(predictions):
    no=no+i[0]
    c=c+i[1]
    w=w+i[2]
    cw=cw+i[3]
    sum=sum+i[0]+i[1]+i[2]+i[3]
print(no*100/sum)
print(c*100/sum)
print(w*100/sum)
print(cw*100/sum)'''
print("done")
