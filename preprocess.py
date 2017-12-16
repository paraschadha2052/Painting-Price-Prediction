from six.moves import cPickle as pickle
import os
import glob
import csv
import cv2
from numpy import array, asarray, ndarray

con_rate = {'USD': 1, 'EUR': 1.17, 'GBP': 1.32, 'NZD': 0.68, '': 1, 'AUD': 0.75, 'HKD': 0.13, 'CHF': 1.01, 'GRD': 0.0034, 'UAH': 0.038, 'BRL': 0.31, 'SEK': 0.12, 'CAD': 0.78, 'NOK': 0.12, 'HRK': 0.16, 'YUM': 0.6, 'TWD': 0.033, 'MXN': 0.053, 'ESP': 0.01, 'ZAR': 0.071, 'SGD': 0.74, 'JPY': 0.0089, 'PLN': 0.28, 'DKK': 0.16, 'CNY': 0.15, 'VEB': 0.0001, 'PHP': 0.02}

training_size = 0.7

img_rows, img_cols = 32, 32

x_test = []
x_train = []
y_test= []
y_train= []
dct = {}

dataset = csv.reader(open("final_input_with_image_urls.csv",encoding="utf8",errors='replace'), delimiter=",")

next(dataset)

lst = list(dataset)

flist=glob.glob('images/*.jpg')

for row in lst:
    if row[-1]:
        year, month, name = row[-1].split('_')
        price = row[-7]
        dct[name] = int(price)*con_rate[row[-6]]

length=int(len(flist)*training_size)

i=0
for filename in flist:
    name=filename.split('/')[-1]
    img = cv2.imread(filename)
    resized = cv2.resize(img, (img_cols, img_rows))
    print("Processed image ... ", i+1)
    
    if(i<length):
        x_train.append(array(resized))
        y_train.append(dct[name])
    else:
        x_test.append(array(resized))
        y_test.append(dct[name])
    i+=1

data_root = '.'
pickle_file = os.path.join(data_root, 'data'+str(img_rows)+'x'+str(img_cols)+'.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test,
    'y_test': y_test,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
