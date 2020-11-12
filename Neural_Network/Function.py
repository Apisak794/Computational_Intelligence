import numpy as np

def create(file):
  f = open(file,'r').read().splitlines()
  line = f[2:] #remove 2 rows up
  data = []
  for i in line:
    data.append(i.split('\t'))
  data = np.array(list(map(lambda sl: list(map(float, sl)), data)))
  return data


def Normalization(data):
  M = np.max(data)
  m = np.min(data)
  data = (data - m) / (M - m)
  return data.tolist()


def crossValidation(data, random): 
  data = data.copy()
  train, test = [],[]
  rd = np.random.randint(1,11)
  while rd in random:
    rd = np.random.randint(1,11)
  random.append(rd)
  l = round(len(data)/10)
  a = l * rd
  test = data[(a - l):a]
  del data[(a - l):a]
  train = data
  return train, test
    

def split(data):
  inputs = []
  output = []

  for i in data:
    inputs.append(i[:8])
    output.append(i[8])

  return np.array(inputs), np.array(output)  


def randomTrain(data):
  y = data.copy()
  random = []
  train = []
  t = []
  l = int(len(y)/10)

  for i in range(10):
    rd = np.random.randint(0,10)
    while rd in random:
      rd = np.random.randint(0,10)
    random.append(rd)
  for i in random:
    t.append(y[rd*l:(rd*l)+l])

  for i in range(len(t)):
    for j in t[i]:
      train.append(j)
  return train


def sumSquareError(e):
  err = 0
  for i in e:
    err = err + pow(i,2)
  return err/2

