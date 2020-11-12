import Function as fn
import Neural_Network as NN
import numpy as np
import matplotlib.pyplot as plt

file = 'Flood_data_set.txt'
errortrain, errortest = [], []
errortn, errorts, avrtrain = [], [], []
random = []
epochs = int(input("Epoch = "))

data = fn.create(file)
data = fn.Normalization(data)
train, test = fn.crossValidation(data, random)
train = fn.randomTrain(train)
xTest, dTest = fn.split(test)
xTrain, dTrain = fn.split(train)
ep = 0
NN = NN.Neural_Network()
while ep < epochs:
    for i in range(int(epochs/10)): 
        errortin, errortst = [], []
        for j in range(len(xTrain)):
            e = NN.train(xTrain[j], dTrain[j])
            errortin.append(e)
        train = fn.randomTrain(train)
        xTrain, dTrain = fn.split(train)
        ep = ep + 1
        errortrain.append(fn.sumSquareError(errortin))
        avrtrain.append(errortrain)
        # print(avrtrain)
        if ep == epochs:
            break
        

    print('epoch' + str(ep))
    errortn.append(np.average(avrtrain))
    print('Sum Sqaure Error Train[Average] = ',np.average(avrtrain))
    avrtrain = []
    
    for k in range(len(xTest)):
        et = NN.test(xTest[k], dTest[k])
        errortst.append(et)
    errortest.append(fn.sumSquareError(errortst[0][0]))
    errortst.append(errortest[0])
    print('Sum Sqaure Error Test = ',errortest)
    errortrain = []
    errortest = []

    if len(random) == 10:
        random = []
        break
    
    train, test = fn.crossValidation(data, random)
    train = fn.randomTrain(train)
    xTest, dTest = fn.split(test)
    xTrain, dTrain = fn.split(train)


#plot graph
x = [1,2,3,4,5,6,7,8,9,10]
# plt.bar(x, errors, width = 0.8, color = ['lightblue']) 
plt.plot(x, errortn, color='lightblue', linewidth = 3, marker='o', markerfacecolor='dodgerblue', markersize=12, label = "Train")
# plt.plot(x, errors, color='plum', linewidth = 3, marker='o', markerfacecolor='m', markersize=12, label = "Test")
plt.legend()
plt.xlabel('Iteration')  
plt.ylabel('Error')  
plt.show()