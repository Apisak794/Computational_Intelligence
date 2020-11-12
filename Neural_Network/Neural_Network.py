import numpy as np

class Neural_Network():
    def __init__(self):
        # กำหนดโหนดแต่ละเลเยอร์ [input,hidden,output]
        self.inputLayerSize = 8
        self.outputLayerSize = 1
        self.hiddenLayer = int(input("Hidden Layer = "))
        Layers = []
        Layers.append(self.inputLayerSize)
        HLayer = 0
        while HLayer < self.hiddenLayer:
            Layers.append(int(input("Hidden Layer " + str(HLayer + 1) + " Size = ")))
            HLayer += 1
        Layers.append(self.outputLayerSize)
        # กำหนดค่า learningRate และ momentumRate
        self.learningRate = float(input("Learning Rate = "))
        self.momentumRate = float(input("Momentum Rate = "))
        self.deltaW, self.deltaB = [], []
        self.wP, self.wbP = 0, 0
        # ให้ค่า Weight และ bias
        self.weight  = []
        Layer1, Layer2 = Layers.copy(), Layers.copy()
        Layer1.pop() # เอา outputLayer ออก
        Layer2.pop(0) # เอา inputLayer ออก
        for i in range(len(Layer1)):
            w = 2 * np.random.rand(Layer1[i], Layer2[i]) -1
            self.weight.append(w)
        self.biass(Layer2)


    def biass(self, n):
        self.weightBias = []
        for i in n: 
            self.weightBias.append(2 * np.random.rand(i) - 1)


    def sigmoid(self, s): 
        return 1/(1+np.exp(-s))


    def error(self, d, y):
        e = d - y[-1] 
        return e


    def forward(self, x):
        yHat = []
        for i in range(self.hiddenLayer + 1):
            z2 = np.dot(x,self.weight[i]) 
            x = z2 + self.weightBias[i]
            x = self.sigmoid(x) 
            yHat.append(np.reshape(x, (len(x), 1)))
        return yHat


    def diffSigmoid(self, s): 
        return (s) * (1 - (s))
    
    
    def backward(self, x, e):
        gradient, hidOut = self.gradients(e)
        self.deltaW, self.deltaB = self.delta(gradient, hidOut, x)
        if self.wP == 0 and self.wbP == 0:
            self.wP = self.weight.copy()
            self.wbP = self.weightBias.copy()
        for i in range(self.hiddenLayer + 1):
            self.weight[i] = self.weight[i] + self.momentumRate * (self.weight[i] - self.wP[i]) + self.deltaW[i]
            self.weightBias[i] = self.weightBias[i] + self.momentumRate * (self.weightBias[i] - self.wbP[i]) + self.deltaB[i]
        self.wP = self.weight.copy()
        self.wbP = self.weightBias.copy()


    def gradients(self, e):
        grad = []
        out = self.diffSigmoid(self.y[-1]) * e
        self.weight1 = np.flip(self.weight).copy() 
        self.y1 = np.flip(self.y).copy() 
        hidOut = self.y1[1:] 
        hidWeight = self.weight1[:-1] 
        grad.append(out)
        for i in range(self.hiddenLayer):
            grad.append((np.dot(grad[i], np.transpose(hidWeight[i]))) * np.transpose(self.diffSigmoid(hidOut[i])))
        return grad, hidOut


    def delta(self, g, hidOut, x):
        dws, dbs, gd = [], [], []
        gd = g[:-1]
        self.weightBias1 = np.flip(self.weightBias).copy()
        for i in range(self.hiddenLayer):
            dw = -(self.learningRate) * gd[i] * hidOut[i]
            dws.insert(0, dw)
        dw = -(self.learningRate) * g[-1] * np.reshape(x, (len(x), 1))
        dws.insert(0,dw)
        for j in range(self.hiddenLayer + 1):
            db = (-(self.learningRate) * g[j][0] * self.weightBias1[j])
            dbs.insert(0, db)
        return dws, dbs


    def train(self, x, d):
        self.y = []
        self.y = self.forward(x)
        er = self.error(d, self.y) 
        self.backward(x, er)
        return er

        
    def test(self, x, d):
        y = []
        y = self.forward(x)
        er = self.error(d, y)
        return er


    


    









