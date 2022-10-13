import random
import numpy as np
import seaborn as sns


class dataset:
#     points = 10000
    # init method or constructor
    def __init__(self, points=10000):
        self.points = points
    def get(self,add_noise=False):
        print("Generating dataset")
        X_dataset,y_dataset = [],[]
        for i in range(self.points):
            label_decider = random.randint(0,1)
            sign_decider = random.randint(0,1)
            if(label_decider == 0):
#                 x^2 + y^2 = 1
                  x = round(random.uniform(-1, 1),2)
                  y = round((1 - (x**2))**0.5,2) * ((-1)**sign_decider)
                  X_dataset.append([x,y])
                  y_dataset.append(0)
            else:
#                 x^2 + (y-3)^2 = 1
                  x = round(random.uniform(-1, 1),2)
                  y = 3. + round((1 - (x**2))**0.5,2) * ((-1)**sign_decider)
                  X_dataset.append([x,y])
                  y_dataset.append(1)
        
        X_dataset = np.array(X_dataset)
        y_dataset = np.array(y_dataset)
        
        if(add_noise == True):
            noise = np.random.normal(0,0.1,len(X_dataset)*2).reshape((-1,2))
            X_dataset = X_dataset + noise
            
                
        return X_dataset,y_dataset

    
class PTA:
    def train(self,X,y,bias_learnable = True):
        weights = np.zeros(len(X[0]),dtype=float)
        bias = 0 

        convergence = 100 # max times pta is allowed to run , However i'll terminate it as soon as it stops making changes
        iteration = 1
        while(iteration<=convergence):
            count = 0 # to see if no change is happpening
            for i in range(len(X)): 
                h = np.dot(X[i],weights) + bias
                y_pred = 0
                # signum func implementation
                if(h >= 0):
                    y_pred = 1
                else:
                    y_pred =-1

                # Changing 0->-1 and 1->1
                y_true = y[i]
                if(y[i] == 0):
                    y_true = -1

                error_i =y_true - y_pred

                if(error_i == 0):
                    count+=1
                if(error_i != 0):
                    weights[0] = weights[0] + error_i * X[i][0]
                    weights[1] = weights[1] + error_i * X[i][1]
                    if(bias_learnable == True):
                        bias = bias + error_i
    #                 print(weights,bias)
            if(count==len(X)): # no updates happened 
                print("No updates after",iteration,"iteration")
                break
            iteration+=1
        print("Model Trained")
        return weights,bias
    
    def decisionBoundary(self,X,y,weights,bias):
        sns.set(rc={'figure.figsize':(12,8)})

        trueplot = sns.scatterplot(X[:,0],X[:,-1],hue = y)
        # w0X+w1Y+bias => Y = -(w0/w1)X + -(bias/w1)
        slope = -(weights[0]/weights[1])
        intercept = -(bias/weights[1])

        plt_x = np.array([-1.5,-1,0,1,1.5])
        plt_y = plt_x*slope + intercept
        trueplot = sns.lineplot(x = plt_x,y =plt_y,color='red',label = 'decision boundary')
        trueplot.set(xlabel='X', ylabel='Y', title='Decision Boundary')