import numpy as np
import sys
sys.path.append('..') # add utils file to path
from sklearn.tree import DecisionTreeRegressor
from utils import *
class GradientBoost:

    def __init__(self, n_trees, max_depth=10, alpha=0.1):

        self.tree_list = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.alpha = alpha
        self.baseline = None


    def fit(self, X, y):

        # Baseline prediction (H0)
        self.baseline = np.mean(y) 
        residual = y - np.mean(y)
        # final_h = h_0 = np.mean(y)
        
        for _ in range(self.n_trees): #TODO figure out wether the baseline counts as a tree
            tree = DecisionTreeRegressor(max_depth=self.max_depth )
            tree.fit(X, residual)    
            #residual -= self.alpha * tree.predict(X) #something diff previous prediction1 and target1?
            print(y.shape)
            print("prediction shape",self.predict(X).shape)
            print("residual shape",residual.shape)
            residual=y-self.predict(X)
            print("shape new residual", residual.shape)
            self.tree_list.append(tree)
            #now we update the new hypotesis: H(X)2=H0+\alpha*(H1(X))
            # final_h += alpha*new_h
            
            
    def predict(self, X): 
        
        prediction = self.baseline #self.baseline*np.ones_like(X)#
        #print("shape base line",prediction.shape)
        for tree in self.tree_list:
           # print("shape tree sklearn",tree.predict(X).shape)
           # print("alpha times predict", (self.alpha * tree.predict(X)).shape)
            prediction +=  self.alpha * tree.predict(X)
           # print("total prediction", prediction.shape)

        return prediction


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    X, y = get_regression_data()
    #print("non squeezed y:", y.shape)
    y = np.squeeze(y)
    #print("squeezed y",y.shape)
    gradientboost=GradientBoost(n_trees=25, max_depth=10, alpha=0.1) #what is the best balance between n_tress and max_depth 
    gradientboost.fit(X, y)     #note Aiz if you change back and forth 100 and 25 for n_trees and max_depth you will realise your best bet to fit is in bigger number of trees.
    predictions = gradientboost.predict(X)

    #visualise_predictions(gradientboost.predict, X)
    plt.plot(X, predictions)
    plt.plot(X, y)
    plt.show()


