#%%
from sklearn.tree import  DecisionTreeClassifier
import sys
sys.path.append('..') # add utils file to path
from utils import get_classification_data, calc_accuracy, visualise_predictions, show_data
import numpy as np




class AdaBoost:

    def __init__(self, n_models):
        self.n_models = n_models
        self.stumps=[]      #Define our list of 1-level models or stomps
        self.y_hats_list=[]
        self.alphas_list = []     #Define our lists of alphas and models

    def convert_labels(self, labels):
        #print(labels)
        """
        for i in range(len(labels)):
            if labels[i] == 0:
                labels[i]= -1    # convert 0-label into -1
        """
        labels[labels==0]=-1    # convert 0-label into -1
        #print(labels)
        return labels

        
    """
    def final_prediction(self, X):
        #print(sum(np.array(self.alphas_list)*np.array(self.y_hats_list)))
        #print("shapes",np.array(self.alphas_list).shape)
        #print("shapes2:", np.array(self.y_hats_list).shape)
        #y_final = np.sign(sum(np.array(self.y_hats_list).T *np.array(self.alphas_list))) #sum of alpha_i * h_i
        y_final=0
        #print(len(self.alphas_list))
        #print(self.y_hats_list)
        #print(self.alphas_list)
        for i in range(len(self.alphas_list)):
            y_final+=self.alphas_list[i]*self.y_hats_list[i]   #Problem: if you evaluate only a datapoint. the function will still give back an array
        print("model saying and predictions:",list(zip(self.alphas_list,self.y_hats_list)))
        #print("y_final:",np.sign(y_final))
        return np.sign(y_final).astype(int)
    """

    def final_prediction(self, X):
        prediction = np.zeros(len(X))
        for model in self.stumps:
            prediction += model.weight * model.predict(X)
        #print("prediction:",prediction)
        return np.sign(prediction)

    def resample(self,X,Y, weights):
        #resample our X, Y
        #print("weights: ", weights)
        idx = np.random.choice(range(len(Y)), size=len(X), replace=True, p=weights)
        newX=X[idx]
        newY=Y[idx]
        return newX,newY

    def fit(self, X, Y):
        Y=self.convert_labels(Y)
        def error( Y, y_hat, weights):
            # calculate error and get the new weights
            incorrect_labels=np.ones(len(Y))
            print("Y:",Y)
            print("y_hat",y_hat)
            incorrect_labels[Y==y_hat]=0  
            error = np.dot(incorrect_labels, weights)/np.sum(weights)
            print("Error:",error)
            return error 
        

        def saying( error,delta= 0.0000001):
            # calculating alpha = 1/2*log((1-error)/error)
            #error=self.error(labels,y_hat, weights)
            alpha = 0.5*np.log( (1-error) / (error+delta) + delta )
            print("apha:",alpha)
            #alpha2=np.log((1-error)/error)/2
            #print(alpha2)
            return alpha

        def update_weights( alpha,weights,  labels, y_hat):
             # new weights = weights * np.exp(- alpha* labels * y_hat)
             weights = weights * np.exp(- alpha* labels * y_hat)
             weights = weights/np.sum(weights)    #normalise weights so they add up to 1
             #print("weights in update:",weights)
             return weights


        
        n_examples = len(Y)
        #weights = np.ones(n_examples, self.n_models)/n_examples  # initialize start weights equal to 1/ number of training examples
        weights=np.ones(n_examples)/n_examples
        bootstrapped_X,bootstrapped_Y=X,Y
        for model_idx in range(self.n_models):   # repeat N times
           
            model = DecisionTreeClassifier(max_depth = 1)
            model.fit(bootstrapped_X,bootstrapped_Y)
            self.stumps.append(model)
            #print("Old X and Y",X,Y)
            #print("New X and Y:",bootstrapped_X, bootstrapped_Y)
            y_hat = model.predict(X)   # calculate h(x) hypothesis
            #visualise_predictions(model.predict, X)
            #print("prediction each model {}".format(model_idx),y_hat)
            self.y_hats_list.append(y_hat)
            
            error_l = error(Y, y_hat, weights)
            alpha = saying(error_l)
            self.alphas_list.append(alpha)
            model.weight=alpha

            weights=update_weights(alpha,weights,Y,y_hat)
            # calculate h(x) hypothesis
            
            # resample the dataset using these new weights
            bootstrapped_X,  bootstrapped_Y  = self.resample(X,Y,weights)

X, Y = get_classification_data(sd=10,m=50)
adaboost=AdaBoost(n_models=20)
adaboost.fit(X,Y)
print("This is the final prediction:",adaboost.final_prediction(X))
print("This is the original labels:",Y)
print(adaboost.final_prediction(X)==Y)
print("Shape:", adaboost.final_prediction(X).shape)
print("type:", type(adaboost.final_prediction(X)))
visualise_predictions(adaboost.final_prediction, X)
print(f'accuracy: {calc_accuracy(adaboost.final_prediction(X), Y)}')

show_data(X,Y)
print("Evaluate for a point: ", adaboost.final_prediction(np.array([[1,1]])))

# %%
import sklearn.ensemble

adaBoost = sklearn.ensemble.AdaBoostClassifier()
adaBoost.fit(X, Y)
predictions = adaBoost.predict(X)
calc_accuracy(predictions, Y)
#visualise_predictions(adaBoost.predict, X,Y)
#show_data(X, Y)
print("Adaboosts sklearn predictions:",predictions)
print(predictions.shape)
print(type(predictions))
print(f'accuracy: {calc_accuracy(predictions, Y)}')

# %%
