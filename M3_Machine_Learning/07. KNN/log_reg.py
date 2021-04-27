import numpy as np
# 2 points
def sigmoid(scores):
    y = (1 + np.exp(-scores))
    #print(y)
    x = 1 / y
    return x

    
def log_likelihood(features, target, weights):

    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return 

def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
 
        
        scores = np.dot(features, weights)
        #print(scores)
        predictions = sigmoid(scores)


        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        

        if step % 10000 == 0:
            print(log_likelihood(features, target, weights))
        
    return weights