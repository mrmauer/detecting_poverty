import numpy as np 
from sklearn.linear_model import ElasticNetCV
import json

def elastic_net(Xtrain, Ytain, Xdev, Ydev, verbose=False):
    """
    Trains and Elastic Net Linear Model on the provided. Scores the model 
    and returns both the model and the score. It also prints the optimal
    hyperparameters.

    Inputs:
        Xtrain
        Ytrain
        Xdev
        Ydev

    Returns:
        float: the R^2 on the dev data for the best model specifications.
        ElasticNetCV: the trained model.
    """
    print("\n========================\nTraining Elastic Net\n")
    enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
    enet.fit(Xtrain, Ytrain)
    best_score = enet.score(Xdev, Ydev)
    results = {
        "R2" : best_score,
        "alpha" : enet.alpha_,
        "l1_ratio" : enet.l1ratio_
    }
    if verbose:
        results['coefficients'] = enet.coef_.tolist()
    print(json.dumps(results, indent=4))
    return best_score, enet