import numpy as np 
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn import metrics
import json
import utils
# from sklearn.metrics import f1_score, make_scorer


def elastic_net(Xtrain, Ytrain, Xdev, Ydev, verbose=False):
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
    enet = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
        max_iter=1e4,
        tol=1e-2
    )
    enet.fit(Xtrain, Ytrain)
    best_score = enet.score(Xdev, Ydev)
    results = {
        "R2" : best_score,
        "alpha" : enet.alpha_,
        "l1_ratio" : enet.l1_ratio_
    }
    if verbose:
        results['coefficients'] = enet.coef_.tolist()
    print(results, indent=4)
    return best_score, enet

def logistic(Xtrain, Ytrain, Xdev, Ydev, verbose=False, scoring='f1'):
    """
    Trains a Logist Regression Model on the provided data. Scores the model 
    and returns both the model and the score. It also prints the optimal
    hyperparameters. 5-fold cross validation is performed to tune l1 loss ratio
    and C (regularization weight).

    Inputs:
        Xtrain
        Ytrain
        Xdev
        Ydev

    Returns:
        float: the F1 on the dev data for the best model specifications.
        LogisticRegression: the best trained model.
    """
    print("\n========================\nTraining Logistic Regression\n")
    if scoring == 'f1':
        scoring = metrics.make_scorer(metrics.f1_score , average='binary')
    logit = LogisticRegressionCV(
        l1_ratios=[.1, .5, .7, .9, .95, .99, 1],
        Cs=[0.1, 1, 10],
        max_iter=1e4,
        solver='saga',
        scoring=scoring,
        penalty='elasticnet'
    )
    logit.fit(Xtrain, Ytrain)
    best_score = logit.score(Xdev, Ydev)
    Ydev_pred = logit.predict(Xdev)
    num_coeff = len(logit.coef_[logit.coef_ !=0 ])
    results = {
        "F1" : best_score,
        "l1_ratio" : logit.l1_ratio_[0],
        "C" : logit.C_[0],
        "n_nonzero_weights" : num_coeff,
        "accuracy" : metrics.accuracy_score(Ydev, Ydev_pred),
        "precision" : metrics.precision_score(Ydev, Ydev_pred, average='binary'),
        "recall" : metrics.recall_score(Ydev, Ydev_pred, average='binary')
    }
    
    try:
        print(results)
    except Exception as e:
        print(f"Error occured printing results: {e}")
        
    if verbose:
        print(f"There are {num_coeff} non-zero weights in the logistic " +
              "regression model.")
        utils.confusion_matrix(Ydev, Ydev_pred)
        utils.roc_auc(logit, Xdev, Ydev)
        utils.precision_recall(logit, Xdev, Ydev)
    return results, logit

