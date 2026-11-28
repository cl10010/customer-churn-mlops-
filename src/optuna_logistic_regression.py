# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 2026

@author: lyssa
"""

import warnings
warnings.filterwarnings("ignore")

#Question 1
from data_prep import X_train, X_val, X_test, y_train, y_val, y_test

#Question 2
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("Telco_Customer_Churn_Optuna")

#Question 3
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def objective(trial):
    C = trial.suggest_float("C", 0.001, 100, log=True)
    solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

    steps = [('scaler',StandardScaler()),('model', LogisticRegression(C=C, solver=solver, penalty='l2', class_weight=class_weight, max_iter=1000))]
    p1 = Pipeline(steps)
    p1.fit(X_train, y_train)
    pred = p1.predict(X_val)
    pred1 = p1.predict_proba(X_val)

    print("accuracy", accuracy_score(y_val, pred))
    print("recall", recall_score(y_val, pred))
    print("precision", precision_score(y_val, pred))
    print("F1", f1_score(y_val, pred))
    print("AUC", roc_auc_score(y_val, pred1[:, 1]))

    return f1_score(y_val, pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3)
print(study.best_params)
print(study.best_value)