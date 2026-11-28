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
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix, ConfusionMatrixDisplay

def objective(trial):
    with mlflow.start_run(nested=True, run_name="trial_" + str(trial.number)) as child_run:

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

        params = {"C": C, "solver": solver, "penalty": "l2", "class_weight": class_weight, "max_iter": 1000}
        mlflow.log_params(params)

        metriques = {"accuracy": accuracy_score(y_val, pred), "precision": precision_score(y_val, pred),
                     "recall": recall_score(y_val, pred), "f1_score": f1_score(y_val, pred),
                     "roc_auc": roc_auc_score(y_val, pred1[:, 1])}
        mlflow.log_metrics(metriques)

        mlflow.sklearn.log_model(p1, name="model")

        fpr0, tpr0, th0 = roc_curve(y_val==0, pred1[:, 0])
        fpr1, tpr1, th1 = roc_curve(y_val==1, pred1[:, 1])
        plt.figure()
        plt.plot(fpr0, tpr0, label="class0")
        plt.plot(fpr1, tpr1, label="class1")
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel("Proportion mal classée")
        plt.ylabel("Proportion bien classée")
        plt.title('ROC(s) des différentes classes')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")
        plt.close()
        mlflow.log_artifact("roc_curve.png")

        confusion_Matrix = confusion_matrix(y_val, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_Matrix, display_labels = ['non_churn', 'churn'])
        disp.plot()
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")

        return f1_score(y_val, pred)

#Question 4
with mlflow.start_run(run_name="study_LogisticRegression") as run:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

print(study.best_params)
print(study.best_value)