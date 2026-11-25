# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 2026

@author: lyssa
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Q1 création du DATAFRAME
df = pd.read_csv(r"C:\Users\lyssa\customer-churn-mlops-\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.info()) #affiche un résumé
print(df.head())


#Q2 Analyse exploratoire : on vérifie s'il manque des données
print(df.isnull().sum())
print(df.select_dtypes(include='str').isnull().sum()) #colonnes str

print(df.astype(str).apply(lambda x: x.str.contains(r'^[^a-zA-Z0-9.]+$')).sum()) # on cherche les caractères spéciaux du type ?*! ou des espaces
print(df.dtypes) #TotalCharges est en object alors que c'est un nombre


#Q3 Traitement de TotalCharges
#11 lignes contiennent " " donc isnull() ne les voit pas
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],
                                   errors='coerce') #convertie en valeur numérique, les " " deviennent NaN
print(df['TotalCharges'].isnull().sum())
print(df[df['TotalCharges'].isnull()]['tenure']) #ces clients ont tenure = 0

#ce sont des nouveaux clients qui n'ont encore rien payé donc on remplace par 0 (pas par la moyenne)
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(df.isnull().sum()) #on vérifie que c'est good


#Q4 Boxplot et matrice de corrélation sur les colonnes numériques
numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
df[numeric_cols].boxplot(figsize=(10,6))
plt.show()

corr_matrix = df[numeric_cols].corr().round(2)
plt.figure(figsize=(8,6))
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm')
plt.show()


#Q5 On enlève customerID (identifiant, n'apporte rien pour prédire)
df = df.drop("customerID", axis=1)


#Q6 Encodage de la variable cible Churn : No -> 0, Yes -> 1
from sklearn.preprocessing import LabelEncoder
df['Churn'] = LabelEncoder().fit_transform(df['Churn'])
print(df['Churn'].value_counts())


#Q7 Encodage des colonnes str avec get_dummies
#on n'utilise pas LabelEncoder ici car il crée un ordre entre les catégories (mauvais pour la Logistic Regression)
object_columns = df.select_dtypes(include='str').columns
print(object_columns)
df = pd.get_dummies(df, columns=object_columns, drop_first=True, dtype=int) #drop_first enlève la colonne en double (ex : Yes/No)
print(df.info()) #on vérifie la structure du new dataframe

#les colonnes "No internet service" sont identiques à InternetService_No (et "No phone service" à PhoneService_Yes)
#colonnes redondantes donc on les enlève
cols_redondantes = [col for col in df.columns if 'No internet service' in col or 'No phone service' in col]
print(cols_redondantes)
df = df.drop(cols_redondantes, axis=1)
print(df.shape)


#Définition des variables X et y
X = df.drop('Churn', axis=1)
y = df['Churn']


#Q8 Découpage train / validation / test (60 / 20 / 20)
#même random_state pour avoir les mêmes données que Clara
#stratify=y pour garder le même taux de churn partout
from sklearn.model_selection import train_test_split
X_temp,X_test,y_temp,y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)

#25 % de ce qui reste = 20 % du total pour la validation (utilisé par Optuna)
X_train,X_val,y_train,y_val = train_test_split(X_temp,y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

#Pas de normalisation ici : on la fait dans le Pipeline avec StandardScaler
#pour éviter la fuite de données (le scaler apprend seulement sur le train)
