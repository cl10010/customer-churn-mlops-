# customer-churn-mlops-

TP MLOps : prédiction du désabonnement des clients (dataset Telco Customer Churn) avec GitHub, Optuna et MLflow.

## Récupérer les données nettoyées

Le dataset n'est pas sur le dépôt (dossier `data/` dans le `.gitignore`), chacune le télécharge sur son ordi.

1. Télécharger **Telco Customer Churn** sur Kaggle (fichier `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. Le mettre dans le dossier `data/` du projet.
3. Dans `src/data_prep.py`, ligne 13, **modifier le chemin du `read_csv`** avec le chemin sur son ordi :
   ```python
   df = pd.read_csv(r"C:\Users\<votre_nom>\customer-churn-mlops-\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
   ```
4. Lancer `src/data_prep.py` (F5 dans Spyder) pour vérifier. On doit obtenir :
   ```
   Train : (4225, 23)
   Validation : (1409, 23)
   Test : (1409, 23)
   ```
5. Dans son script Optuna, récupérer les données avec :
   ```python
   from data_prep import X_train, X_val, X_test, y_train, y_val, y_test
   ```

⚠️ Ne pas committer `data_prep.py` après avoir changé le chemin (pas de `git add` dessus), sinon le chemin de l'autre est écrasé.

## Ce que fait `data_prep.py`

- `TotalCharges` : 11 valeurs contiennent un espace `" "` → converties en NaN avec `pd.to_numeric`, puis remplacées par 0 (clients avec `tenure = 0`, ils n'ont encore rien payé)
- Suppression de `customerID`
- `Churn` encodé avec `LabelEncoder` (No → 0, Yes → 1)
- Variables catégorielles encodées avec `get_dummies`
- Suppression des 7 colonnes redondantes `No internet service` / `No phone service`
- Découpage train / validation / test (60 / 20 / 20) avec `stratify=y` et `random_state=42` → mêmes données pour les deux
- Pas de normalisation ici : elle est faite dans le Pipeline (`StandardScaler`) pour éviter la fuite de données
