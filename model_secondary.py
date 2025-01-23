import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from statsmodels.graphics.mosaicplot import mosaic


import warnings

from data_cleaning_secondary import data_cleaning

warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV
from sklearn import tree

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize_metrics(model, X_test, y_test):
    # Previsione sul set di test
    y_pred = model.predict(X_test)

    # Calcolare le metriche
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Rapporto di classificazione (Precision, Recall, F1-score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Visualizzare la matrice di confusione come una heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()



#andiamo a leggere il dataset dal file csv
dataset= pd.read_csv("secondary_data_shuffled.csv",delimiter=";")
dataset_cleaned= data_cleaning(dataset)

#Ora andiamo a convertire tutte le variabili categoriche in valori numerici andando ad utilizzare One Hot Encoding
from sklearn.preprocessing import OneHotEncoder

# Separiamo le variabili indipendenti da quella che vogliamo predire
X_variables = dataset_cleaned[['cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
       'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-surface', 'stem-color',  'has-ring',
        'habitat', 'season','cap-diameter',
                               'stem-height', 'stem-width']]
y_variable = dataset_cleaned['class']

# Diamo le variabili indipendenti al OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_variables)

print(X_encoded)
#Ora andiamo a fare il training del classificatore binario nello specifico un decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score


# Specifichiamo la proporzione fra i due con test_size; in questo caso abbiamo impostato il training set al 80% e il test set al 10%
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_variable, test_size=0.2, random_state=42)

#ora definiamo gli hyperparameters
parameters = {"max_depth": range(3,10)}
grid= GridSearchCV(tree.DecisionTreeClassifier(),parameters, n_jobs=4, error_score='raise')
grid.fit(X_train, y_train)

#stampiano i migliori hyperparameter
print(grid.best_params_)

#modelliamo un modello usando i migliori hyperparameter
final_model= tree.DecisionTreeClassifier(max_depth=grid.best_params_['max_depth'])
final_model.fit(X_train, y_train)

y_pred= final_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision_score= precision_score(y_test, y_pred)
recall_score= recall_score(y_test, y_pred)
f1_score= f1_score(y_test, y_pred)

#visualizzazione albero
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 800)

tree.plot_tree(final_model)
fig.show()

#valutiamo la performance del modello finale
visualize_metrics(final_model, X_test, y_test)

#ora che abbiamo accertato che la migliore profondità sia 9 andiamo ad applicare la cross validation per accertarci
#che si abbiano risultati simili per più fold
from sklearn.model_selection import cross_validate

clf=tree.DecisionTreeClassifier(max_depth=grid.best_params_['max_depth'])
scoring=['precision_macro','recall_macro','f1_macro']
scores= cross_validate(clf, X_encoded, y_variable,scoring=scoring, cv=10)
print(scores['test_precision_macro'].mean(),"precision con una deviazione standard:",scores['test_recall_macro'].std())
print(scores['test_recall_macro'].mean(),"recall con una deviazione standard:",scores['test_recall_macro'].std())
print(scores['test_f1_macro'].mean(),"f1 con una deviazione standard:",scores['test_f1_macro'].std())

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
#prova Random Forest
rf=RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred= rf.predict(X_test)
accuracyRf = accuracy_score(y_test, y_pred)
precision_scoreRf= precision_score(y_test, y_pred)
recall_scoreRf= recall_score(y_test, y_pred)
f1_scoreRf= f1_score(y_test, y_pred)
print("random forest accuracy",accuracyRf)
print("random forest precision",precision_scoreRf)
print("random forest recall",recall_scoreRf)
print("random forest f1",f1_scoreRf)












