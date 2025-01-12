import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic


import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV
from sklearn import tree

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import matthews_corrcoef

#andiamo a leggere il dataset dal file csv
dataset= pd.read_csv("secondary_data_shuffled.csv",delimiter=";")
print (dataset.head())

#ora andiamo a vedere la distribuzione di funghi velenosi e non e stampiamo la quantità di valori nulli per ogni colonna
print("la quantità di funghi velenosi è:",len(dataset[(dataset['class'] == 'p')]))
print("la quantità di funghi commestibili è:",len(dataset[(dataset['class'] == 'e')]))

print("il numero totale di righe è:",len(dataset))

nan_mask= dataset.isna()
nan_count= nan_mask.sum()
print(nan_count)

#visualizzazione dei valori mancanti tramite heatmap
plt.figure(figsize=(18,12))
plt.title("Visualizing Missing Values")
graf=sns.heatmap(dataset.isnull(), cbar=False, cmap=sns.color_palette("Spectral_r", n_colors=13), yticklabels=False)

#andiamo ad eliminare le colonne che hanno la maggior parte di valori nulli
dataset_cleaned= dataset.copy()
dataset_cleaned = dataset_cleaned.drop('stem-root', axis=1)
dataset_cleaned = dataset_cleaned.drop('veil-type', axis=1)
dataset_cleaned = dataset_cleaned.drop('veil-color', axis=1)
dataset_cleaned = dataset_cleaned.drop('spore-print-color', axis=1)

#Ora andiamo a isolare vari tipi di colonne da così poter larovare in maniera piú agevole

target_column= 'class'
#andiamo a prendere tutte le colonne categoriche
categorical_columns= dataset_cleaned.select_dtypes(include=['object']).columns.drop(target_column)

#andiamo a prendere tutte le colonne numeriche
numerical_columns= dataset_cleaned.select_dtypes(exclude=['object']).columns.drop(target_column, errors='ignore')

print("target:", target_column)
print("\ncategorical:", categorical_columns)
print("\nnumerical:",numerical_columns)

#ora andiamo a vedere quante categorie uniche ha ogni colonna
for column in categorical_columns:
  num_unique= dataset_cleaned[column].nunique()
  print(f"'{column}' ha {num_unique} categorie uniche")

for column in categorical_columns:
  print(f"\nTop value counts in {column}:\n{dataset_cleaned[column].value_counts().head(10)}")

#non andiamo a ragruppare nessuna gategoria visto che la maggior parte hanno un bel po' di elementi

#ora andiamo a valutare la simmestria dei valori numerici delle colonne numeriche
print("La simmetria delle colonne:")
print (dataset_cleaned[numerical_columns].skew())

#visto che sono tutti maggiori di 1 andiamo ad usare la mediana
medians= dataset_cleaned[numerical_columns].median()

dataset_cleaned[numerical_columns]= dataset_cleaned[numerical_columns].fillna(medians)

#ora andiamo ad imputare i dati categorici andando a riempire gli spazi vuoti  con la categoria "unhnown"
dataset_cleaned = dataset_cleaned.fillna("Unknown")
dataset_cleaned= dataset_cleaned.drop_duplicates()

#esplorazione delle feature per capire se ci sono pattern che ci possono aiutare
plt.figure(figsize=(8, 15))

for i, column in enumerate(numerical_columns):
    plt.subplot(3, 1, i+1)
    sns.histplot(data=dataset_cleaned, x=column, kde=True, bins=20)
    plt.title(f'Distribution of {column}')
    sns.despine()

plt.tight_layout()
plt.show()

for column in categorical_columns:

    filtered_data = dataset_cleaned.loc[dataset_cleaned[column] != 'Unknown']

    plt.figure(figsize=(8, 5))
    sns.countplot(data=filtered_data, x=column)
    plt.title(f'Countplot of {column}')

    plt.tight_layout()
    plt.show()


for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.violinplot(dataset_cleaned, x='class', y=column)
    plt.title(f'Distribution of {column} by class')

    plt.tight_layout()
    plt.show()

#una volta che siamo andati ad analizzare la distribuzione dei dati e abbiamo visto che sono presenti degli
#out liers nei valori numerici andiamo ad eliminarli

def removal_box_plot(df, column, threshold):
  sns.boxplot(df[column])
  plt.title(f'Original Box Plot of {column}')
  plt.show()

  removed_outliers = df[df[column] <= threshold]
  df.drop(df[df[column] >= threshold].index, inplace=True)

  sns.boxplot(removed_outliers[column])
  plt.title(f'Box Plot without Outliers of {column}')
  plt.show()
  return removed_outliers


threshold_value = 15

no_outliers = removal_box_plot(dataset_cleaned, 'cap-diameter', threshold_value)

threshold_value = 11

no_outliers2 = removal_box_plot(dataset_cleaned, 'stem-height', threshold_value)

threshold_value = 28

no_outliers3 = removal_box_plot(dataset_cleaned, 'stem-width', threshold_value)

#il valore del threshold lo siamo andati a decidere in modo individuale in base a come esano distribuiti i dati

#ora andiamo a sostituire i valori della variabile target
dataset_cleaned['class'].replace({'e': 0, 'p': 1}, inplace=True)

#Ora andiamo a convertire tutte le variabili categoriche in valori numerici andando ad utilizzare One Hot Encoding
from sklearn.preprocessing import OneHotEncoder

# Separiamo le variabili indipendenti da quella che vogliamo predire
X_variables = dataset_cleaned[['cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
       'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-surface', 'stem-color',  'has-ring',
       'ring-type', 'habitat', 'season','cap-diameter',
                               'stem-height', 'stem-width']]
y_variable = dataset_cleaned['class']

# Diamo le variabili indipendenti al OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_variables)

print(X_encoded)
#Ora andiamo a fare il training del classificatore binario nello specifico un decision tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# Separiamo i nostri dati in dati di training e dati di test
# Specifichiamo la proporzione fra i due con test_size; in questo caso abbiamo impostato il training set al 90% e il test set al 10%
cross_validation= KFold(n_splits=10, shuffle= True, random_state= 1)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_variable, test_size=0.05, random_state=42)

# Utilizziamo il training set per addestrare il modello
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1_score= f1_score(y_test, y_pred)
recall= recall_score(y_test, y_pred)

print(f"Accuracy del modello sul test set: {accuracy:.2f}")
print(f"Precison del modello sul test set: {precision:.2f}")
print(f"F1 del modello sul test set: {f1_score:.2f}")
print(f"recall del modello sul test set: {recall:.2f}")

#Come vediamo il decision tree ha un accuracy e
# precision di 1 che è troppo alta quindi probabilmente non è adatto come modello di ML per questo dataset
# forse probabilmente quindi andiamo a valutare altri parametri per capire come si comporta il modello

parameters = {"max_depth": range(3,20)}
clf= GridSearchCV(tree.DecisionTreeClassifier(),parameters, n_jobs=4, error_score='raise')
clf.fit(X=X_encoded, y=y_variable)
tree_model= clf.best_estimator_
print (clf.best_score_, clf.best_params_)












