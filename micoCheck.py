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

#non andiamo a ragruppare nessuna gategoria visto che la maggior parte hanno hanno un bel po' di elementi

#ora andiamo a valutare la simmestria dei valori numerici delle colonne numeriche
print("La simmetria delle colonne:")
print (dataset_cleaned[numerical_columns].skew())

#visto che sono tutti maggiori di 1 andiamo ad usare la mediana
medians= dataset_cleaned[numerical_columns].median()

dataset_cleaned[numerical_columns]= dataset_cleaned[numerical_columns].fillna(medians)

#ora andiamo ad imputare i dati categorici andando a riempire gli spazi vuoti  con la categoria "unhnown"
dataset_cleaned = dataset_cleaned.fillna("Unknown")
dataset_cleaned= dataset_cleaned.drop_duplicates()

