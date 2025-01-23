import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
def data_cleaning(dataset):
    print(dataset.head())

    # ora andiamo a vedere la distribuzione di funghi velenosi e non e stampiamo la quantità di valori nulli per ogni colonna
    print("la quantità di funghi velenosi è:", len(dataset[(dataset['class'] == 'p')]))
    print("la quantità di funghi commestibili è:", len(dataset[(dataset['class'] == 'e')]))

    print("il numero totale di righe è:", len(dataset))

    nan_mask = dataset.isna()
    nan_count = nan_mask.sum()
    print(nan_count)

    # visualizzazione dei valori mancanti tramite heatmap
    plt.figure(figsize=(18, 12))
    plt.title("Visualizing Missing Values")
    graf = sns.heatmap(dataset.isnull(), cbar=False, cmap=sns.color_palette("Spectral_r", n_colors=13),
                       yticklabels=False)

    # andiamo ad eliminare le colonne che hanno la maggior parte di valori nulli
    dataset_cleaned = dataset.copy()
    dataset_cleaned = dataset_cleaned.drop('stem-root', axis=1)
    dataset_cleaned = dataset_cleaned.drop('veil-type', axis=1)
    dataset_cleaned = dataset_cleaned.drop('veil-color', axis=1)
    dataset_cleaned = dataset_cleaned.drop('spore-print-color', axis=1)

    # Ora andiamo a isolare vari tipi di colonne da così poter larovare in maniera piú agevole

    target_column = 'class'
    # andiamo a prendere tutte le colonne categoriche
    categorical_columns = dataset_cleaned.select_dtypes(include=['object']).columns.drop(target_column)

    # andiamo a prendere tutte le colonne numeriche
    numerical_columns = dataset_cleaned.select_dtypes(exclude=['object']).columns.drop(target_column, errors='ignore')

    print("target:", target_column)
    print("\ncategorical:", categorical_columns)
    print("\nnumerical:", numerical_columns)

    # ora andiamo a vedere quante categorie uniche ha ogni colonna
    for column in categorical_columns:
        num_unique = dataset_cleaned[column].nunique()
        print(f"'{column}' ha {num_unique} categorie uniche")

    for column in categorical_columns:
        print(f"\nTop value counts in {column}:\n{dataset_cleaned[column].value_counts().head(10)}")

    # non andiamo a ragruppare nessuna gategoria visto che la maggior parte hanno un bel po' di elementi

    # valutazione distribuzione variabile target
    # Calculate counts for the pie chart and add labels
    class_counts = dataset_cleaned['class'].value_counts().sort_index()
    labels = ["Edible", "Poisonous"]

    plt.figure(figsize=(6, 6))
    plt.pie(class_counts, labels=labels,
            autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Classes')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()
    # ora andiamo a valutare la simmestria dei valori numerici delle colonne numeriche
    print("La simmetria delle colonne:")
    print(dataset_cleaned[numerical_columns].skew())

    # visto che sono tutti maggiori di 1 andiamo ad usare la mediana
    medians = dataset_cleaned[numerical_columns].median()

    dataset_cleaned[numerical_columns] = dataset_cleaned[numerical_columns].fillna(medians)

    # ora andiamo ad imputare i dati categorici andando a riempire gli spazi vuoti  con la categoria "unhnown"
    dataset_cleaned = dataset_cleaned.fillna("Unknown")
    dataset_cleaned = dataset_cleaned.drop_duplicates()

    # esplorazione delle feature per capire se ci sono pattern che ci possono aiutare
    plt.figure(figsize=(8, 15))

    for i, column in enumerate(numerical_columns):
        plt.subplot(3, 1, i + 1)
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

    #visto che nella categoria ring-type abbiamo la maggior parte di elementi con
    #categoria f eliminiamoo la colonne
    dataset_cleaned = dataset_cleaned.drop('ring-type', axis=1)


    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.violinplot(dataset_cleaned, x='class', y=column)
        plt.title(f'Distribution of {column} by class')

        plt.tight_layout()
        plt.show()

    # una volta che siamo andati ad analizzare la distribuzione dei dati e abbiamo visto che sono presenti degli
    # out liers nei valori numerici andiamo ad eliminarli

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

    # il valore del threshold lo siamo andati a decidere in modo individuale in base a come esano distribuiti i dati
    for i, column in enumerate(numerical_columns):
        plt.subplot(3, 1, i + 1)
        sns.histplot(data=dataset_cleaned, x=column, kde=True, bins=20)
        plt.title(f'Distribution of {column}')
        sns.despine()

    plt.tight_layout()
    plt.show()

    # ora andiamo a sostituire i valori della variabile target
    dataset_cleaned['class'].replace({'e': 0, 'p': 1}, inplace=True)
    return dataset_cleaned