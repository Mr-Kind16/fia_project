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

plt.figure(figsize=(18,12))
plt.title("Visualizing Missing Values")
graf=sns.heatmap(dataset.isnull(), cbar=False, cmap=sns.color_palette("Spectral_r", n_colors=13), yticklabels=False)
