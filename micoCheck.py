import warnings

import pandas as pd

from data_cleaning_secondary import data_cleaning

warnings.filterwarnings("ignore")

from sklearn import tree

from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



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

#Ora andiamo a fare il training del classificatore binario nello specifico un decision tree

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


#valutiamo la performance del modello finale
visualize_metrics(final_model, X_test, y_test)

# Convert target variable labels to strings to avoid TypeError in plot_tree
final_model_classes = [str(cls) for cls in final_model.classes_]  # Convert class names to string

# Plot the decision tree
from sklearn.tree import plot_tree

# Get encoded feature names
feature_names = encoder.get_feature_names_out(X_variables.columns)

plt.figure(figsize=(20, 10))  # Adjust figure size as needed
plot_tree(
    final_model,
    feature_names=feature_names,  # Transformed feature names
    class_names=final_model_classes,  # Ensure class names are strings
    max_depth=3,  # Visualize up to depth 3
    filled=True,
    rounded=True
)
plt.show()


#ora che abbiamo accertato che la migliore profondità sia 9 andiamo ad applicare la cross validation per accertarci
#che si abbiano risultati simili per più fold
from sklearn.model_selection import cross_validate

clf=tree.DecisionTreeClassifier(max_depth=grid.best_params_['max_depth'])
scoring=['precision_macro','recall_macro','f1_macro']
scores= cross_validate(clf, X_encoded, y_variable,scoring=scoring, cv=10)
print(scores['test_precision_macro'].mean(),"precision con una deviazione standard:",scores['test_recall_macro'].std())
print(scores['test_recall_macro'].mean(),"recall con una deviazione standard:",scores['test_recall_macro'].std())
print(scores['test_f1_macro'].mean(),"f1 con una deviazione standard:",scores['test_f1_macro'].std())




#prova Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_variable, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)  # Using 100 trees
rf_model.fit(X_train, y_train)

# Step 6: Make predictions using the trained model
y_pred = rf_model.predict(X_test)

print("metriche dell random forest")
# Step 7: Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importances = rf_model.feature_importances_

# Combine feature importances with the feature names
feature_names = encoder.get_feature_names_out(X_variables.columns)  # Get the encoded feature names
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the features by importance (descending)
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the sorted feature importance
print("\nFeature Importances:")
print(importance_df.head(30))

# Optional: Plot the feature importances for better visualization
most_important = importance_df.head(30)  # Keep only the top feature
plt.figure(figsize=(6, 6))
sns.barplot(x=most_important['Importance'], y=most_important['Feature'])
plt.title("Most Important Feature")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()







