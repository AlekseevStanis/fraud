#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%

path = r'C:\Users\StasAndLiza\port\classification\data\raw\creditcard.csv'
df=pd.read_csv(path)
df
# %%
df.corr()
# %%
plt.figure(figsize=(20,20))
sns.heatmap(data=df.corr(), annot=True, fmt='.3f')
plt.show()
# %%
df.describe()
# %%
from scipy.stats import shapiro
for col in df.columns:
    stat, p = shapiro(df[col])
    if p>0.05:
        print(f"Распределение для колонки {col} нормально со значением  p={p}")
    else:
        print(f"Распределение для колонки {col} не нормально со значением  p={p}")
# %%
from scipy.stats import shapiro
df_fraud=df[df['Class']==0]
for col in df.columns:
    stat, p = shapiro(df_fraud[col])
    if p>0.05:
        print(f"Распределение для колонки {col} нормально со значением  p={p}")
    else:
        print(f"Распределение для колонки {col} не нормально со значением  p={p}")
# %%
for col in df.columns:
    if col != 'Class':
        plt.figure(figsize=(10, 6))
        
        # Plot class 0
        sns.histplot(df[df['Class'] != 1][col], color='gray', label='Other Classes', kde=True, stat="density", alpha=0.5)
        
        # Plot class 1
        sns.histplot(df[df['Class'] == 1][col], color='green', label='Class 1', kde=True, stat="density", alpha=0.7)
        
        plt.title(f"Distribution of {col} - Class 1 vs Others")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.show()

# %%

for col in df.columns:
    if col != 'Class':
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Class', y=col)

        # Calculate 1st and 99th percentiles
        q1 = df[col].quantile(0.001)
        q99 = df[col].quantile(0.999)

        # Add vertical lines
        plt.axhline(q1, color='red', linestyle='--', label='1st Percentile')
        plt.axhline(q99, color='green', linestyle='--', label='99th Percentile')

        plt.title(f"Boxplot of {col} by Class")
        plt.xlabel("Class")
        plt.ylabel(col)
        plt.legend()
        plt.show()


# %% Mutial information
from sklearn.feature_selection import mutual_info_classif
X=df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
y=df[['Class']]
mi_scores = mutual_info_classif(X, y, discrete_features=False)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
mi_scores.plot(kind='bar', figsize=(15,5), title="Mutual Information")

# %% Time analysis
df['hour_of_day'] = (df['Time'] % 86400) // 3600  # 86400 сек в сутках
fraud_by_hour = df.groupby('hour_of_day')['Class'].mean()
fraud_by_hour.plot(title="Доля фрода по часам", figsize=(12,4))



# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

scores = {}
for col in X.columns:
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X[[col]], y)
    preds = tree.predict_proba(X[[col]])[:,1]
    scores[col] = roc_auc_score(y, preds)

pd.Series(scores).sort_values(ascending=False).plot(kind='barh', figsize=(8,10))

 # %%


# %%
