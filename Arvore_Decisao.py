# Importando bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as grafico
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Lendo arquivos CSV
data1 = pd.read_csv('BankChurners.csv')
data2 = pd.read_csv('Salary_Data.csv', sep=';', names=['Years_Exp', 'Customer_Age', 'Salary'], header=0)

# Realizando merge para junção dos arquivos
DtFrame_Final = pd.merge(data1, data2, how='left')

# Tratando valores nulos com fillna
DtFrame_Final.fillna(DtFrame_Final['Salary'].mean(),inplace=True)

# Excluindo colunas
DtFrame_Final = DtFrame_Final.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'])
DtFrame_Final = DtFrame_Final.drop(columns=['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])

# Separando o campo Attrition_Flag dos demais campos
X = DtFrame_Final.iloc[:,2:].values
y= DtFrame_Final.iloc[:,1].values

# Transformando os atributos categoricos
label_encoder_gender = LabelEncoder()
label_encoder_Education_Level = LabelEncoder()
label_encoder_Marital_Status = LabelEncoder()
label_encoder_Income_Category = LabelEncoder()
label_encoder_Card_Category = LabelEncoder()

X[:,1] = label_encoder_gender.fit_transform(X[:,1])
X[:,3] = label_encoder_Education_Level.fit_transform(X[:,3])
X[:,4] = label_encoder_Marital_Status.fit_transform(X[:,4])
X[:,5] = label_encoder_Income_Category.fit_transform(X[:,5])
X[:,6] = label_encoder_Card_Category.fit_transform(X[:,6])

# Separando 70% da base para treinamento
p = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = p, random_state = 42)

# Treinando o modelo de arvore de decisão:
arv_Dec = DecisionTreeClassifier(max_depth=100)
arv_Dec = arv_Dec.fit(X_train,y_train)

# Resultados
resultado = arv_Dec.predict(X_test)
score = accuracy_score(resultado, y_test)
print(metrics.classification_report(y_test,resultado))
