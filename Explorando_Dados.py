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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Configurando pandas para exibição de várias colunas
pd.set_option('display.max.columns', None)
pd.set_option('display.max.rows', None)

# Lendo arquivos CSV
data1 = pd.read_csv('BankChurners.csv')
data2 = pd.read_csv('Salary_Data.csv', sep=';', names=['Years_Exp', 'Customer_Age', 'Salary'], header=0)

# Realizando merge para junção dos arquivos
Df = pd.merge(data1, data2, how='left')

# Exibindo base com valores nulos
print(Df.isnull().sum())

# Tratando valores nulos com fillna
Df.fillna(Df['Salary'].mean(),inplace=True)
print(Df.isnull().sum())

# Base após tratamento
Df.info()

# Descrição da base
print(Df.describe())

## Explorando os dados

## Analise por idade

grafico.ylabel('Quantidade')
grafico.xlabel('Idade')
grafico.title("Quantidade de clientes por faixa etária")
graf_hist = grafico.hist(Df['Customer_Age'], bins=15)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]
grafico.ylabel('Quantidade')
grafico.xlabel('Idade')
grafico.title("Quantidade de clientes cancelados por faixa etária")
graf_hist = grafico.hist(filtered_df['Customer_Age'], bins=15)
grafico.show()

## Analise por sexo M e F
grafico.ylabel('Quantidade')
grafico.xlabel('Sexo')
grafico.title("Quantidade de clientes por Sexo")

y = Df.Gender.value_counts().values
x = Df.Gender.value_counts().index

grafico.bar(x,y)
grafico.show()

tb_Pivot = pd.pivot_table(data=Df, values='CLIENTNUM', index='Gender', columns='Attrition_Flag', aggfunc='count')
pz_F = tb_Pivot.loc['F']
pz_M = tb_Pivot.loc['M']

fig, eixos = grafico.subplots(nrows=1, ncols=2)

pz_F = eixos[0].pie(pz_F, labels=['Cancelamento','Cliente Ativo'],autopct='%1.1f%%')
eixos[0].set_title('Feminino')
eixos[0].axis('equal')

pz_M = eixos[1].pie(pz_M, labels=['Cancelamento','Cliente Ativo'],autopct='%1.1f%%')
eixos[1].set_title('Masculino')
grafico.axis('equal')

grafico.subplots_adjust(wspace=1)
grafico.show()

## Analise número de dependentes

y = Df.Dependent_count.value_counts().values
x = Df.Dependent_count.value_counts().index

grafico.ylabel('Quantidade de clientes')
grafico.xlabel('Num de dependentes')
grafico.title("Quantidade de dependentes por cliente")
grafico.bar(x,y)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]

y = filtered_df.Dependent_count.value_counts().values
x = filtered_df.Dependent_count.value_counts().index

grafico.ylabel('Quantidade de clientes')
grafico.xlabel('Num de dependentes')
grafico.title("Quantidade de dependentes por clientes cancelados")
grafico.bar(x,y)
grafico.show()

## Analise por escolaridade

data_df = Df.groupby(['Attrition_Flag', 'Education_Level']).agg( 
    count=('Attrition_Flag', 'count')) 
  
data_df = data_df.reset_index()

sns.barplot(x="Education_Level", y="count", 
            hue="Attrition_Flag", data=data_df, 
            palette='Greens') 
  
grafico.ylabel("Quantidade de clientes", size=14) 
grafico.xlabel("Escolaridade", size=14) 
grafico.title("Clientes Ativos e Inativos por escolaridade", size=18)
grafico.show()

## Analise por estado civil

data_df = Df.groupby(['Attrition_Flag', 'Marital_Status']).agg( 
    count=('Attrition_Flag', 'count')) 
  
data_df = data_df.reset_index()

sns.barplot(x="Marital_Status", y="count", 
            hue="Attrition_Flag", data=data_df, 
            palette='Greens') 
  
grafico.ylabel("Quantidade de clientes", size=14) 
grafico.xlabel("Estado Civil", size=14) 
grafico.title("Clientes Ativos e Inativos por estado civil", size=18)
grafico.show()

## Analise por renda

data_df = Df.groupby(['Attrition_Flag', 'Income_Category']).agg( 
    count=('Attrition_Flag', 'count')) 
  
data_df = data_df.reset_index()

sns.barplot(x="Income_Category", y="count", 
            hue="Attrition_Flag", data=data_df, 
            palette='Greens') 
  
grafico.ylabel("Quantidade de clientes", size=14) 
grafico.xlabel("Renda", size=14) 
grafico.title("Clientes Ativos e Inativos por renda", size=18)
grafico.show()

## Analise por tipo de cartão

data_df = Df.groupby(['Attrition_Flag', 'Card_Category']).agg( 
    count=('Attrition_Flag', 'count')) 
  
data_df = data_df.reset_index()

sns.barplot(x="Card_Category", y="count", 
            hue="Attrition_Flag", data=data_df, 
            palette='Greens') 
  
grafico.ylabel("Quantidade de clientes", size=14) 
grafico.xlabel("Tipo de cartão", size=14) 
grafico.title("Clientes Ativos e Inativos por tipo de cartão", size=18)
grafico.show()

## Analise por meses na carteira

grafico.ylabel('Quantidade')
grafico.xlabel('Meses na carteira')
grafico.title("Quantidade de clientes por meses na carteira")
graf_hist = grafico.hist(Df['Months_on_book'], bins=7)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]
grafico.ylabel('Quantidade')
grafico.xlabel('Meses na carteira')
grafico.title("Quantidade de clientes cancelados por meses na carteira")
graf_hist = grafico.hist(filtered_df['Months_on_book'], bins=7)
grafico.show()

## Analise por número de produtos adquiridos

y = Df.Total_Relationship_Count.value_counts().values
x = Df.Total_Relationship_Count.value_counts().index

grafico.ylabel('Quantidade de clientes')
grafico.xlabel('Quantidade de produtos adquiridos')
grafico.title("Quantidade de clientes por produtos adquiridos")
grafico.bar(x,y)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]

y = filtered_df.Total_Relationship_Count.value_counts().values
x = filtered_df.Total_Relationship_Count.value_counts().index

grafico.ylabel('Quantidade de clientes')
grafico.xlabel('Quantidade de produtos adquiridos')
grafico.title("Quantidade de clientes cancelados por produtos adquiridos")
grafico.bar(x,y)
grafico.show()

## Analise por meses inativos nos ultimos 12 meses

y = Df.Months_Inactive_12_mon.value_counts().values
x = Df.Months_Inactive_12_mon.value_counts().index

grafico.ylabel('Quantidade de clientes')
grafico.xlabel('Meses inativos')
grafico.title("Quantidade de clientes por meses inativos")
grafico.bar(x,y)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]

y = filtered_df.Months_Inactive_12_mon.value_counts().values
x = filtered_df.Months_Inactive_12_mon.value_counts().index

grafico.ylabel('Quantidade de clientes')
grafico.xlabel('Meses inativos')
grafico.title("Quantidade de clientes cancelados por meses inativos")
grafico.bar(x,y)
grafico.show()

## Analise por número de contatos

y = Df.Contacts_Count_12_mon.value_counts().values
x = Df.Contacts_Count_12_mon.value_counts().index

grafico.ylabel('Quantidade de clientes')
grafico.xlabel('Número de contatos')
grafico.title("Quantidade de clientes por contatos")
grafico.bar(x,y)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]

y = filtered_df.Contacts_Count_12_mon.value_counts().values
x = filtered_df.Contacts_Count_12_mon.value_counts().index

grafico.ylabel('Quantidade de clientes')
grafico.xlabel('Número de contatos')
grafico.title("Quantidade de clientes cancelados por contatos")
grafico.bar(x,y)
grafico.show()

## Analise por limite de crédito

grafico.ylabel('Quantidade')
grafico.xlabel('Limite de crédito')
grafico.title("Quantidade de clientes por limite de crédito")
graf_hist = grafico.hist(Df['Credit_Limit'], bins=10)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]
grafico.ylabel('Quantidade')
grafico.xlabel('Limite de crédito')
grafico.title("Quantidade de clientes cancelados por limite de crédito")
graf_hist = grafico.hist(filtered_df['Credit_Limit'], bins=10)
grafico.show()

## Analise por saldo rotativo

grafico.ylabel('Quantidade')
grafico.xlabel('Saldo rotativo')
grafico.title("Quantidade de clientes por saldo rotativo")
graf_hist = grafico.hist(Df['Total_Revolving_Bal'], bins=10)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]
grafico.ylabel('Quantidade')
grafico.xlabel('Saldo rotativo')
grafico.title("Quantidade de clientes cancelados por saldo rotativo")
graf_hist = grafico.hist(filtered_df['Total_Revolving_Bal'], bins=10)
grafico.show()

## Analise por limite disponivel

grafico.ylabel('Quantidade')
grafico.xlabel('Média de limite disponível')
grafico.title("Quantidade de clientes por média de limite disponível")
graf_hist = grafico.hist(Df['Avg_Open_To_Buy'], bins=10)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]
grafico.ylabel('Quantidade')
grafico.xlabel('Média de limite disponível')
grafico.title("Quantidade de clientes cancelados por média de limite disponível")
graf_hist = grafico.hist(filtered_df['Avg_Open_To_Buy'], bins=10)
grafico.show()

## Analise por valor total de transações

grafico.ylabel('Quantidade')
grafico.xlabel('Valor total de transações')
grafico.title("Quantidade de clientes por valor total de transações")
graf_hist = grafico.hist(Df['Total_Trans_Amt'], bins=10)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]
grafico.ylabel('Quantidade')
grafico.xlabel('Valor total de transações')
grafico.title("Quantidade de clientes cancelados por valor total de transações")
graf_hist = grafico.hist(filtered_df['Total_Trans_Amt'], bins=10)
grafico.show()

## Analise por contagem de transações

grafico.ylabel('Quantidade')
grafico.xlabel('Contagem de transações')
grafico.title("Quantidade de clientes por contagem de transações")
graf_hist = grafico.hist(Df['Total_Trans_Ct'], bins=10)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]
grafico.ylabel('Quantidade')
grafico.xlabel('Contagem de transações')
grafico.title("Quantidade de clientes cancelados por contagem de transações")
graf_hist = grafico.hist(filtered_df['Total_Trans_Ct'], bins=10)
grafico.show()

## Analise por tx média de utilização

grafico.ylabel('Quantidade')
grafico.xlabel('Taxa de utilização do cartão')
grafico.title("Quantidade de clientes por taxa de utilização do cartão")
graf_hist = grafico.hist(Df['Avg_Utilization_Ratio'], bins=5)
grafico.show()

df_mask=Df['Attrition_Flag']=='Attrited Customer'
filtered_df = Df[df_mask]
grafico.ylabel('Quantidade')
grafico.xlabel('Taxa de utilização do cartão')
grafico.title("Quantidade de clientes cancelados por taxa de utilização do cartão")
graf_hist = grafico.hist(filtered_df['Avg_Utilization_Ratio'], bins=5)
grafico.show()
