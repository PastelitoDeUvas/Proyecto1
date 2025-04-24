import codes
import pandas as pd
import numpy as np  

df = pd.read_csv("train_df.csv")

df.isnull().sum()  # Revisa cuántos valores nulos hay


df.dropna(inplace=True)  # Eliminar filas con nulos

df = pd.get_dummies(df, drop_first=True)







X = df.drop('target', axis=1)
y = df['target']




# Copia del dataframe para no alterar el original
df_estandarizado = df.copy()

# Definimos columnas a excluir de la estandarización
excluir = ['genero', 'historial_diabetes', 'target']

# Seleccionamos columnas numéricas que NO estén en 'excluir'
columnas_a_estandarizar = df.select_dtypes(include=['float64', 'int64']).columns
columnas_a_estandarizar = columnas_a_estandarizar.drop(excluir, errors='ignore')

# Calculamos media y desviación por columna
medias = df_estandarizado[columnas_a_estandarizar].mean()
desviaciones = df_estandarizado[columnas_a_estandarizar].std()

# Estandarizamos solo esas columnas
df_estandarizado[columnas_a_estandarizar] = (
    df_estandarizado[columnas_a_estandarizar] - medias
) / desviaciones

summary = df_estandarizado.describe()
print(summary)