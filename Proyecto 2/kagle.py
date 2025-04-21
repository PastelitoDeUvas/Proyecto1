import codes
import pandas as pd
import numpy as np

####Read csv file
train = pd.read_csv("train_df.csv")
test = pd.read_csv("test_df.csv")

# Separar las columnas que no se usarán
excluded_columns = ["paciente_id", "target"]

# Convertir a numérico y eliminar filas con valores faltantes para entrenamiento
X_train = pd.to_numeric(train.drop(columns=excluded_columns).stack(), errors='coerce') \
             .unstack().dropna()

# Alinear los targets con los índices válidos de X_train
y_train = train.loc[train.index.isin(X_train.index), "target"].to_numpy().reshape(-1, 1)

# Procesar el conjunto de prueba
X_test = pd.to_numeric(test.drop(columns=["paciente_id"]).stack(), errors='coerce') \
            .unstack().dropna()




# Normalizar
X_train_normalized = normalize(X_train.values.astype(np.float64))
X_test_normalized = normalize(X_test.values.astype(np.float64))

# Añadir bias
X_train_final = np.hstack([np.ones((X_train_normalized.shape[0], 1)), X_train_normalized])
X_test_final = np.hstack([np.ones((X_test_normalized.shape[0], 1)), X_test_normalized])