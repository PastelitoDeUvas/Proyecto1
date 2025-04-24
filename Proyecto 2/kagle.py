import codes
import pandas as pd
import numpy as np  


train = pd.read_csv("C:\\Users\\perao\\Desktop\\Algebra_lineal\\Proyecto1\\Proyecto 2\\train_df.csv")

x,y=codes.cleaning_data(train)
W, b = codes.logistic_regression(x, y, lr=0.1, epochs=1000)

y_pred = codes.predict(x, W, b)
score = codes.f1_score(y, y_pred)

print(f"F1 Score: {score:.4f}")





