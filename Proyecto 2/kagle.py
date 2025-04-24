import codes
import pandas as pd
import numpy as np  


train = pd.read_csv("C:\\Users\\perao\\Desktop\\Algebra_lineal\\Proyecto1\\Proyecto 2\\train_df.csv")
x,y=codes.cleaning_data(train)
X_train, X_test, y_train, y_test = codes.train_test_split(x, y)


W, b = codes.logistic_regression(X_train, y_train, lr=0.001, epochs=1000)
y_pred_test = codes.predict(X_test, W, b,threshold=0.1)
print("F1 (test):", codes.f1_score(y_test, y_pred_test))
w,b=codes.logistic_regression_with_regularization(X_train, y_train, lr=0.001, epochs=1000)
y_pred_test = codes.predict(X_test, w, b,threshold=0.01)
print("F1 (test):", codes.f1_score(y_test, y_pred_test))



# Probabilidades reales
probs = codes.sigmoid(np.dot(X_test, W) + b)
best_thresh, best_f1 = codes.find_best_threshold(y_test, probs)
print(f"Mejor threshold: {best_thresh:.2f} → F1: {best_f1:.4f}")












