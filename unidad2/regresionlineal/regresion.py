# Entrenando modelo de regresión lineal y evaluando errores de predicción
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carga de los datos
df = pd.read_csv("/Admission_Predict.csv")

# Variables independientes
X = df[["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR ", "CGPA", "Research"]]
#Variables dependientes
y = df["Chance of Admit "]

# Dividir en conjunto de entrenamiento y prueba
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajuste del modelo de regresión lineal
model = LinearRegression()
model.fit(Xtrain, ytrain)

# Obtención de coeficientes
coeff = pd.Series(model.coef_, index=X.columns)

# Predicciones en el conjunto completo para promedio de admisión
ypredall = model.predict(X)
avrg = ypredall.mean()

# Predicciones en el conjunto de prueba
ypred = model.predict(Xtest)

# Calculo de errores
mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)

# Mostrar resultados
print("Coeficientes del modelo de regresión lineal:")
print(coeff)
print(f"\nPromedio del porcentaje de admisión: {avrg * 100:.2f}%")
print(f"\nError Absoluto Medio (MAE) en el conjunto de prueba: {mae:.4f}")
print(f"Error Cuadrático Medio (MSE) en el conjunto de prueba: {mse:.4f}")
