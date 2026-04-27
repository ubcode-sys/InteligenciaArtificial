# Entrenando modelo de regresión lineal y evaluando errores de predicción
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
	ConfusionMatrixDisplay,
	confusion_matrix,
	mean_absolute_error,
	mean_squared_error,
	r2_score,
)

# Carga de los datos
csv_path = Path(__file__).resolve().parent / "Admission_Predict.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Variables independientes
X = df[["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]]
#Variables dependientes
y = df["Chance of Admit"]

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
rmse = np.sqrt(mse)
r2 = r2_score(ytest, ypred)
residuals = ytest - ypred

# Mostrar resultados
print("Coeficientes del modelo de regresión lineal:")
print(coeff)
print(f"\nPromedio del porcentaje de admisión: {avrg * 100:.2f}%")
print(f"\nError Absoluto Medio (MAE) en el conjunto de prueba: {mae:.4f}")
print(f"Error Cuadrático Medio (MSE) en el conjunto de prueba: {mse:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de determinación (R²): {r2:.4f}")

# Visualizaciones para interpretar el modelo y sus resultados
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Importancia relativa de variables según coeficientes
coeff.sort_values().plot(kind="barh", ax=axes[0, 0], color="#2E86AB")
axes[0, 0].set_title("Coeficientes del modelo")
axes[0, 0].set_xlabel("Peso en la predicción")

# 2) Relación entre valores reales y predichos
axes[0, 1].scatter(ytest, ypred, alpha=0.75, color="#F18F01", edgecolor="black", linewidth=0.3)
min_val = min(ytest.min(), ypred.min())
max_val = max(ytest.max(), ypred.max())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
axes[0, 1].set_title("Reales vs Predichos")
axes[0, 1].set_xlabel("Valor real")
axes[0, 1].set_ylabel("Valor predicho")

# 3) Distribución de residuos para evaluar sesgo del modelo
axes[1, 0].hist(residuals, bins=15, color="#6A994E", edgecolor="black", alpha=0.85)
axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=2)
axes[1, 0].set_title("Distribución de residuos")
axes[1, 0].set_xlabel("Error (real - predicho)")
axes[1, 0].set_ylabel("Frecuencia")

# 4) Correlación de cada variable con la probabilidad de admisión
corr_with_target = df.corr(numeric_only=True)["Chance of Admit"].drop("Chance of Admit").sort_values()
corr_with_target.plot(kind="barh", ax=axes[1, 1], color="#BC4749")
axes[1, 1].set_title("Correlación con Chance of Admit")
axes[1, 1].set_xlabel("Coeficiente de correlación")

fig.suptitle("Diagnóstico del Modelo de Regresión Lineal", fontsize=14, fontweight="bold")
plt.tight_layout()

plot_path = Path(__file__).resolve().parent / "regresion_diagnostico.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nGráfica guardada en: {plot_path}")

if "agg" in plt.get_backend().lower():
	print("Backend no interactivo detectado: la gráfica se guardó en archivo.")
else:
	plt.show()

# Histogramas para entender mejor la distribución de los datos originales
fig_hist, axes_hist = plt.subplots(2, 2, figsize=(14, 10))

hist_specs = [
	("GRE Score", "#1D3557"),
	("TOEFL Score", "#457B9D"),
	("CGPA", "#A8DADC"),
	("Chance of Admit", "#E63946"),
]

for ax, (column, color) in zip(axes_hist.flat, hist_specs):
	ax.hist(df[column], bins=20, color=color, edgecolor="black", alpha=0.85)
	ax.set_title(f"Histograma de {column}")
	ax.set_xlabel(column)
	ax.set_ylabel("Frecuencia")

fig_hist.suptitle("Distribución de variables clave", fontsize=14, fontweight="bold")
plt.tight_layout()

hist_path = Path(__file__).resolve().parent / "regresion_histogramas.png"
plt.savefig(hist_path, dpi=300, bbox_inches="tight")
print(f"Histograma guardado en: {hist_path}")

# Matriz de confusión auxiliar usando un umbral de admisión
# En regresión no existen falsos positivos/negativos de forma nativa,
# así que esto transforma el problema a una decisión binaria interpretable.
threshold = 0.70
ytest_binary = (ytest >= threshold).astype(int)
ypred_binary = (ypred >= threshold).astype(int)
cm = confusion_matrix(ytest_binary, ypred_binary)
tn, fp, fn, tp = cm.ravel()

fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(
	confusion_matrix=cm,
	display_labels=[f"< {int(threshold * 100)}%", f">= {int(threshold * 100)}%"],
)
disp.plot(ax=ax_cm, cmap="Blues", colorbar=False, values_format="d")
ax_cm.set_title(f"Matriz de confusión con umbral de {int(threshold * 100)}%")

cm_path = Path(__file__).resolve().parent / "regresion_matriz_confusion_umbral.png"
plt.tight_layout()
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
print(f"Matriz de confusión guardada en: {cm_path}")
print(f"Umbral usado para la decisión binaria: {threshold:.2f}")
print(f"Verdaderos negativos: {tn}, Falsos positivos: {fp}, Falsos negativos: {fn}, Verdaderos positivos: {tp}")

# Diagnóstico adicional: residuos contra predicción para detectar sesgo y heterocedasticidad
fig_res, ax_res = plt.subplots(figsize=(8, 6))
ax_res.scatter(ypred, residuals, alpha=0.75, color="#4C78A8", edgecolor="black", linewidth=0.3)
ax_res.axhline(0, color="red", linestyle="--", linewidth=2)
ax_res.set_title("Residuos vs Predicción")
ax_res.set_xlabel("Valor predicho")
ax_res.set_ylabel("Residuo (real - predicho)")

res_path = Path(__file__).resolve().parent / "regresion_residuos_vs_prediccion.png"
plt.tight_layout()
plt.savefig(res_path, dpi=300, bbox_inches="tight")
print(f"Gráfica de residuos vs predicción guardada en: {res_path}")

#como conclusión, podemos ver que el CGPA es el factor dominante por el hecho que refleja el desempeño
#académico sostenido, ya que este mantiene la correlación más fuerte con la probabilidad de aceptación,
#seguido de la experiencia en investigación y las cartas de recomendación, que actuan como diferenciadores
#de clave, aumentando las posibilidades y resfuerza el perfil del candidato, el resto de las variables
# el SOP juegan un papel secundario, como requisitos base.