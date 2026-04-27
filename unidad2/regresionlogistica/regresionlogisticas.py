import os
import pandas as pd
import numpy as np

# AHORA SÍ, IMPORTAMOS ESTAS LIBRERÍAS (¡PACIENCIA AL INICIAR!)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# ==========================================
# 1. CARGAR EL DATASET (Método Robust OS)
# ==========================================
# Obtenemos la ruta exacta de la carpeta donde está este script de Python
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Unimos esa ruta con el nombre exacto de tu archivo descargado de Kaggle
nombre_archivo = 'Titanic-Dataset.csv' # Asegúrate de que este es el nombre real
ruta_csv = os.path.join(directorio_actual, nombre_archivo)

print(f"Intentando cargar archivo desde: {ruta_csv}")

try:
    df = pd.read_csv(ruta_csv)
    print("¡Archivo cargado exitosamente!")
except FileNotFoundError:
    print(f"ERROR Fatal: No se encontró el archivo '{nombre_archivo}' en la carpeta '{directorio_actual}'.")
    exit() # Detenemos la ejecución si no hay datos

# ==========================================
# 2. PREPROCESAMIENTO DE DATOS
# ==========================================
# Llenamos valores nulos básicos
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Eliminamos columnas no predictivas directamente
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convertimos variables categóricas a numéricas (One-Hot Encoding)
# 'Sex' se vuelve 'Sex_male' (0=mujer, 1=hombre)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# ==========================================
# 3. SEPARACIÓN DE DATOS (80/20)
# ==========================================
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ==========================================
# 4. ENTRENAMIENTO DEL MODELO
# ==========================================
# Usamos un solucionador compatible con penalización l2 por defecto
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

# ==========================================
# 5. COEFICIENTES (BETA) DEL MODELO
# ==========================================
print("\n" + "="*50)
print("COEFICIENTES DEL MODELO (BETAS) - Requisito Tarea")
print("="*50)
print(f"Intercepto (Beta 0): {modelo.intercept_[0]:.4f}\n")
for feature, coef in zip(X.columns, modelo.coef_[0]):
    print(f"Variable: {feature: <15} | Coeficiente (Beta): {coef:.4f}")

# ==========================================
# 6. PREDICCIONES Y OBTENCIÓN DE PROBABILIDADES
# ==========================================
y_pred_classes = modelo.predict(X_test)

# predict_proba nos da la probabilidad de cada clase (0 y 1)
# Nos interesa la columna [1], que es la probabilidad de sobrevivir (clase 1)
y_pred_probs = modelo.predict_proba(X_test)[:, 1]

# ==========================================
# 7. EVALUACIÓN Y ERRORES
# ==========================================
aciertos = (y_pred_classes == y_test).sum()
fallos = (y_pred_classes != y_test).sum()

mse = mean_squared_error(y_test, y_pred_classes)
mae = mean_absolute_error(y_test, y_pred_classes)

print("\n" + "="*50)
print("MÉTRICAS DE RENDIMIENTO SOBRE PRUEBAS (20%) - Requisito Tarea")
print("="*50)
print(f"Total de casos de prueba : {len(y_test)}")
print(f"Veces que ACIERTA        : {aciertos}")
print(f"Veces que FALLA          : {fallos}")
print(f"Precisión global (Accuracy): {(aciertos/len(y_test))*100:.2f}%")
print(f"\nNivel de Error:")
print(f"Error Cuadrático Medio (MSE) : {mse:.4f}")
print(f"Error Absoluto Medio (MAE)   : {mae:.4f}")

# ==========================================
# 8. GENERACIÓN DE GRÁFICAS VISUALES (NUEVO)
# ==========================================
print("\n" + "="*50)
print("GENERANDO GRÁFICAS VISUALES...")
print("="*50)

# Configuración estética de Seaborn
sns.set_theme(style="whitegrid")

# Creamos una figura con dos subgráficas (una al lado de la otra)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- GRÁFICA 1: Distribución General de Probabilidades ---
# Muestra cuántas personas cayeron en cada rango de probabilidad calculada por el modelo.
sns.histplot(y_pred_probs, bins=20, kde=True, ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Distribución Global de Probabilidades de Supervivencia Predichas', fontsize=14)
axes[0].set_xlabel('Probabilidad Calculada (0.0 a 1.0)', fontsize=12)
axes[0].set_ylabel('Frecuencia (Número de Pasajeros)', fontsize=12)
# Añadimos una línea roja en el umbral por defecto (0.5)
axes[0].axvline(0.5, color='red', linestyle='--', label='Umbral de Decisión (0.5)')
axes[0].legend()

# --- GRÁFICA 2: Distribución de Probabilidades vs. Desenlace Real ---
# Esta es la gráfica más potente: separa a los que realmente murieron de los que sobrevivieron
# y muestra qué probabilidad les asignó el modelo. Un buen modelo tendrá
# las curvas separadas en los extremos.
data_plot = pd.DataFrame({'Probabilidad': y_pred_probs, 'Realidad': y_test.values})
# Mapeamos numérico a texto para la leyenda
data_plot['Realidad'] = data_plot['Realidad'].map({0: 'Falleció (Real)', 1: 'Sobrevivió (Real)'})

sns.histplot(data=data_plot, x='Probabilidad', hue='Realidad', element="step",
             stat="density", common_norm=False, kde=True, bins=20, ax=axes[1],
             palette={'Falleció (Real)': 'red', 'Sobrevivió (Real)': 'green'})

axes[1].set_title('Separación de Probabilidades por Desenlace Real', fontsize=14)
axes[1].set_xlabel('Probabilidad Calculada (0.0 a 1.0)', fontsize=12)
axes[1].set_ylabel('Densidad', fontsize=12)
axes[1].axvline(0.5, color='black', linestyle='--', alpha=0.5)

plt.tight_layout() # Ajusta automáticamente los márgenes
print("Mostrando gráficas. Cierra la ventana de gráficos para finalizar el programa.")
plt.show() # Esta línea detiene la ejecución hasta que cierres la ventana