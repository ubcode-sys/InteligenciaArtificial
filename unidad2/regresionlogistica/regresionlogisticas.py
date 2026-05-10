import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report

try:
    df = pd.read_csv('Titanic-Dataset.csv') 
except FileNotFoundError:
    print("Error: No se encontró el archivo 'Titanic-Dataset.csv'. Asegúrate de que esté en la carpeta correcta.")
    exit()

sns.set_theme(style="whitegrid")

print("--- 1. ANÁLISIS EXPLORATORIO Y LIMPIEZA ---")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Análisis de Variables Categóricas', fontsize=16, fontweight='bold', color='#2c3e50')

# Género
sns.countplot(data=df, x='Sex', ax=axes[0], palette='magma', hue='Sex', legend=False)
axes[0].set_title('Distribución por Género')
axes[0].set_xlabel('Sexo')
axes[0].set_ylabel('Cantidad')
axes[0].set_xticklabels(['Hombre', 'Mujer'] if df['Sex'].iloc[0] == 'male' else ['Mujer', 'Hombre'])

# Puerto
sns.countplot(data=df, x='Embarked', ax=axes[1], palette='viridis', hue='Embarked', legend=False)
axes[1].set_title('Puerto de Origen')
axes[1].set_xlabel('Puerto')
axes[1].set_ylabel('Cantidad')

# Supervivencia
sns.countplot(data=df, x='Survived', ax=axes[2], palette='icefire', hue='Survived', legend=False)
axes[2].set_title('Estado de Supervivencia')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Fallecido', 'Sobreviviente'])
axes[2].set_xlabel('¿Sobrevivió?')
axes[2].set_ylabel('Cantidad')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 2. Histogramas de Variables Numéricas
cols_num = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribución de Variables Numéricas', fontsize=16, fontweight='bold')
axes = axes.flatten()

for i, col in enumerate(cols_num):
    if col in df.columns:
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='#34495e')
        media_val = df[col].mean()
        mediana_val = df[col].median()
        axes[i].axvline(media_val, color='red', label=f'Media: {media_val:.1f}')
        axes[i].axvline(mediana_val, color='orange', linestyle='--', label=f'Mediana: {mediana_val:.1f}')
        axes[i].set_title(f'Distribución de: {col}')
        axes[i].set_ylabel('Frecuencia')
        axes[i].set_xlabel('Valor')
        axes[i].legend()

fig.delaxes(axes[5])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

columnas_a_quitar = [c for c in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if c in df.columns]
df_clean = df.drop(columnas_a_quitar, axis=1)

df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
if 'Embarked' in df_clean.columns:
    df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])


df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
df_clean = pd.get_dummies(df_clean, columns=['Embarked'], drop_first=True)

X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']
nombres_columnas = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Total de registros: {len(df_clean)}")
print(f"Set Entrenamiento: {len(X_train)} | Set Prueba: {len(X_test)}\n")

print("--- 2. ENTRENAMIENTO Y PESOS DEL MODELO (BETAS) ---")
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_train_scaled, y_train)

print(f"Punto de Corte (Beta 0): {modelo_logistico.intercept_[0]:.4f}")
for col, coef in zip(nombres_columnas, modelo_logistico.coef_[0]):
    print(f"Coeficiente β para {col}: {coef:.4f}")

importancias = pd.DataFrame({'Variable': nombres_columnas, 'Peso': modelo_logistico.coef_[0]})
importancias = importancias.sort_values(by='Peso')

plt.figure(figsize=(10, 6))
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in importancias['Peso']]
sns.barplot(data=importancias, x='Peso', y='Variable', palette=colors)
plt.title('Influencia de las Variables en la Supervivencia', fontsize=14)
plt.xlabel('Impacto (Negativo = Menos probabilidad | Positivo = Más probabilidad)')
plt.show()

print("--- 3. PREDICCIONES INDIVIDUALES (MUESTRA) ---")
y_pred_class = modelo_logistico.predict(X_test_scaled)
y_pred_proba = modelo_logistico.predict_proba(X_test_scaled)[:, 1]
y_test_array = y_test.values

for i in range(10):
    error_i = (y_test_array[i] - y_pred_proba[i]) ** 2
    resultado = "✓ Acierto" if y_pred_class[i] == y_test_array[i] else "✗ Falla"
    print(f"Pax {i+1} | Prob: {y_pred_proba[i]*100:5.2f}% | Pred: {y_pred_class[i]} | Real: {y_test_array[i]} | {resultado}")


accuracy = accuracy_score(y_test, y_pred_class)
print(f"\nPrecisión Final: {accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Murió', 'Vivió'], yticklabels=['Murió', 'Vivió'])
plt.title('Matriz de Confusión Final')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()
