import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. CARGA Y UNIÓN DE DATOS
# ==========================================
# Cargamos los archivos (asegúrate de que estén en la misma carpeta)
df_sales = pd.read_csv('sales.csv')
df_products = pd.read_csv('products.csv')
df_customers = pd.read_csv('customers.csv')
df_stores = pd.read_csv('stores.csv')

# Unificamos todo en un solo DataFrame
df = df_sales.merge(df_products, on='product_id') \
             .merge(df_customers, on='customer_id') \
             .merge(df_stores, on='store_id')

# ==========================================
# 2. ANÁLISIS DE DATOS (EDA)
# ==========================================
print("--- Análisis de Distribución ---")
print(df.describe())

# Gráfica de distribución de Profit
plt.figure(figsize=(10, 5))
sns.histplot(df['profit'], kde=True, color='brown')
plt.title('Distribución del Beneficio (Profit)')
plt.show()

# Gráfica de Miembros Loyalty
sns.countplot(x='loyalty_member', data=df)
plt.title('Distribución de Miembros Loyalty')
plt.show()

# ==========================================
# 3. PREPROCESAMIENTO
# ==========================================
# Convertimos variables categóricas a numéricas
le = LabelEncoder()
df['brand_enc'] = le.fit_transform(df['brand'])
df['city_enc'] = le.fit_transform(df['city'])
df['gender_enc'] = le.fit_transform(df['gender'])
# Convertimos Booleano a 0 y 1
df['loyalty_member'] = df['loyalty_member'].astype(int)

# ==========================================
# TAREA 1: Predicción de Loyalty Member
# ==========================================
print("\n>>> TAREA 1: Clasificación de Loyalty Member")
X1 = df[['age', 'gender_enc', 'revenue', 'profit']]
y1 = df['loyalty_member']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

model_loyalty = RandomForestClassifier(n_estimators=100, random_state=42)
model_loyalty.fit(X_train1, y_train1)
preds_loyalty = model_loyalty.predict(X_test1)

# Comparación y Métricas
print(f"Accuracy: {accuracy_score(y_test1, preds_loyalty):.4f}")
cm = confusion_matrix(y_test1, preds_loyalty)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión: Loyalty Member')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.show()

# ==========================================
# TAREA 2: Predicción de Profit
# ==========================================
print("\n>>> TAREA 2: Regresión de Profit")
# Características: marca, cacao, ciudad, descuento
X2 = df[['brand_enc', 'cocoa_percent', 'city_enc', 'discount']]
y2 = df['profit']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

model_profit = RandomForestRegressor(n_estimators=100, random_state=42)
model_profit.fit(X_train2, y_train2)
preds_profit = model_profit.predict(X_test2)

# Métricas de Error
print(f"MSE: {mean_squared_error(y_test2, preds_profit):.4f}")
print(f"MAE: {mean_absolute_error(y_test2, preds_profit):.4f}")

# Gráfica Real vs Predicho
plt.figure(figsize=(10, 5))
plt.scatter(y_test2, preds_profit, alpha=0.5)
plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'r--')
plt.title('Profit: Real vs Predicho')
plt.show()

# ==========================================
# TAREA 3: Simulador de Descuento
# ==========================
print("\n>>> TAREA 3: Simulación de Beneficio Final")
def simular_profit(marca_nombre, porcentaje_cacao, descuento_x):
    try:
        m_enc = le.transform([marca_nombre])[0]
    except:
        m_enc = df['brand_enc'].mean() # Fallback si la marca no existe
    
    # Supongamos una ciudad promedio para la simulación
    ciudad_prom = df['city_enc'].mode()[0]
    
    entrada = pd.DataFrame([[m_enc, porcentaje_cacao, ciudad_prom, descuento_x/100]], 
                           columns=['brand_enc', 'cocoa_percent', 'city_enc', 'discount'])
    resultado = model_profit.predict(entrada)
    return resultado[0]

# Ejemplo de uso solicitado: Marca 'y' (ajusta al nombre real en tu csv), 70% cacao
descuentos = [5, 10, 15, 20]
marca_test = df['brand'].iloc[0] # Usamos la primera marca como ejemplo 'y'

print(f"Simulación para marca: {marca_test} con 70% de Cacao:")
for d in descuentos:
    ganancia = simular_profit(marca_test, 70, d)
    print(f"Descuento {d}% -> Beneficio Estimado: ${ganancia:.2f}")