import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para evitar bloqueos
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, roc_auc_score

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
plt.savefig('profit_distribution.png')
plt.close()

# Gráfica de Miembros Loyalty
sns.countplot(x='loyalty_member', data=df)
plt.title('Distribución de Miembros Loyalty')
plt.savefig('loyalty_distribution.png')
plt.close()

# ==========================================
# 3. PREPROCESAMIENTO
# ==========================================
# Convertimos Booleano a 0 y 1
df['loyalty_member'] = df['loyalty_member'].astype(int)

# Configuración de tamaño de datos para entrenamiento
# True: usa todos los registros. False: usa una muestra para acelerar ejecución.
USE_FULL_DATA = True
SAMPLE_SIZE = 50000

def fit_encoder(values):
    unique_values = pd.Series(values).dropna().astype(str).unique()
    return {value: idx for idx, value in enumerate(sorted(unique_values))}

def transform_with_encoder(values, encoder, unknown_value=-1):
    return pd.Series(values).astype(str).map(encoder).fillna(unknown_value).astype(int)

# ==========================================
# TAREA 1: Predicción de Loyalty Member
# ==========================================
print("\n>>> TAREA 1: Clasificación de Loyalty Member")
print("Tamaño del dataset:", len(df))
print("Preparando datos...")

# Selección de datos según configuración
if USE_FULL_DATA:
    df_sample = df
    print(f"Usando todos los registros: {len(df_sample)}")
else:
    df_sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    print(f"Usando muestra de {len(df_sample)} registros")

X1 = df_sample[['age', 'gender', 'revenue', 'profit']]
y1 = df_sample['loyalty_member']

print("Dividiendo en entrenamiento/prueba...")
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
print(f"Entrenamiento: {len(X_train1)} ({len(X_train1)/len(df_sample):.0%}) | Prueba: {len(X_test1)} ({len(X_test1)/len(df_sample):.0%})")

# Codificar usando solo el conjunto de entrenamiento (sin fuga de datos)
gender_encoder = fit_encoder(X_train1['gender'])
X_train1 = X_train1.copy()
X_test1 = X_test1.copy()
X_train1['gender_enc'] = transform_with_encoder(X_train1['gender'], gender_encoder)
X_test1['gender_enc'] = transform_with_encoder(X_test1['gender'], gender_encoder)
X_train1 = X_train1[['age', 'gender_enc', 'revenue', 'profit']]
X_test1 = X_test1[['age', 'gender_enc', 'revenue', 'profit']]

print("Entrenando modelo (esto puede tardar)...")
model_loyalty = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
model_loyalty.fit(X_train1, y_train1)

print("Realizando predicciones...")
preds_loyalty = model_loyalty.predict(X_test1)
proba_loyalty = model_loyalty.predict_proba(X_test1)[:, 1]

# Comparación y Métricas
accuracy = accuracy_score(y_test1, preds_loyalty)
print(f"Accuracy: {accuracy:.4f}")
precision = precision_score(y_test1, preds_loyalty)
recall = recall_score(y_test1, preds_loyalty)
f1 = f1_score(y_test1, preds_loyalty)
roc_auc = roc_auc_score(y_test1, proba_loyalty)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_test1, preds_loyalty, digits=4))

print("Calculando matriz de confusión...")
cm = confusion_matrix(y_test1, preds_loyalty)
print(f"Matriz de confusión calculada: {cm.shape}")

tn, fp, fn, tp = cm.ravel()
print(f"TN (Verdaderos Negativos): {tn}")
print(f"FP (Falsos Positivos): {fp}")
print(f"FN (Falsos Negativos): {fn}")
print(f"TP (Verdaderos Positivos): {tp}")

aciertos = int((preds_loyalty == y_test1).sum())
fallos = int((preds_loyalty != y_test1).sum())
print(f"Aciertos en prueba: {aciertos}")
print(f"Fallos en prueba: {fallos}")

classification_eval = pd.DataFrame({
    'y_real': y_test1.values,
    'y_pred': preds_loyalty,
    'prob_clase_1': proba_loyalty
})
classification_eval['resultado'] = np.where(
    classification_eval['y_real'] == classification_eval['y_pred'],
    'Acierto',
    'Fallo'
)
classification_eval.to_csv('classification_test_predictions.csv', index=False)
print("Se guardó detalle de predicciones de clasificación en classification_test_predictions.csv")

print("Creando gráfico de matriz de confusión...")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión: Loyalty Member')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.savefig('confusion_matrix_loyalty.png')
plt.close()

# Gráfico de aciertos vs fallos
plt.figure(figsize=(8, 5))
bars = plt.bar(['Aciertos', 'Fallos'], [aciertos, fallos], color=['green', 'red'])
plt.title('Clasificación en Test: Aciertos vs Fallos')
plt.ylabel('Cantidad')
for bar in bars:
    yval = int(bar.get_height())
    plt.text(bar.get_x() + bar.get_width() / 2, yval, str(yval), ha='center', va='bottom')
plt.savefig('classification_hits_vs_errors.png')
plt.close()

# Curva ROC
fpr_tpr = pd.DataFrame({'y_real': y_test1.values, 'prob': proba_loyalty}).sort_values('prob', ascending=False)
tpr_list = [0.0]
fpr_list = [0.0]
positives = (fpr_tpr['y_real'] == 1).sum()
negatives = (fpr_tpr['y_real'] == 0).sum()
tp_running = 0
fp_running = 0
for real_value in fpr_tpr['y_real']:
    if real_value == 1:
        tp_running += 1
    else:
        fp_running += 1
    tpr_list.append(tp_running / positives if positives > 0 else 0)
    fpr_list.append(fp_running / negatives if negatives > 0 else 0)

plt.figure(figsize=(8, 6))
plt.plot(fpr_list, tpr_list, label=f'ROC AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Azar')
plt.title('Curva ROC - Loyalty Member')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_curve_loyalty.png')
plt.close()

# ==========================================
# TAREA 2: Predicción de Profit
# ==========================================
print("\n>>> TAREA 2: Regresión de Profit")
print("Preparando datos para regresión...")

if USE_FULL_DATA:
    df_sample2 = df
    print(f"Usando todos los registros: {len(df_sample2)}")
else:
    df_sample2 = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    print(f"Usando muestra de {len(df_sample2)} registros")

# Características: marca, cacao, ciudad, descuento
X2 = df_sample2[['brand', 'cocoa_percent', 'city', 'discount']]
y2 = df_sample2['profit']

print("Dividiendo en entrenamiento/prueba...")
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
print(f"Entrenamiento: {len(X_train2)} ({len(X_train2)/len(df_sample2):.0%}) | Prueba: {len(X_test2)} ({len(X_test2)/len(df_sample2):.0%})")

# Codificar usando solo el conjunto de entrenamiento (sin fuga de datos)
brand_encoder = fit_encoder(X_train2['brand'])
city_encoder = fit_encoder(X_train2['city'])
X_train2 = X_train2.copy()
X_test2 = X_test2.copy()
X_train2['brand_enc'] = transform_with_encoder(X_train2['brand'], brand_encoder)
X_test2['brand_enc'] = transform_with_encoder(X_test2['brand'], brand_encoder)
X_train2['city_enc'] = transform_with_encoder(X_train2['city'], city_encoder)
X_test2['city_enc'] = transform_with_encoder(X_test2['city'], city_encoder)
X_train2 = X_train2[['brand_enc', 'cocoa_percent', 'city_enc', 'discount']]
X_test2 = X_test2[['brand_enc', 'cocoa_percent', 'city_enc', 'discount']]

print("Entrenando modelo de regresión...")
model_profit = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
model_profit.fit(X_train2, y_train2)

print("Realizando predicciones de profit...")
preds_profit = model_profit.predict(X_test2)

# Métricas de Error
mse = mean_squared_error(y_test2, preds_profit)
mae = mean_absolute_error(y_test2, preds_profit)
rmse = np.sqrt(mse)
r2 = r2_score(y_test2, preds_profit)
mape = (np.abs((y_test2 - preds_profit) / y_test2.replace(0, np.nan))).mean() * 100

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Error por cada predicción del conjunto de prueba
profit_eval = pd.DataFrame({
    'y_real': y_test2.values,
    'y_pred': preds_profit
})
profit_eval['error'] = profit_eval['y_pred'] - profit_eval['y_real']
profit_eval['error_abs'] = np.abs(profit_eval['error'])
profit_eval['error_cuadratico'] = profit_eval['error'] ** 2
profit_eval.to_csv('profit_test_errors.csv', index=False)

print("\nPrimeras 10 predicciones con su error:")
print(profit_eval.head(10).to_string(index=False))
print("Se guardó detalle completo de errores en profit_test_errors.csv")

# Gráfica Real vs Predicho
plt.figure(figsize=(10, 5))
plt.scatter(y_test2, preds_profit, alpha=0.5)
plt.plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()], 'r--')
plt.title('Profit: Real vs Predicho')
plt.savefig('profit_prediction.png')
plt.close()

# Histograma de errores absolutos
plt.figure(figsize=(10, 5))
sns.histplot(profit_eval['error_abs'], bins=40, kde=True, color='orange')
plt.title('Distribución del Error Absoluto en Test')
plt.xlabel('Error Absoluto')
plt.ylabel('Frecuencia')
plt.savefig('profit_absolute_error_distribution.png')
plt.close()

# Residuos (error) vs predicción
plt.figure(figsize=(10, 5))
plt.scatter(profit_eval['y_pred'], profit_eval['error'], alpha=0.35)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuos vs Predicción')
plt.xlabel('Predicción de Profit')
plt.ylabel('Residuo (Pred - Real)')
plt.savefig('profit_residuals_vs_prediction.png')
plt.close()

# Top errores absolutos
top_errors = profit_eval.nlargest(20, 'error_abs').reset_index(drop=True)
plt.figure(figsize=(12, 6))
plt.bar(top_errors.index.astype(str), top_errors['error_abs'], color='crimson')
plt.title('Top 20 Mayores Errores Absolutos en Test')
plt.xlabel('Índice en ranking de error')
plt.ylabel('Error Absoluto')
plt.savefig('profit_top20_absolute_errors.png')
plt.close()

# ==========================================
# TAREA 3: Simulador de Descuento
# ==========================
print("\n>>> TAREA 3: Simulación de Beneficio Final")
def simular_profit(marca_nombre, porcentaje_cacao, descuento_x):
    m_enc = brand_encoder.get(str(marca_nombre), -1)
    
    # Supongamos una ciudad promedio para la simulación
    ciudad_prom = int(X_train2['city_enc'].mode()[0])
    
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