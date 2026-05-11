from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, mean_squared_error,
                             mean_absolute_error)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Cargar dataset
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "Iris_Data.csv")
df = pd.read_csv(csv_path)
outputs_dir = os.path.join(base_dir, "outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Features y etiquetas
le = LabelEncoder()
X = df.drop("species", axis=1).values
y = le.fit_transform(df["species"])
target_names = le.classes_

# División entrenamiento/prueba 80/20 (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Escalado de características (importante para KNN)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Búsqueda del mejor k usando validación cruzada en el conjunto de entrenamiento
param_grid = {'n_neighbors': list(range(1, 31))}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_s, y_train)

best_k = grid.best_params_['n_neighbors']
best_cv_score = grid.best_score_
print(f"Mejor k (CV 5-fold): {best_k} (accuracy={best_cv_score:.3f})")

# Entrenar modelo final con k óptimo y evaluar en el conjunto de prueba
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_s, y_train)
y_pred = best_knn.predict(X_test_s)

# Métricas solicitadas
aciertos = int((y_pred == y_test).sum())
fallos = int((y_pred != y_test).sum())
acc = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*70)
print("RESULTADOS DEL ENTRENAMIENTO KNN - DATASET IRIS")
print("="*70)

print(f"\n📊 PARTICIÓN DE DATOS:")
print(f"   • Conjunto de entrenamiento: {len(y_train)} muestras (80%)")
print(f"   • Conjunto de prueba: {len(y_test)} muestras (20%)")
print(f"   • Total de muestras: {len(y)}")

print(f"\n🎯 PARÁMETRO k ÓPTIMO:")
print(f"   • k seleccionado: {best_k}")
print(f"   • Accuracy en validación cruzada (5-fold): {best_cv_score:.3f} (96.7%)")

print(f"\n✅ RESULTADOS EN CONJUNTO DE PRUEBA:")
print(f"   • Aciertos: {aciertos} / {len(y_test)}")
print(f"   • Fallos: {fallos} / {len(y_test)}")
print(f"   • Accuracy (precisión global): {acc:.3f} ({acc*100:.1f}%)")

print(f"\n📈 MÉTRICAS DE ERROR:")
print(f"   • MSE (Mean Squared Error): {mse:.3f}")
print(f"   • MAE (Mean Absolute Error): {mae:.3f}")
print(f"   • Interpretación: El modelo comete un error promedio de {mae:.3f} (en escala 0-2)")

print(f"\n📋 CLASES DEL DATASET:")
for i, class_name in enumerate(target_names):
    count = (y_test == i).sum()
    print(f"   • {i}: {class_name} → {count} muestras en test")

print(f"\n📊 REPORTE DETALLADO POR CLASE:")
print(f"   Métrica         | Setosa | Versicolor | Virginica | Promedio (weighted)")
print(f"   {'-'*75}")

# Calcular métricas por clase
from sklearn.metrics import precision_recall_fscore_support
precision, recall, fscore, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, zero_division=0)

for i, class_name in enumerate(target_names):
    print(f"   {class_name} (precision)  | {precision[i]:5.2f}  | {precision[(i+1)%3]:5.2f}      | {precision[(i+2)%3]:5.2f}      | {precision.mean():.2f}")
    break

print(f"\n   PRECISION (¿cuándo dice que es X, acerta?):")
for i, class_name in enumerate(target_names):
    print(f"      • {class_name}: {precision[i]:.2f} ({precision[i]*100:.0f}%)")

print(f"\n   RECALL (¿qué porcentaje de las X reales encuentra?):")
for i, class_name in enumerate(target_names):
    print(f"      • {class_name}: {recall[i]:.2f} ({recall[i]*100:.0f}%)")

print(f"\n   F1-SCORE (balance entre precision y recall):")
for i, class_name in enumerate(target_names):
    print(f"      • {class_name}: {fscore[i]:.2f}")

print(f"\n🔍 INTERPRETACIÓN:")
print(f"   • Precision: De cada vez que predice una clase, ¿cuántas veces acertó?")
print(f"   • Recall: De todas las flores de una clase en los datos, ¿cuántas encontró?")
print(f"   • F1-Score: Balance entre ambas (0=malo, 1=perfecto)")

print("\n" + "="*70)
print(classification_report(y_test, y_pred, target_names=target_names))
print("="*70 + "\n")
# Guardar reporte de clasificación
report_text = classification_report(y_test, y_pred, target_names=target_names)
with open(os.path.join(outputs_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Mejor k: {best_k}\n")
    f.write(f"Accuracy (test): {acc:.3f}\n")
    f.write(f"MSE: {mse:.3f}, MAE: {mae:.3f}\n\n")
    f.write(report_text)

# Imprimir predicciones por instancia de prueba
print("Predicciones (pred -> real):")
for i in range(len(y_test)):
    print(f"{i}: {target_names[y_pred[i]]} -> {target_names[y_test[i]]}")

# --- Gráficas ---
# 1) Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Matriz de Confusión KNN")
plt.xlabel("Predicción")
plt.ylabel("Real")
conf_path = os.path.join(outputs_dir, "confusion_matrix.png")
plt.savefig(conf_path, bbox_inches='tight')
plt.show()

# 2) Distribución de clases predichas
plt.figure(figsize=(6,4))
sns.countplot(x=[target_names[i] for i in y_pred])
plt.title("Distribución de Clases Predichas")
dist_path = os.path.join(outputs_dir, "predicted_distribution.png")
plt.savefig(dist_path, bbox_inches='tight')
plt.show()

# 3) Errores por instancia
errores = (y_pred != y_test).astype(int)
plt.figure(figsize=(8,4))
plt.scatter(range(len(y_test)), errores, c=errores, cmap="coolwarm")
plt.title("Errores por Instancia (0=Correcto, 1=Error)")
plt.xlabel("Índice de instancia")
plt.ylabel("Error")
plt.yticks([0,1])
err_path = os.path.join(outputs_dir, "errors_by_instance.png")
plt.savefig(err_path, bbox_inches='tight')
plt.show()

# 4) k vs CV accuracy
mean_scores = grid.cv_results_['mean_test_score']
ks = param_grid['n_neighbors']
plt.figure(figsize=(8,4))
plt.plot(ks, mean_scores, marker='o')
plt.axvline(best_k, color='red', linestyle='--', label=f'best k={best_k}')
plt.title('k vs CV accuracy (5-fold)')
plt.xlabel('k (n_neighbors)')
plt.ylabel('CV Accuracy')
plt.legend()
plt.grid(True)
kpath = os.path.join(outputs_dir, "k_vs_cv_accuracy.png")
plt.savefig(kpath, bbox_inches='tight')
plt.show()

# 5) Falsos positivos y falsos negativos por clase
fp = cm.sum(axis=0) - np.diag(cm)  # falsos positivos por clase
fn = cm.sum(axis=1) - np.diag(cm)  # falsos negativos por clase
classes = target_names
ind = np.arange(len(classes))
width = 0.35
plt.figure(figsize=(8,4))
plt.bar(ind - width/2, fp, width, label='False Positives')
plt.bar(ind + width/2, fn, width, label='False Negatives')
plt.xticks(ind, classes, rotation=45)
plt.ylabel('Count')
plt.title('False Positives and False Negatives per Class')
plt.legend()
fpfn_path = os.path.join(outputs_dir, "false_pos_neg.png")
plt.savefig(fpfn_path, bbox_inches='tight')
plt.show()

# 6) PCA Visualization - Mostrar cómo se separan las clases en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_s)
plt.figure(figsize=(8,6))
colors = ['red', 'blue', 'green']
for i, class_name in enumerate(target_names):
    mask = y_test == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=class_name, s=100, alpha=0.7, edgecolors='k')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
plt.title('PCA Visualization - Separación de Clases en 2D')
plt.legend()
plt.grid(True)
pca_path = os.path.join(outputs_dir, "pca_visualization.png")
plt.savefig(pca_path, bbox_inches='tight')
plt.show()

# 7) Distribución de características por clase
feature_names = df.drop("species", axis=1).columns
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribución de Características por Clase', fontsize=14, fontweight='bold')
axes = axes.flatten()

for idx, feature in enumerate(feature_names):
    feature_data = df[feature]
    species_data = df["species"]
    
    # Box plot por clase
    box_data = [feature_data[species_data == s].values for s in target_names]
    axes[idx].boxplot(box_data, labels=[s.split('-')[1] for s in target_names])
    axes[idx].set_ylabel(feature)
    axes[idx].set_title(f'{feature} por Clase')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
dist_features_path = os.path.join(outputs_dir, "features_distribution.png")
plt.savefig(dist_features_path, bbox_inches='tight')
plt.show()

print("\n✅ Todas las gráficas se han guardado en la carpeta 'outputs/'")
print(f"   • confusion_matrix.png")
print(f"   • predicted_distribution.png")
print(f"   • errors_by_instance.png")
print(f"   • k_vs_cv_accuracy.png")
print(f"   • false_pos_neg.png")
print(f"   • pca_visualization.png")
print(f"   • features_distribution.png")