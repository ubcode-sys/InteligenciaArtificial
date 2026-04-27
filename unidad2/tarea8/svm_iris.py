import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error
)
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  CLASIFICACIÓN DE FLORES IRIS CON SVM")
print("=" * 65)

df = pd.read_csv("Iris_Data.csv")

print("\n📋 Primeras filas del dataset:")
print(df.head())
print(f"\n📐 Forma del dataset: {df.shape[0]} filas × {df.shape[1]} columnas")

print("\n" + "=" * 65)
print("  ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 65)

print("\n📊 Estadísticas descriptivas:")
print(df.describe())

print("\n🔍 Valores nulos por columna:")
print(df.isnull().sum())

print("\n🌸 Distribución de clases:")
print(df["species"].value_counts())

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Análisis Exploratorio – Dataset Iris", fontsize=15, fontweight="bold")

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
colors = {"Iris-setosa": "#e74c3c", "Iris-versicolor": "#2ecc71", "Iris-virginica": "#3498db"}

for i, feat in enumerate(features):
    ax = axes[i // 2][i % 3]
    for species, grp in df.groupby("species"):
        ax.hist(grp[feat], alpha=0.6, label=species, color=colors[species], bins=15)
    ax.set_title(f"Distribución: {feat}", fontsize=11)
    ax.set_xlabel(feat)
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=8)

# Boxplot de todas las características
ax_box = axes[1][2]
df_melt = df.melt(id_vars="species", value_vars=features, var_name="Característica", value_name="Valor")
sns.boxplot(data=df_melt, x="Característica", y="Valor", hue="species",
            palette=colors, ax=ax_box)
ax_box.set_title("Boxplot por Clase", fontsize=11)
ax_box.legend(fontsize=7)
ax_box.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("1_analisis_exploratorio.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n✅ Gráfica guardada: 1_analisis_exploratorio.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Dispersión de características clave", fontsize=13, fontweight="bold")

for species, grp in df.groupby("species"):
    axes[0].scatter(grp["petal_length"], grp["petal_width"],
                    label=species, alpha=0.7, color=colors[species], s=60)
axes[0].set_xlabel("Petal Length")
axes[0].set_ylabel("Petal Width")
axes[0].set_title("Petal Length vs Petal Width")
axes[0].legend()

for species, grp in df.groupby("species"):
    axes[1].scatter(grp["sepal_length"], grp["sepal_width"],
                    label=species, alpha=0.7, color=colors[species], s=60)
axes[1].set_xlabel("Sepal Length")
axes[1].set_ylabel("Sepal Width")
axes[1].set_title("Sepal Length vs Sepal Width")
axes[1].legend()

plt.tight_layout()
plt.savefig("2_scatter_caracteristicas.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅ Gráfica guardada: 2_scatter_caracteristicas.png")

# Mapa de correlación
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Mapa de Correlación entre Características", fontsize=12)
plt.tight_layout()
plt.savefig("3_correlacion.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅ Gráfica guardada: 3_correlacion.png")


print("\n" + "=" * 65)
print("  PREPROCESAMIENTO Y DIVISIÓN 80/20")
print("=" * 65)

X = df[features].values
le = LabelEncoder()
y = le.fit_transform(df["species"])  
class_names = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n🔢 Total de muestras:      {len(X)}")
print(f"   Entrenamiento (80 %):  {len(X_train)}")
print(f"   Prueba        (20 %):  {len(X_test)}")

print("\n" + "=" * 65)
print("  BÚSQUEDA DEL MEJOR VALOR DE C")
print("=" * 65)

from sklearn.model_selection import cross_val_score

C_values = [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
train_acc = []
cv_acc    = []   # validación cruzada SOLO sobre datos de entrenamiento

print(f"\n{'C':>8}  {'Train Acc':>10}  {'CV Acc (5-fold)':>16}")
print("-" * 38)
for c in C_values:
    svm = SVC(kernel="linear", C=c, random_state=42)
    # Accuracy sobre entrenamiento completo
    svm.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, svm.predict(X_train)))
    # Validación cruzada 5-fold SOLO sobre X_train — el test set NO se toca
    scores = cross_val_score(SVC(kernel="linear", C=c, random_state=42),
                             X_train, y_train, cv=5, scoring="accuracy")
    cv_acc.append(scores.mean())

best_idx = np.argmax(cv_acc)
best_C   = C_values[best_idx]

for c, ta, cv in zip(C_values, train_acc, cv_acc):
    marker = " ◄ MEJOR" if c == best_C else ""
    print(f"{c:>8}  {ta:>10.4f}  {cv:>16.4f}{marker}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(C_values, train_acc, "o-", label="Entrenamiento (80%)", color="#e74c3c")
ax.semilogx(C_values, cv_acc,    "s-", label="CV 5-fold (dentro del 80%)", color="#f39c12")
ax.axvline(best_C, linestyle="--", color="gray", alpha=0.7, label=f"Mejor C = {best_C}")
ax.set_xlabel("Valor de C (escala log)", fontsize=11)
ax.set_ylabel("Exactitud (Accuracy)", fontsize=11)
ax.set_title("Selección de C mediante Validación Cruzada (sin tocar el 20% de prueba)", fontsize=11)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("4_seleccion_C.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\n✅ Gráfica guardada: 4_seleccion_C.png")
print(f"\n⭐ Mejor C seleccionado: {best_C}  (elegido con CV sobre el 80% de entrenamiento)")
print(f"   ⚠️  El conjunto de prueba (20%) NO fue usado en este paso.")

print("\n" + "=" * 65)
print(f"  ENTRENAMIENTO FINAL  (C = {best_C})")
print("=" * 65)

svm_final = SVC(kernel="linear", C=best_C,
                probability=True, random_state=42)
svm_final.fit(X_train, y_train)

print("\n" + "=" * 65)
print("  PREDICCIONES SOBRE EL CONJUNTO DE PRUEBA (20 %)")
print("=" * 65)

y_pred = svm_final.predict(X_test)

aciertos = np.sum(y_pred == y_test)
fallos   = np.sum(y_pred != y_test)

print(f"\n✅ Aciertos:  {aciertos} / {len(y_test)}")
print(f"❌ Fallos:    {fallos}   / {len(y_test)}")
print(f"\n📈 Accuracy:  {accuracy_score(y_test, y_pred):.4f} "
      f"({accuracy_score(y_test, y_pred)*100:.2f} %)")

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\n📉 MSE  (Error Cuadrático Medio):  {mse:.4f}")
print(f"📉 MAE  (Error Absoluto Medio):    {mae:.4f}")
print(f"📉 RMSE (Raíz del MSE):            {rmse:.4f}")

print("\n📋 Reporte de clasificación completo:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\n🔎 Error por predicción (|clase_pred - clase_real|):")
print(f"{'#':>4}  {'Real':>20}  {'Predicción':>20}  {'Error':>6}  {'OK?':>5}")
print("-" * 62)
for i, (real, pred) in enumerate(zip(y_test, y_pred)):
    err = abs(pred - real)
    ok  = "✅" if err == 0 else "❌"
    print(f"{i+1:>4}  {class_names[real]:>20}  {class_names[pred]:>20}  {err:>6}  {ok:>5}")

fig, ax = plt.subplots(figsize=(14, 4))
errors = np.abs(y_pred - y_test)
bar_colors = ["#2ecc71" if e == 0 else "#e74c3c" for e in errors]
ax.bar(range(len(errors)), errors, color=bar_colors, edgecolor="white")
ax.set_xlabel("Índice de muestra de prueba")
ax.set_ylabel("Error (|pred - real|)")
ax.set_title(f"Error por predicción — {aciertos} aciertos, {fallos} fallos")
ax.set_ylim(0, 2.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2ecc71", label="Correcto"),
                   Patch(color="#e74c3c", label="Incorrecto")])
plt.tight_layout()
plt.savefig("5_error_por_prediccion.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n✅ Gráfica guardada: 5_error_por_prediccion.png")

# Matriz de confusión
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            linewidths=0.5)
ax.set_xlabel("Clase Predicha", fontsize=11)
ax.set_ylabel("Clase Real",     fontsize=11)
ax.set_title("Matriz de Confusión", fontsize=12)
plt.tight_layout()
plt.savefig("6_matriz_confusion.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅ Gráfica guardada: 6_matriz_confusion.png")

fp_values = []
fn_values = []
for cls_idx in range(len(class_names)):
    fp = np.sum((y_pred == cls_idx) & (y_test != cls_idx))
    fn = np.sum((y_pred != cls_idx) & (y_test == cls_idx))
    fp_values.append(int(fp))
    fn_values.append(int(fn))

x = np.arange(len(class_names))
width = 0.36

fig, ax = plt.subplots(figsize=(9, 6))
bars_fp = ax.bar(x - width / 2, fp_values, width, label="Falsos Positivos", color="#e74c3c")
bars_fn = ax.bar(x + width / 2, fn_values, width, label="Falsos Negativos", color="#3498db")

max_count = max(max(fp_values), max(fn_values))
y_top = max(1, max_count + 1)
ax.set_ylim(0, y_top)
ax.set_title("Falsos Positivos y Falsos Negativos por Clase", fontsize=12)
ax.set_xlabel("Clase")
ax.set_ylabel("Cantidad")
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=15, ha="right")
ax.grid(axis="y", alpha=0.3)
ax.legend()

for bar in list(bars_fp) + list(bars_fn):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + (0.03 * y_top), f"{int(height)}", ha="center")

if max_count == 0:
    ax.text(
        0.5,
        0.9,
        "Sin errores en prueba: FP = 0 y FN = 0",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

plt.tight_layout()
plt.savefig("7_fp_fn_por_clase.png", dpi=120, bbox_inches="tight")
plt.close()
print("✅ Gráfica guardada: 7_fp_fn_por_clase.png")

print("\n" + "=" * 65)
print("  PREDICCIÓN DE NUEVAS FLORES (ejemplos)")
print("=" * 65)

nuevas_flores = [
    [5.1, 3.5, 1.4, 0.2],   # típica setosa
    [6.0, 2.9, 4.5, 1.5],   # típica versicolor
    [6.7, 3.1, 5.6, 2.4],   # típica virginica
    [5.8, 2.8, 5.1, 2.4],   # caso límite
]
nombres_ejemplos = ["Ejemplo A", "Ejemplo B", "Ejemplo C", "Ejemplo D (límite)"]

nuevas_scaled = scaler.transform(nuevas_flores)
preds  = svm_final.predict(nuevas_scaled)
probas = svm_final.predict_proba(nuevas_scaled)

print(f"\n{'Muestra':>18}  {'Clase Predicha':>20}  Probabilidades por clase")
print(f"{'':>18}  {'':>20}  "
      f"{'setosa':>10}  {'versicolor':>12}  {'virginica':>10}")
print("-" * 76)
for nom, flor, pred, prob in zip(nombres_ejemplos, nuevas_flores, preds, probas):
    clase = class_names[pred]
    print(f"{nom:>18}  {clase:>20}  "
          f"{prob[0]:>10.4f}  {prob[1]:>12.4f}  {prob[2]:>10.4f}")
    print(f"{'Valores entrada':>18}: sepal_l={flor[0]}, sepal_w={flor[1]}, "
          f"petal_l={flor[2]}, petal_w={flor[3]}")

print("\n" + "=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"  Modelo:            SVM con kernel Lineal")
print(f"  Mejor C:           {best_C}")
print(f"  Total pruebas:     {len(y_test)}")
print(f"  Aciertos:          {aciertos}  ({aciertos/len(y_test)*100:.1f} %)")
print(f"  Fallos:            {fallos}  ({fallos/len(y_test)*100:.1f} %)")
print(f"  MSE:               {mse:.4f}")
print(f"  MAE:               {mae:.4f}")
print(f"  RMSE:              {rmse:.4f}")
print("=" * 65)
print("\n🎓 Programa finalizado correctamente.\n")
