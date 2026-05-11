KNN Iris - Tarea

Instrucciones rápidas:

- Instalar dependencias (recomendado crear un entorno virtual):

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

- Ejecutar el script:

```bash
python KNN.py
```

Qué hace el script:
- Carga `Iris_Data.csv` desde la misma carpeta.
- Divide los datos 80/20 (con `random_state=42` y `stratify`).
- Escala las características con `StandardScaler`.
- Busca el mejor `k` (1..30) usando `GridSearchCV` (5-fold) sobre el conjunto de entrenamiento.
- Entrena el KNN final con `k` óptimo y evalúa en el conjunto de prueba.
- Imprime: `k` seleccionado, aciertos/fallos, `accuracy`, `MSE`, `MAE`, y el `classification_report`.
- Guarda las gráficas y el reporte en `outputs/`.

Archivos generados:
- `outputs/confusion_matrix.png`
- `outputs/predicted_distribution.png`
- `outputs/errors_by_instance.png`
- `outputs/k_vs_cv_accuracy.png`
- `outputs/classification_report.txt`

Notas:
- `requirements.txt` contiene paquetes necesarios; usa `pip install -r requirements.txt`.
- Si quieres que no se muestren las ventanas de las gráficas y solo se guarden, edita `KNN.py` y elimina las llamadas a `plt.show()`.
