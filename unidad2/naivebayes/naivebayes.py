
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_iris_dataframe():
    csv_path = Path(__file__).resolve().parent / 'Iris_Data.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        'No se encontro Iris_Data.csv en la misma carpeta del script.'
    )


class GaussianNaiveBayes:
    def fit(self, X_df, y_series):
        self.classes = sorted(y_series.unique())
        n_total = len(y_series)

        self.prior = {}
        for cls in self.classes:
            self.prior[cls] = (y_series == cls).sum() / n_total

        self.stats = {}
        for cls in self.classes:
            subset = X_df[y_series == cls]
            self.stats[cls] = {}
            for feat in X_df.columns:
                mean = subset[feat].mean()
                std = subset[feat].std()
                if std == 0:
                    std = 1e-9
                self.stats[cls][feat] = (mean, std)

    def _gaussian_pdf(self, x, mean, std):
        exponent = -((x - mean) ** 2) / (2 * std ** 2)
        return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(exponent)

    def predict_with_details(self, sample, verbose=True):
        if verbose:
            print('\n' + '=' * 65)
            print('PARAMETROS DE ENTRADA AL MODELO')
            print('=' * 65)
            for feat, val in sample.items():
                print(f'  {feat:<15}: {val}')

            print('\nPROBABILIDADES A PRIORI:')
            for cls in self.classes:
                print(f'  P({cls}) = {self.prior[cls]:.6f}')

        log_posteriors = {}
        posteriors = {}

        for cls in self.classes:
            log_p = math.log(self.prior[cls])

            if verbose:
                print(f'\nVEROSIMILITUDES P(x|{cls}):')

            for feat, val in sample.items():
                mean, std = self.stats[cls][feat]
                lik = self._gaussian_pdf(val, mean, std)
                log_p += math.log(lik + 1e-300)

                if verbose:
                    print(
                        f'  P({feat}={val} | {cls}) = {lik:.10f} '
                        f'(mu={mean:.4f}, sigma={std:.4f})'
                    )

            log_posteriors[cls] = log_p

        max_log = max(log_posteriors.values())
        exp_vals = {c: math.exp(v - max_log) for c, v in log_posteriors.items()}
        total = sum(exp_vals.values())
        for cls in self.classes:
            posteriors[cls] = exp_vals[cls] / total

        if verbose:
            print('\nCALCULOS DE NAIVE BAYES:')
            for cls in self.classes:
                print(f'  log P({cls}|x) = {log_posteriors[cls]:.6f}')

            print('\nPOSTERIORES NORMALIZADAS:')
            for cls in self.classes:
                print(f'  P({cls}|x) = {posteriors[cls]:.6f} ({posteriors[cls] * 100:.2f}%)')

        best_class = max(posteriors, key=posteriors.get)
        return best_class, posteriors

    def predict(self, X_df):
        preds = []
        for _, row in X_df.iterrows():
            sample = {f: row[f] for f in X_df.columns}
            pred, _ = self.predict_with_details(sample, verbose=False)
            preds.append(pred)
        return preds


def stratified_split(df, target_col, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    train_idx = []
    test_idx = []

    for cls in sorted(df[target_col].unique()):
        cls_idx = df.index[df[target_col] == cls].tolist()
        rng.shuffle(cls_idx)
        split_cls = int(train_ratio * len(cls_idx))
        train_idx.extend(cls_idx[:split_cls])
        test_idx.extend(cls_idx[split_cls:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def print_class_distribution(df_subset, target_col, title):
    print(f'\n{title}:')
    counts = df_subset[target_col].value_counts().sort_index()
    total = len(df_subset)
    for cls, count in counts.items():
        pct = (count / total) * 100
        print(f'  {cls:<15}: {count:>3} ({pct:6.2f}%)')


def generate_feature_scatter(df, target_col, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {
        'Iris-setosa': '#1f77b4',
        'Iris-versicolor': '#ff7f0e',
        'Iris-virginica': '#2ca02c',
    }

    for cls, group in df.groupby(target_col):
        axes[1].scatter(
            group['sepal_length'],
            group['petal_length'],
            label=cls,
            alpha=0.8,
            s=35,
            color=colors.get(cls, '#666666'),
        )

    for cls, group in df.groupby(target_col):
        axes[0].scatter(
            group['petal_length'],
            group['petal_width'],
            label=cls,
            alpha=0.8,
            s=35,
            color=colors.get(cls, '#666666'),
        )

    axes[0].set_title('Petal length vs Petal width')
    axes[0].set_xlabel('Petal length')
    axes[0].set_ylabel('Petal width')
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].set_title('Sepal length vs Petal length')
    axes[1].set_xlabel('Sepal length')
    axes[1].set_ylabel('Petal length')
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    fig.suptitle('Separacion de clases en Iris', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def generate_confusion_matrix_plot(y_true, y_pred, labels, output_path):
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for real, pred in zip(y_true, y_pred):
        matrix.loc[real, pred] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix.values, cmap='Blues')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Prediccion')
    ax.set_ylabel('Clase real')
    ax.set_title('Matriz de confusion')

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix.iloc[i, j]
            ax.text(j, i, str(value), ha='center', va='center', color='black')

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def generate_posterior_plot(posteriors, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(posteriors.keys())
    values = [posteriors[label] for label in labels]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bars = ax.bar(labels, values, color=colors[:len(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probabilidad posterior')
    ax.set_title('Confianza del modelo para la muestra evaluada')
    ax.grid(axis='y', alpha=0.2)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}', ha='center')

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def generate_fp_fn_plot(y_true, y_pred, labels, output_path):
    fp_values = []
    fn_values = []

    for cls in labels:
        fp = sum(1 for real, pred in zip(y_true, y_pred) if pred == cls and real != cls)
        fn = sum(1 for real, pred in zip(y_true, y_pred) if real == cls and pred != cls)
        fp_values.append(fp)
        fn_values.append(fn)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_fp = ax.bar(x - width / 2, fp_values, width, label='Falsos positivos', color='#E45756')
    bars_fn = ax.bar(x + width / 2, fn_values, width, label='Falsos negativos', color='#4C78A8')

    max_count = max(max(fp_values), max(fn_values))
    y_top = max(1, max_count + 1)
    ax.set_ylim(0, y_top)

    ax.set_title('Falsos positivos y falsos negativos por clase')
    ax.set_xlabel('Clase')
    ax.set_ylabel('Cantidad')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.grid(axis='y', alpha=0.2)
    ax.legend(frameon=False)

    for bar in list(bars_fp) + list(bars_fn):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + (0.03 * y_top), f'{int(height)}', ha='center')

    if max_count == 0:
        ax.text(
            0.5,
            0.9,
            'Sin errores en pruebas: FP = 0 y FN = 0',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=11,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc'),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def generate_test_bell_histogram_with_mean(X_test_df, output_path):
    values = X_test_df.to_numpy(dtype=float).ravel()

    mean = values.mean()
    std = values.std(ddof=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(
        values,
        bins=15,
        density=True,
        color='#4C78A8',
        alpha=0.60,
        edgecolor='white',
        label='Histograma (general pruebas)',
    )

    x = np.linspace(values.min() - 0.5, values.max() + 0.5, 400)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    ax.plot(x, y, color='#F58518', linewidth=2.2, label='Curva de campana (normal)')

    ax.axvline(mean, color='#E45756', linewidth=2.4, linestyle='--', label=f'Promedio: {mean:.3f}')

    ax.set_title('Histograma de campana general (conjunto de pruebas)')
    ax.set_xlabel('Valor de caracteristicas (test)')
    ax.set_ylabel('Densidad')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


df = load_iris_dataframe()
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'

train_idx, test_idx = stratified_split(df, target_col=target, train_ratio=0.8, seed=42)

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print('=' * 65)
print('ANALISIS DE DATOS Y PARTICION 80/20')
print('=' * 65)
print(f'  Total de muestras : {len(df)}')
print(f'  Entrenamiento     : {len(train_df)} ({(len(train_df) / len(df)) * 100:.2f}%)')
print(f'  Pruebas           : {len(test_df)} ({(len(test_df) / len(df)) * 100:.2f}%)')
print(f'  Caracteristicas   : {features}')

print_class_distribution(df, target, 'Distribucion de clases (dataset completo)')
print_class_distribution(train_df, target, 'Distribucion de clases (entrenamiento)')
print_class_distribution(test_df, target, 'Distribucion de clases (pruebas)')

output_dir = Path(__file__).resolve().parent
scatter_path = output_dir / 'iris_separacion.png'
confusion_path = output_dir / 'iris_matriz_confusion.png'
posterior_path = output_dir / 'iris_probabilidades.png'
test_histogram_path = output_dir / 'iris_campana_promedio_pruebas.png'
fp_fn_path = output_dir / 'iris_falsos_positivos_negativos.png'

generate_feature_scatter(df, target, scatter_path)
generate_test_bell_histogram_with_mean(X_test, test_histogram_path)
print(f'\nSe genero la visualizacion: {scatter_path.name}')
print(f'Se genero la visualizacion: {test_histogram_path.name}')

model = GaussianNaiveBayes()
model.fit(X_train, y_train)

sample_row = test_df.iloc[0]
sample = {f: sample_row[f] for f in features}
real_class = sample_row[target]

pred_class, posteriors = model.predict_with_details(sample, verbose=True)

print('\n' + '=' * 65)
print('CLASE PARA LOS PARAMETROS DE ENTRADA')
print('=' * 65)
print(f'  Clase predicha: {pred_class}')
print(f'  Clase real    : {real_class}')

preds = model.predict(X_test)
aciertos = sum(p == r for p, r in zip(preds, y_test))
fallos = len(y_test) - aciertos
accuracy = aciertos / len(y_test)
error_rate = fallos / len(y_test)

print('\n' + '=' * 65)
print('RESULTADO EN SUBCONJUNTO DE PRUEBAS')
print('=' * 65)
print(f'  Total pruebas: {len(y_test)}')
print(f'  Aciertos     : {aciertos}')
print(f'  Fallos       : {fallos}')
print(f'  Exactitud    : {accuracy:.4f} ({accuracy * 100:.2f}%)')
print(f'  Error        : {error_rate:.4f} ({error_rate * 100:.2f}%)')

generate_confusion_matrix_plot(y_test.tolist(), preds, model.classes, confusion_path)
generate_posterior_plot(posteriors, posterior_path)
generate_fp_fn_plot(y_test.tolist(), preds, model.classes, fp_fn_path)
print(f'Se genero la visualizacion: {confusion_path.name}')
print(f'Se genero la visualizacion: {posterior_path.name}')
print(f'Se genero la visualizacion: {fp_fn_path.name}')