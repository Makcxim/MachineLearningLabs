import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Сохранение без вывода на экран
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Настройки для визуализации
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")
sns.set_palette("husl")

# 1. Загрузка и подготовка данных
print("Загрузка данных...")
df = pd.read_csv('../lab2/gtd_clean.csv', encoding='utf-8')

def clean_numeric(series):
    if series.dtype == 'object':
        result = series.str.extract(r'([\d,]+\.?\d*)')[0].str.replace(',', '')
        return pd.to_numeric(result, errors='coerce')
    return series


# полей мало, используем iyear, nkill, nwound
numeric_cols = ['iyear', 'nkill', 'nwound']
for col in numeric_cols:
    df[col] = clean_numeric(df[col])

# Удаляем пропуски
df = df.dropna(subset=numeric_cols + ['region_txt'])

# 2. Создание целевой переменной (Target)
# В примере с машинами создавали классы цен.
# Здесь мы выберем Топ-4 региона, чтобы сделать задачу классификации на 4 класса (как Budget, Mid, Premium, Luxury)
top_4_regions = df['region_txt'].value_counts().head(4).index.tolist()
print(f"Выбранные классы (регионы): {top_4_regions}")

# Фильтруем датасет, оставляем только эти 4 региона
df_clean = df[df['region_txt'].isin(top_4_regions)].copy()

# Признаки для классификации (Features)
features = ['nkill', 'nwound', 'iyear']
target = 'region_txt'

print("\nИнформация о данных для классификации:")
print(f"Размерность: {df_clean.shape}")
print(f"Классы:\n{df_clean[target].value_counts()}")
print("\nСтатистика по признакам:")
print(df_clean[features].describe())

# 3. Визуализация распределения классов (Аналог стр. 9 PDF)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Круговая диаграмма
class_dist = df_clean[target].value_counts()
axes[0, 0].pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', startangle=90)
axes[0, 0].set_title('Распределение атак по регионам (Классы)')

# Boxplot убитых по регионам
sns.boxplot(data=df_clean, x=target, y='nkill', ax=axes[0, 1], showfliers=False) # showfliers=False чтобы график был читаемым
axes[0, 1].set_title('Распределение убитых (nkill) по классам')
axes[0, 1].tick_params(axis='x', rotation=15)

# Boxplot раненых по регионам
sns.boxplot(data=df_clean, x=target, y='nwound', ax=axes[1, 0], showfliers=False)
axes[1, 0].set_title('Распределение раненых (nwound) по классам')
axes[1, 0].tick_params(axis='x', rotation=15)

# Boxplot годов по регионам
sns.boxplot(data=df_clean, x=target, y='iyear', ax=axes[1, 1])
axes[1, 1].set_title('Распределение атак по годам')
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Pairplot (Аналог стр. 10 PDF)
# Берем выборку, иначе pairplot зависнет на больших данных
sample_df = df_clean.sample(min(500, len(df_clean)), random_state=42)
sns.pairplot(sample_df[features + [target]], hue=target, diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Попарное распределение признаков по Регионам', y=1.02)
plt.savefig('pairplot_classes.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Подготовка к ML
X = df_clean[features]
y = df_clean[target]

# Масштабирование (StandardScaler) - критично для kNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nРазмеры выборок: Обучающая {X_train.shape}, Тестовая {X_test.shape}")

# Функция построения и отчета
def build_knn_classifier(k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nРезультаты для K = {k}:")
    print(f"Точность: {acc:.4f}")

    # Матрица ошибок (Аналог стр. 11-12 PDF)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=knn.classes_, yticklabels=knn.classes_)
    plt.title(f'Матрица ошибок для K = {k}')
    plt.ylabel('Истинный регион')
    plt.xlabel('Предсказанный регион')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_k{k}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return acc

# 5. Тесты для разных K (Аналог стр. 2 PDF)
print("=" * 50)
print("КЛАССИФИКАЦИЯ С РАЗНЫМИ K")
print("=" * 50)
for k in [3, 5, 7, 10]:
    build_knn_classifier(k)

# 6. Анализ Hold-out (влияние размера выборки) - Аналог стр. 13 (верх) PDF
test_sizes = [0.2, 0.3, 0.4]
k_values = range(1, 21, 2) # Нечетные шаги
results_holdout = []

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, ts in enumerate(test_sizes):
    accuracies = []
    for k in k_values:
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=ts, random_state=42, stratify=y)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr, y_tr)
        accuracies.append(knn.score(X_te, y_te))

    axes[i].plot(k_values, accuracies, marker='o')
    axes[i].set_title(f'Test size = {ts}')
    axes[i].set_xlabel('K соседей')
    axes[i].set_ylabel('Точность')
    axes[i].grid(True)

    # Лучшее K
    best_idx = np.argmax(accuracies)
    axes[i].axvline(x=k_values[best_idx], color='r', linestyle='--', label=f'Best K={k_values[best_idx]}')
    axes[i].legend()

plt.suptitle('Влияние размера тестовой выборки и K на точность', fontsize=14)
plt.savefig('holdout_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Cross-Validation - Аналог стр. 13 (низ) PDF
cv_folds = [5, 10]
k_range = range(1, 21)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, cv in enumerate(cv_folds):
    means, stds = [], []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=cv, scoring='accuracy')
        means.append(scores.mean())
        stds.append(scores.std())

    axes[i].plot(k_range, means, marker='o', label='Mean Accuracy')
    axes[i].fill_between(k_range, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2, label='std')
    axes[i].set_title(f'CV Folds = {cv}')
    axes[i].set_xlabel('K')
    axes[i].grid(True)
    best_k = k_range[np.argmax(means)]
    axes[i].axvline(best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    axes[i].legend()

plt.suptitle('Анализ кросс-валидации', fontsize=14)
plt.savefig('cv_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. GridSearchCV (Подбор параметров) - Аналог стр. 3 PDF
print("\nЗапуск Grid Search...")
param_grid = {
    'n_neighbors': range(3, 16),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_scaled, y)

print("Лучшие параметры:", grid.best_params_)
print(f"Лучшая точность CV: {grid.best_score_:.4f}")

# Финальный отчет
best_model = grid.best_estimator_
X_tr_fin, X_te_fin, y_tr_fin, y_te_fin = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
best_model.fit(X_tr_fin, y_tr_fin)
y_fin_pred = best_model.predict(X_te_fin)

print("\n" + "="*50)
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
print("="*50)
print(classification_report(y_te_fin, y_fin_pred))

# Интерактивное предсказание
def predict_region(nkill, nwound, iyear):
    # Данные нужно масштабировать тем же скалером!
    vec = np.array([[nkill, nwound, iyear]])
    vec_scaled = scaler.transform(vec)
    pred = best_model.predict(vec_scaled)[0]
    probs = best_model.predict_proba(vec_scaled)[0]

    print("\n" + "-"*30)
    print("ПРЕДСКАЗАНИЕ РЕГИОНА")
    print(f"Год: {iyear}, Убито: {nkill}, Ранено: {nwound}")
    print(f"-> Прогноз: {pred}")
    print("Вероятности:")
    for cls, prob in zip(best_model.classes_, probs):
        print(f"  {cls}: {prob:.2f}")

# Примеры (замените цифры на реалистичные для вашего сета)
print("\nПримеры работы модели:")
# 1. Мало жертв, современность (Скорее всего Западная Европа или Южная Америка в поздние годы)
predict_region(0, 2, 2015)

# 2. Много жертв, 80-е (Возможно Южная Америка или Азия)
predict_region(5, 10, 1985)

# 3. Очень много жертв, современность (Ближний Восток)
predict_region(50, 100, 2016)

print("\nЛабораторная работа выполнена. Графики сохранены.")
