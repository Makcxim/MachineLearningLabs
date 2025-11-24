import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Сохранение в файл без вывода на экран
import matplotlib.pyplot as plt
import seaborn as sns

# улучшения визуализации
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")
sns.set_palette("husl")

df = pd.read_csv('gtd_clean.csv', encoding='utf-8')

def clean_numeric(series):
    if series.dtype == 'object':
        result = series.str.extract(r'([\d,]+\.?\d*)')[0].str.replace(',', '')
        return pd.to_numeric(result, errors='coerce')
    return series


# полей мало, используем iyear, nkill, nwound
numeric_cols = ['iyear', 'nkill', 'nwound']
for col in numeric_cols:
    df[col] = clean_numeric(df[col])

# дополнительный столбец "Всего жертв", чтобы заполнить сетку графиков
df['Total Victims'] = df['nkill'] + df['nwound']

print("Основная информация о данных:")
print(df.info())
print(f"\nРазмерность данных: {df.shape}")

### Гистограммы распределения
# Создаем фигуру с несколькими subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Гистограмма Убитых (nkill)
axes[0,0].hist(df['nkill'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Распределение количества убитых (nkill)')
axes[0,0].set_xlabel('Убитые')
axes[0,0].set_ylabel('Количество инцидентов')
axes[0,0].set_yscale('log') # Лог-шкала, т.к. разброс в терроризме огромный

# 2. Гистограмма Раненых (nwound)
axes[0,1].hist(df['nwound'].dropna(), bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
axes[0,1].set_title('Распределение количества раненых (nwound)')
axes[0,1].set_xlabel('Раненые')
axes[0,1].set_ylabel('Количество инцидентов')
axes[0,1].set_yscale('log')

# 3. Гистограмма Лет (iyear)
axes[0,2].hist(df['iyear'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0,2].set_title('Распределение по годам (iyear)')
axes[0,2].set_xlabel('Год')
axes[0,2].set_ylabel('Количество инцидентов')

# 4. Гистограмма Всего жертв
axes[1,0].hist(df['Total Victims'].dropna(), bins=30, alpha=0.7, color='gold', edgecolor='black')
axes[1,0].set_title('Распределение общего числа жертв')
axes[1,0].set_xlabel('Жертвы (Убитые + Раненые)')
axes[1,0].set_ylabel('Количество')
axes[1,0].set_yscale('log')

# 5. Повтор Убитых (для заполнения сетки)
axes[1,1].hist(df['nkill'].dropna(), bins=30, alpha=0.7, color='violet', edgecolor='black')
axes[1,1].set_title('Распределение убитых (повтор)')
axes[1,1].set_xlabel('Убитые')
axes[1,1].set_ylabel('Количество')
axes[1,1].set_yscale('log')

# 6. Повтор Раненых (для заполнения сетки)
axes[1,2].hist(df['nwound'].dropna(), bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[1,2].set_title('Распределение раненых (повтор)')
axes[1,2].set_xlabel('Раненые')
axes[1,2].set_ylabel('Количество')
axes[1,2].set_yscale('log')

plt.tight_layout()
plt.savefig('histograms_detailed.png', dpi=300, bbox_inches='tight')
plt.close(fig)

### Графики "Ящик с усами" (Boxplot)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Boxplot Убитых
axes[0,0].boxplot(df['nkill'].dropna())
axes[0,0].set_title('Boxplot: Убитые (nkill)')
axes[0,0].set_ylabel('Человек')
axes[0,0].set_yscale('log') # Используем лог шкалу, иначе ящики сплющит

# Boxplot Раненых
axes[0,1].boxplot(df['nwound'].dropna())
axes[0,1].set_title('Boxplot: Раненые (nwound)')
axes[0,1].set_ylabel('Человек')
axes[0,1].set_yscale('log')

# Boxplot Годов
axes[0,2].boxplot(df['iyear'].dropna())
axes[0,2].set_title('Boxplot: Годы (iyear)')
axes[0,2].set_ylabel('Год')

# Boxplot Всего жертв
axes[1,0].boxplot(df['Total Victims'].dropna())
axes[1,0].set_title('Boxplot: Всего жертв')
axes[1,0].set_ylabel('Человек')
axes[1,0].set_yscale('log')

# Boxplot Убитых (повтор)
axes[1,1].boxplot(df['nkill'].dropna())
axes[1,1].set_title('Boxplot: Убитые (дубль)')
axes[1,1].set_ylabel('Человек')
axes[1,1].set_yscale('log')

# Boxplot Раненых (повтор)
axes[1,2].boxplot(df['nwound'].dropna())
axes[1,2].set_title('Boxplot: Раненые (дубль)')
axes[1,2].set_ylabel('Человек')
axes[1,2].set_yscale('log')

plt.tight_layout()
plt.savefig('boxplots_detailed.png', dpi=300, bbox_inches='tight')
plt.close(fig)

### Сравнительные Boxplot по Регионам
# Берем только топ-5 регионов, чтобы график не был кашей
top_regions = df['region_txt'].value_counts().head(5).index
df_top = df[df['region_txt'].isin(top_regions)]

fig = plt.figure(figsize=(14, 8))

# Убитые по Регионам
plt.subplot(2, 2, 1)
sns.boxplot(data=df_top, x='region_txt', y='nkill')
plt.title('Распределение убитых по Топ-5 регионам')
plt.xticks(rotation=45)
plt.yscale('log')

# Раненые по Регионам
plt.subplot(2, 2, 2)
sns.boxplot(data=df_top, x='region_txt', y='nwound')
plt.title('Распределение раненых по Топ-5 регионам')
plt.xticks(rotation=45)
plt.yscale('log')

# Всего жертв по Регионам
plt.subplot(2, 2, 3)
sns.boxplot(data=df_top, x='region_txt', y='Total Victims')
plt.title('Всего жертв по Топ-5 регионам')
plt.xticks(rotation=45)
plt.yscale('log')

# Годы по Регионам
plt.subplot(2, 2, 4)
sns.boxplot(data=df_top, x='region_txt', y='iyear')
plt.title('Распределение инцидентов по годам и регионам')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('boxplot_by_region.png', dpi=300, bbox_inches='tight')
plt.close(fig)

### Countplot для категориальных признаков
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Распределение по Топ-10 Регионам
top_10_regions = df['region_txt'].value_counts().head(10)
axes[0,0].bar(top_10_regions.index, top_10_regions.values, color='lightblue')
axes[0,0].set_title('Топ-10 Регионов по количеству атак')
axes[0,0].set_ylabel('Количество атак')
axes[0,0].tick_params(axis='x', rotation=45)

# Распределение по Регионам (Pie)
# Берем топ-6 для читаемости пирога
region_counts = df['region_txt'].value_counts().head(6)
axes[0,1].pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%', startangle=90)
axes[0,1].set_title('Доли атак по Топ-6 регионам')

# Распределение по Топ-8 годам
year_counts = df['iyear'].value_counts().head(8).sort_index()
axes[1,0].bar(year_counts.index.astype(str), year_counts.values, color='lightgreen')
axes[1,0].set_title('Топ-8 лет с максимальной активностью')
axes[1,0].set_ylabel('Количество')
axes[1,0].tick_params(axis='x', rotation=45)

# Еще раз регионы (для заполнения)
axes[1,1].bar(top_10_regions.index, top_10_regions.values, color='gold')
axes[1,1].set_title('Распределение по регионам (дубль)')
axes[1,1].set_ylabel('Количество')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
plt.close(fig)

### Детальный countplot с Seaborn
fig = plt.figure(figsize=(15, 10))

# Countplot по Регионам (вертикальный)
plt.subplot(2, 2, 1)
sns.countplot(data=df, y='region_txt', order=df['region_txt'].value_counts().index[:10])
plt.title('Топ-10 Регионов (Seaborn)')
plt.xlabel('Количество')

# Countplot по Годам (выборка)
plt.subplot(2, 2, 2)
# Берем последние 10 лет
recent_df = df[df['iyear'] >= 2008]
sns.countplot(data=recent_df, x='iyear')
plt.title('Атаки за последние 10 лет датасета')
plt.xticks(rotation=45)

# Количество атак по Регионам (горизонтальный)
plt.subplot(2, 1, 2)
top_15_regions = df['region_txt'].value_counts().head(15)
sns.barplot(x=top_15_regions.values, y=top_15_regions.index, palette='viridis')
plt.title('Топ-15 Регионов по количеству атак')
plt.xlabel('Количество атак')

plt.tight_layout()
plt.savefig('countplots_detailed.png', dpi=300, bbox_inches='tight')
plt.close(fig)

### Scatter plots с настройкой цвета
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Выборка для scatter plots (иначе будет тормозить, если данных много)
sample_df = df.sample(min(1000, len(df)))

# Убитые vs Раненые (цвет по Году)
scatter1 = axes[0,0].scatter(sample_df['nkill'], sample_df['nwound'],
                            c=sample_df['iyear'], alpha=0.6, cmap='viridis')
axes[0,0].set_xlabel('Убитые (nkill)')
axes[0,0].set_ylabel('Раненые (nwound)')
axes[0,0].set_title('Убитые vs Раненые (цвет - год)')
plt.colorbar(scatter1, ax=axes[0,0], label='Год')
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')

# Убитые vs Всего жертв (размер по iyear)
sizes = (sample_df['iyear'] - 1970) * 2  # Масштабируем размер
scatter2 = axes[0,1].scatter(sample_df['nkill'], sample_df['Total Victims'],
                            s=sizes, alpha=0.6, c=sample_df['nwound'], cmap='plasma')
axes[0,1].set_xlabel('Убитые')
axes[0,1].set_ylabel('Всего жертв')
axes[0,1].set_title('Убитые vs Всего жертв\n(размер - год, цвет - раненые)')
plt.colorbar(scatter2, ax=axes[0,1], label='Раненые')
axes[0,1].set_xscale('log')
axes[0,1].set_yscale('log')

# Год vs Убитые
axes[1,0].scatter(sample_df['iyear'], sample_df['nkill'], alpha=0.6, color='red')
axes[1,0].set_xlabel('Год (iyear)')
axes[1,0].set_ylabel('Убитые (nkill)')
axes[1,0].set_title('Убитые vs Год')

# Раненые vs Убитые (Regplot)
# Ограничим данные чтобы линия регрессии имела смысл без лог-шкалы здесь
clean_sample = sample_df[(sample_df['nkill'] < 100) & (sample_df['nwound'] < 100)]
sns.regplot(data=clean_sample, x='nkill', y='nwound',
           scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=axes[1,1])
axes[1,1].set_xlabel('Убитые')
axes[1,1].set_ylabel('Раненые')
axes[1,1].set_title('Раненые vs Убитые (с линией регрессии, n < 100)')

plt.tight_layout()
plt.savefig('scatter_advanced.png', dpi=300, bbox_inches='tight')
plt.close(fig)

### Pairplot для многомерного анализа
# Ключевые числовые признаки
pairplot_cols = ['iyear', 'nkill', 'nwound', 'Total Victims']

# Создаем pairplot с раскраской по Региону (Топ-4 для наглядности)
top_4_regions = df['region_txt'].value_counts().head(4).index
pairplot_df = df[df['region_txt'].isin(top_4_regions)][pairplot_cols + ['region_txt']].dropna()

# Выборка
if len(pairplot_df) > 500:
    pairplot_df = pairplot_df.sample(500, random_state=42)

g = sns.pairplot(pairplot_df, hue='region_txt', diag_kind='hist',
                plot_kws={'alpha':0.6, 's':30}, height=2.5)
g.fig.suptitle('Pairplot: Анализ характеристик терроризма', y=1.02)
plt.savefig('pairplot_analysis.png', dpi=300, bbox_inches='tight')
plt.close(g.fig)

### Матрица корреляций с Heatmap
fig = plt.figure(figsize=(12, 10))

# Вычисляем матрицу
correlation_matrix = df[['nkill', 'nwound', 'iyear', 'Total Victims']].corr()

# Маска для треугольника
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Heatmap
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
           center=0, square=True, linewidths=0.5, fmt='.3f',
           cbar_kws={"shrink": .8})

plt.title('Матрица корреляций (GTD Data)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap_detailed.png', dpi=300, bbox_inches='tight')
plt.close(fig)

print("Сильнейшие корреляции (> 0.7):")
strong_corr = correlation_matrix.unstack().sort_values(ascending=False)
strong_corr = strong_corr[(strong_corr < 1.0) & (strong_corr > 0.3)]
print(strong_corr)

### Violin plot для сравнения распределений
fig = plt.figure(figsize=(15, 10))

# Violin plot: Убитые по Топ-5 регионам
plt.subplot(2, 2, 1)
df_violin = df[df['region_txt'].isin(top_regions)]
# Обрежем экстремальные выбросы для визуализации
df_violin_cut = df_violin[df_violin['nkill'] < 20]
sns.violinplot(data=df_violin_cut, x='region_txt', y='nkill')
plt.title('Распределение убитых по регионам (nkill < 20)')
plt.xticks(rotation=45)

# Violin plot: Раненые по Топ-5 регионам
plt.subplot(2, 2, 2)
df_violin_cut2 = df_violin[df_violin['nwound'] < 20]
sns.violinplot(data=df_violin_cut2, x='region_txt', y='nwound')
plt.title('Распределение раненых по регионам (nwound < 20)')
plt.xticks(rotation=45)

# Swarm plot (на маленькой выборке)
plt.subplot(2, 2, 3)
sample_swarm = df_top.sample(100) # Очень маленькая выборка, иначе Swarm виснет
sns.swarmplot(data=sample_swarm, x='region_txt', y='nkill', size=4)
plt.title('Swarm plot: Убитые по регионам (sample 100)')
plt.xticks(rotation=45)

# Strip plot
plt.subplot(2, 2, 4)
sns.stripplot(data=df_violin_cut, x='region_txt', y='nkill',
             alpha=0.7, jitter=True)
plt.title('Strip plot: Убитые по регионам')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('special_plots.png', dpi=300, bbox_inches='tight')
plt.close(fig)

### FacetGrid для сложной визуализации
# FacetGrid по Топ-4 регионам
top_4_list = list(top_4_regions)
g = sns.FacetGrid(df[df['region_txt'].isin(top_4_list)],
                  col='region_txt', col_wrap=2, height=5, aspect=1.2)
# Ограничим оси для наглядности
g.map_dataframe(sns.scatterplot, x='nkill', y='nwound', alpha=0.7)
g.set(xlim=(0, 100), ylim=(0, 100))
g.set_axis_labels('Убитые', 'Раненые')
g.set_titles('Регион: {col_name}')
g.fig.suptitle('Убитые vs Раненые по Регионам', y=1.02)
plt.savefig('facetgrid_analysis.png', dpi=300, bbox_inches='tight')
plt.close(g.fig)

### Jointplot для детального анализа
# Jointplot: Убитые vs Раненые
clean_joint = df[(df['nkill'] < 100) & (df['nwound'] < 100)]
j1 = sns.jointplot(data=clean_joint, x='nkill', y='nwound',
                  kind='scatter', alpha=0.6, height=8)
j1.fig.suptitle('Jointplot: Убитые vs Раненые (n<100)', y=1.02)
plt.savefig('jointplot_hp_price.png', dpi=300, bbox_inches='tight')
plt.close(j1.fig)

# Jointplot с регрессией: Убитые vs Всего жертв
j2 = sns.jointplot(data=clean_joint, x='nkill', y='Total Victims',
                  kind='reg', height=8)
j2.fig.suptitle('Jointplot: Убитые vs Всего жертв (reg)', y=1.02)
plt.savefig('jointplot_hp_acceleration.png', dpi=300, bbox_inches='tight')
plt.close(j2.fig)

print("da end")