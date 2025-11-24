import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dt = np.dtype("f8, U50, f8, f8")
data = np.genfromtxt('gtd_clean.csv', delimiter=',', dtype=dt)
print('успех загрузки')

years = np.array([row[0] for row in data])
regions = np.array([row[1] for row in data])
killed = np.array([row[2] for row in data])
wounded = np.array([row[3] for row in data])

# находим все регионы
unique_regions = np.unique(regions)
print(f"Найдено регионов: {len(unique_regions)}")
print(unique_regions)

# палитра 'tab20', чтобы цвета отличались
colors = cm.tab20(np.linspace(0, 1, len(unique_regions)))

# рисуем один плот для всех регионов сразу
def plot_all_regions(x_data, y_data, xlabel, ylabel, title, fig_num, fig_filename):
    plt.figure(fig_num, figsize=(12, 7))

    for i, region_name in enumerate(unique_regions):
        mask = (regions == region_name)

        # в x_data и y_data берем нужные по маске данные и рисуем
        # s=15 - размер точки, alpha=0.6 - прозрачность
        plt.plot(x_data[mask], y_data[mask],
                 linestyle='', marker='o', markersize=4,
                 color=colors[i], alpha=0.6, label=region_name)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.3)

    # легенда за графиком тк 12 регионов это много
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()  # легенда не обрезалась при сохранении
    plt.savefig(fig_filename)


# 1: Год vs Убитые
plot_all_regions(years, killed, 'Год', 'Убитые', '1. Динамика убитых по всем регионам', 1, 'year_killed.png')

# 2: Год vs Раненые
plot_all_regions(years, wounded, 'Год', 'Раненые', '2. Динамика раненых по всем регионам', 2, 'year_wounded.png')

# 3: Убитые vs Раненые
plot_all_regions(killed, wounded, 'Убитые', 'Раненые', '3. Корреляция жертв', 3, 'killed_wounded.png')

plt.show()