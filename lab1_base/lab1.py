import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

dt = np.dtype("f8, f8, f8, f8, U30")
data = np.genfromtxt('iris.data', delimiter=',', dtype=dt)

print(data)


# Данные из отдельных столбцов
sepal_length = [] # Sepal Length
sepal_width = []  # Sepal Width
petal_length = [] # Petal Length
petal_width = []  # Petal Width

# Выполняем обход всей коллекции data2
for dot in data:
    sepal_length.append(dot[0])
    sepal_width.append(dot[1])
    petal_length.append(dot[2])
    petal_width.append(dot[3])

# Строим графики по проекциям данных
# Учитываем, что каждые 50 типов ирисов идут последовательно
plt.figure(1)
setosa, = plt.plot(sepal_length[:50], sepal_width[:50], 'ro', label='Setosa')
versicolor, = plt.plot(sepal_length[50:100], sepal_width[50:100], 'g^', label='Versicolor')
virginica, = plt.plot(sepal_length[100:150], sepal_width[100:150], 'bs', label='Verginica')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.show()
