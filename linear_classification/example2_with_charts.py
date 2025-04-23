"""
// 
// Автор: WtoP4Ike wtop4ike@gmail.com
// https://github.com/WtoP4Ike/sklearn/blob/main/linear_regression/example_with_charts.py
//
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = np.array([
    [12, 85, 1], [14, 88, 1], [16, 82, 1], [18, 79, 1], [20, 84, 1],
    [22, 35, 0], [24, 40, 0], [26, 38, 0], [28, 42, 0], [30, 37, 0],
    [13, 70, 1], [15, 86, 1], [17, 65, 1], [19, 83, 1], [21, 81, 1],
    [23, 39, 0], [25, 41, 0], [27, 36, 0], [29, 43, 0], [31, 38, 0]
])

X = data[:, :2]  # Первые два столбца - признаки
y = data[:, 2]   # Третий столбец - метки

model = LogisticRegression()
model.fit(X, y)

X_new = np.array([[13, 88]])
prediction = model.predict(X_new)
print(prediction)


ax = plt.subplot(1, 2, 2, projection='3d')
xx3d, yy3d = np.meshgrid(
    np.linspace(X[:,0].min(), X[:,0].max(), 20),
    np.linspace(X[:,1].min(), X[:,1].max(), 20)
)
zz3d = np.zeros_like(xx3d)
ax.plot_surface(xx3d, yy3d, zz3d, alpha=0.3, color='gray')
ax.scatter(X[:,0], X[:,1], y, c=y, cmap='coolwarm', edgecolor='k')
ax.set_title('3D-визуализация')
ax.set_xlabel('Температура, °C')
ax.set_ylabel('Влажность, %')
ax.view_init(30, -70)

plt.tight_layout()
plt.show()
