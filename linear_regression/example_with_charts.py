"""
// 
// Автор: WtoP4Ike wtop4ike@gmail.com
// Copyright (c) 2025, WTOP4IKE
// Лицензия: GNU General Public License v3.0 - https://www.gnu.org/licenses/gpl-3.0.html
// Версия: 1.0.0
// Сайт: https://t.me/wtop4ike
// 
"""

from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [3]])
y = np.array([8, 10])

model = LinearRegression()
model.fit(X, y)

X_new = np.array([[4]])
prediction = model.predict(X_new)
print(prediction)

import matplotlib.pyplot as plt

plt.scatter(X, y, color='blue', label='Данные')
plt.plot(X, model.predict(X), color='red', label='Регрессия')
plt.scatter(4, 11, color='green', label='Предсказание')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
