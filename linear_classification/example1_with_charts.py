import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = np.array([
    [12, 1], [14, 1], [16, 1], [18, 1], [20, 1],
    [23, 0], [25, 0], [27, 0], [29, 0], [31, 0]
])

X = data[:, :1]  # Первый столб столбца - признак
y = data[:, 1]   # Второй столб - метки

model = LogisticRegression()
model.fit(X, y)

X_new = np.array([[24]])
prediction = model.predict(X_new)
print(prediction)

plt.figure(figsize=(12, 4))
plt.scatter(X, y, c=y, cmap='coolwarm', edgecolor='k')
plt.axvline(-model.intercept_[0]/model.coef_[0][0], 
           color='black', linestyle='--', label='Граница решения')
plt.title('Классификация по температуре')
plt.xlabel('Температура, °C')
plt.ylabel('Дождь (1 - да, 0 - нет)')
plt.legend()
plt.show()
