import numpy as np
from sklearn.linear_model import LogisticRegression

data = np.array([
    [12, 85, 1], [14, 88, 1], [16, 82, 1], [18, 79, 1], [20, 84, 1],
    [22, 35, 0], [24, 40, 0], [26, 38, 0], [28, 42, 0], [30, 37, 0],
    [13, 70, 1], [15, 86, 1], [17, 65, 1], [19, 83, 1], [21, 81, 1],
    [23, 39, 0], [25, 41, 0], [27, 36, 0], [29, 43, 0], [31, 38, 0]
])

X = data[:, :2]  # Первые два столбца - признаки
y = data[:, 2]   # Третий столбец - то, что предсказываем

model = LogisticRegression()
model.fit(X, y)

X_new = np.array([[13, 88]])
prediction = model.predict(X_new)
print(prediction)
