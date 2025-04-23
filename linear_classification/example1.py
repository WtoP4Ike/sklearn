import numpy as np
from sklearn.linear_model import LogisticRegression

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
