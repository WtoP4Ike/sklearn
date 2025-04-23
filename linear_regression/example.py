from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [3]])
y = np.array([8, 10])

model = LinearRegression()
model.fit(X, y)

X_new = np.array([[4]])
prediction = model.predict(X_new)
print(prediction)
