import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)
y = x**2

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, verbose=0)
plt.plot(x, y, label='ground truth')
plt.plot(x, model.predict(x), label='predicted')
plt.legend()
plt.show()

x_new = np.array([-2, -1, 0, 1, 2])
y_real = x_new ** 2
x_new = x_new.reshape(-1, 1)
y_pred = model.predict(x_new)
list(map('{:.2f}%'.format, x))
for i in range(len(x_new)):
    print(f"{x_new[i][0]} | {y_pred[i][0]} | {y_real[i]}")
