import matplotlib.pyplot as plt
import numpy as np

#define the function
f = lambda x : np.tanh(x)

#set values
x = np.linspace(0,100)

#calculate the values of the function at the given points
y =  f(x)
y2 = y / 4
# y and y2 are now arrays which we can plot

#plot the resulting arrays
fig, ax = plt.subplots(1,3, figsize=(10,3))

ax[0].set_title("plot y = log2(x)")
ax[0].plot(x,y) # .. "plot f"

ax[1].set_title("plot y = f(y) / 4")
ax[1].plot(x,y2) # .. "plot logarithm of f"

plt.show()