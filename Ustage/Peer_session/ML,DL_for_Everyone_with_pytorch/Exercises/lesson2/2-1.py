import numpy as np
import matplotlib.pyplot as plt
import csv

from numpy.lib.function_base import average

x_data = []
y_data = []
with open('kc_house_data.csv','r') as f :
    reader = csv.DictReader(f)
    idxCounter = 0
    for row in reader:
        if('e' not in row['sqft_living'] and 'e' not in row['price']):
            x_data.append(int(row['price']/500))
            y_data.append(int(row['sqft_living']))
            idxCounter += 1
        if idxCounter > 200:
            break


# our model for the forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# List of weights/Mean square Error (Mse) for each input
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    # Print the weights and initialize the lost
    print("w=", w)
    l_sum = 0

    for x_val, y_val in zip(x_data, y_data):
        # For each input and output, calculate y_hat
        # Compute the total loss and add to the total error
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)
    # Now compute the Mean squared error (mse) of each
    # Aggregate the weight/mse from this run
    print("MSE=", l_sum / len(x_data))
    w_list.append(w)
    mse_list.append(l_sum / len(x_data))

# Plot it all
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
