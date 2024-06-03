import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# Generate dummy data
np.random.seed(0)
X = np.random.rand(4, 2)
y = np.random.rand(4, 2)

# Neural network parameters
input_size = 2
hidden_size = 3
output_size = 2
learning_rate = 0.1
epochs = 20

# Initialize weights
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to format matrices as plain text
def format_matrix_plain(matrix, name):
    rows = [" ".join(["{:.2f}".format(v) for v in row]) for row in matrix]
    matrix_str = "\n".join(rows)
    return "{} =\n{}".format(name, matrix_str)

fig, ax = plt.subplots(figsize=(14, 8))

def add_text(ax, text, position, fontsize=12):
    ax.text(position[0], position[1], text, horizontalalignment='left', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)

def draw_nn_diagram(ax):
    # Draw the input layer
    input_layer = [(0.1, 0.8), (0.1, 0.6)]
    input_labels = ['x1', 'x2']
    for i, (x, y) in enumerate(input_layer):
        circle = plt.Circle((x, y), 0.03, color='blue', fill=True)
        ax.add_artist(circle)
        ax.text(x, y, input_labels[i], horizontalalignment='center', verticalalignment='center', fontsize=10, color='white')
    
    # Draw the hidden layer
    hidden_layer = [(0.3, 0.9), (0.3, 0.7), (0.3, 0.5)]
    hidden_labels = ['h1', 'h2', 'h3']
    for i, (x, y) in enumerate(hidden_layer):
        circle = plt.Circle((x, y), 0.03, color='green', fill=True)
        ax.add_artist(circle)
        ax.text(x, y, hidden_labels[i], horizontalalignment='center', verticalalignment='center', fontsize=10, color='white')
    
    # Draw the output layer
    output_layer = [(0.5, 0.8), (0.5, 0.6)]
    output_labels = ['y1', 'y2']
    for i, (x, y) in enumerate(output_layer):
        circle = plt.Circle((x, y), 0.03, color='red', fill=True)
        ax.add_artist(circle)
        ax.text(x, y, output_labels[i], horizontalalignment='center', verticalalignment='center', fontsize=10, color='white')
    
    # Draw lines between layers with weight labels
    for i, (x0, y0) in enumerate(input_layer):
        for j, (x1, y1) in enumerate(hidden_layer):
            line = Line2D([x0, x1], [y0, y1], color='black', linestyle='-', linewidth=1)
            ax.add_line(line)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, 'W1_{}{}'.format(i+1, j+1), fontsize=8, color='black')

    for i, (x0, y0) in enumerate(hidden_layer):
        for j, (x1, y1) in enumerate(output_layer):
            line = Line2D([x0, x1], [y0, y1], color='black', linestyle='-', linewidth=1)
            ax.add_line(line)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, 'W2_{}{}'.format(i+1, j+1), fontsize=8, color='black')

    # Annotate layers
    ax.text(0.1, 0.85, 'Input Layer', horizontalalignment='center', verticalalignment='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.3, 0.95, 'Hidden Layer', horizontalalignment='center', verticalalignment='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.85, 'Output Layer', horizontalalignment='center', verticalalignment='center', fontsize=10, transform=ax.transAxes)

def animate(epoch):
    global W1, W2
    
    # Forward pass
    z1 = np.dot(X, W1)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2)
    y_pred = sigmoid(z2)
    
    # Compute loss
    loss = mse_loss(y, y_pred)
    
    # Backward pass (gradient descent)
    d_loss_y_pred = 2 * (y_pred - y) / y.size
    d_y_pred_z2 = y_pred * (1 - y_pred)
    d_z2_W2 = a1.T
    d_loss_W2 = np.dot(d_z2_W2, d_loss_y_pred * d_y_pred_z2)
    
    d_z2_a1 = W2.T
    d_a1_z1 = a1 * (1 - a1)
    d_z1_W1 = X.T
    d_loss_W1 = np.dot(d_z1_W1, (np.dot(d_loss_y_pred * d_y_pred_z2, d_z2_a1) * d_a1_z1))
    
    # Update weights
    W1 -= learning_rate * d_loss_W1
    W2 -= learning_rate * d_loss_W2
    
    # Clear previous plots
    ax.clear()
    
    # Prepare plain text strings
    ax.set_axis_off()
    
    add_text(ax, "Epoch {}".format(epoch+1), (0.01, 0.95))

    add_text(ax, format_matrix_plain(X, 'X'), (0.01, 0.80))
    add_text(ax, format_matrix_plain(W1, 'W1'), (0.15, 0.4))

    add_text(ax, "z1 = X * W1\n" + format_matrix_plain(z1, 'z1'), (0.3, 0.40))
    add_text(ax, "a1 = sigmoid(z1)\n" + format_matrix_plain(a1, 'a1'), (0.2, 0.15))
    
    add_text(ax, format_matrix_plain(W2, 'W2'), (0.40, 0.40))
    add_text(ax, "z2 = a1 * W2\n" + format_matrix_plain(z2, 'z2'), (0.60, 0.65))
    add_text(ax, "y_pred = sigmoid(z2)\n" + format_matrix_plain(y_pred, 'y_pred'), (0.60, 0.50))
    add_text(ax, format_matrix_plain(y, 'y'), (0.60, 0.35))
    add_text(ax, "Loss = {:.4f}".format(loss), (0.80, 0.20))
    
    # Draw the neural network diagram
    draw_nn_diagram(ax)
    
    return ax

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=epochs, repeat=True)

# Save as GIF with looping enabled
writer = animation.PillowWriter(fps=1)
ani.save('nn_training_plain.gif', writer=writer)

plt.show()
