import tensorflow as tf
import numpy as np

# Define the function that takes a NumPy array as input
def my_function(array):
    return array * 2  # Example function that doubles the values

# Define the shape of the symbolic tensor
shape = (3, 3)

# Create a graph context
graph = tf.compat.v1.Graph()
with graph.as_default():
    # Create a symbolic tensor placeholder
    symbolic_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=shape)

    # Define a TensorFlow operation using tf.py_function to call the function
    operation_tensor = tf.py_function(my_function, [symbolic_tensor], tf.float32)

print(type(operation_tensor), type(symbolic_tensor))
# Create a TensorFlow session
with tf.compat.v1.Session(graph=graph) as sess:
    # Define a feed dictionary to provide values for the symbolic tensor
    feed_dict = {symbolic_tensor: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)}

    # Run the operation with the provided values
    result = sess.run(operation_tensor, feed_dict=feed_dict)

# Print the result
print("Result:")
print(result)
