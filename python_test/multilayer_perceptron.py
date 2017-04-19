# Import MNIST data
import tensorflow as tf

# Parameters
learning_rate = 0.25
training_epochs = 100

# Network Parameters
n_hidden_1 = 2
n_input = 2 
n_classes = 1 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.sigmoid(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    out_layer = tf.sigmoid(out_layer)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable([[-0.2, 0.1],[-0.1, 0.3]]),
    'out': tf.Variable([[0.2], [0.3]])
}
biases = {
    'b1': tf.Variable([0.1, 0.1]),
    'out': tf.Variable([0.2])
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

squared_deltas = 0.5 * tf.square(pred - y)
loss_function = tf.reduce_sum(squared_deltas)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_function)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.0001).minimize(loss_function)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        batch_x = [[ 0.1, 0.9 ]]
        batch_y = [[ 0.9 ]]

        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0].eval(sess))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[1].eval(sess))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[2].eval(sess))
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[3].eval(sess))
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[4].eval(sess))
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[5].eval(sess))
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[6].eval(sess))
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[7].eval(sess))
        print(0.9 - sess.run(pred, feed_dict={ x: batch_x })[0][0])
        _, c = sess.run([optimizer, loss_function], feed_dict={ x: batch_x, y: batch_y })

        # Compute average loss
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
