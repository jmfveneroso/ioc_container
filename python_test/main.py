# ================
# Example 1
# ================
# import tensorflow as tf
#
# # Linear model.
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# x = tf.placeholder(tf.float32)
# linear_model = W * x + b
# y = tf.placeholder(tf.float32)
# 
# # Loss function.
# squared_deltas = 0.5 * tf.square(linear_model - y)
# loss = tf.reduce_sum(squared_deltas)
# 
# # Optimizer.
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# 
# # Training data.
# x_train = [1,2,3,4]
# y_train = [0,-1,-2,-3]
# 
# # Run session.
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# for i in range(1000):
#   sess.run(train, { x: x_train, y: y_train })
# 
# curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

# ================
# Example 2
# ================
# import tensorflow as tf
# import numpy as np
# 
# # Declare list of features, we only have one real-valued feature
# def model(features, labels, mode):
#   # Build a linear model and predict values
#   W = tf.get_variable("W", [1], dtype=tf.float64)
#   b = tf.get_variable("b", [1], dtype=tf.float64)
#   y = W * features['x'] + b
# 
#   # Loss sub-graph
#   loss = tf.reduce_sum(tf.square(y - labels))
# 
#   # Training sub-graph
#   global_step = tf.train.get_global_step()
#   optimizer = tf.train.GradientDescentOptimizer(0.01)
#   train = tf.group(optimizer.minimize(loss),
#                    tf.assign_add(global_step, 1))
# 
#   # ModelFnOps connects subgraphs we built to the
#   # appropriate functionality.
#   return tf.contrib.learn.ModelFnOps(
#     mode=mode, predictions=y,
#     loss=loss,
#     train_op=train)
# 
# # features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
# # estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
# 
# estimator = tf.contrib.learn.Estimator(model_fn=model)
# 
# x = np.array([1., 2., 3., 4.])
# y = np.array([0., -1., -2., -3.])
# input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)
# estimator.fit(input_fn=input_fn, steps=1000)
# print(estimator.evaluate(input_fn=input_fn, steps=10))

# ================
# Example 3
# ================
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())
y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
