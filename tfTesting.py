from __future__ import print_function
import tensorflow as tf


node1 = tf.constant(3.0) #dtype=tf.float32
node2 = tf.constant(4.0) # also tf.float32 implicitly

sess = tf.Session() # This starts a TF session within the program.
#print(sess.run([node1,node2])) #Runs the session and prints the output. Just printing the nodes will show the details of the nodes themselves but not the constants.

node3 = tf.add(node1, node2) #Makes a new node and adds the values of nodes 1 and 2.
#print("node3:", node3) # This just prints the details of node3.
#print("sess.run(node3)", sess.run(node3)) #This actually runs node3 in a session and returns its value.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a,b)

#print(sess.run(adder_node, {a: 3, b: 4.5})) # defines the parameters, outputs 7.5
#print(sess.run(adder_node, {a: [1, 3], b: [2, 4]})) # defines the parameters, outputs [3., 7.]

add_and_triple = adder_node * 3
#print(sess.run(add_and_triple, {a: 3, b: 4.5})) # 22.5.

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

init = tf.global_variables_initializer() # This is needed to initialize the variables, before you can run the program.

#print(sess.run(linear_model, {x: [1, 2, 3, 4]})) # Outputs [ 0.  0.30000001  0.60000002  0.90000004]

y = tf.placeholder(tf.float32) #This is a placeholder for the training data. We'll input what we want the values to be.
squared_deltas = tf.square(linear_model - y) #This is the squared difference between the output of our linear model and our training data points.
loss = tf.reduce_sum(squared_deltas) #This sums up the squared deltas into one scalar -  a coefficient that gives an indicator for how far off our model is.

optimizer = tf.train.GradientDescentOptimizer(0.01) # This creates an optimizer, which we'll use to adjust the coefficients until they become correct.
train = optimizer.minimize(loss) #This tells our optimizer to minimize the value of 'loss'.

sess.run(init) # Reinitializes the variables.
for i in range(1000): #Runs our optimizer 1000 times over the values of x vs. the values of y, and adjusts the coefficient until the input of x will result in the output of y.
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b])) # Output: W = -1, b = 1.
