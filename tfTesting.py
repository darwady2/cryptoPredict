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

init = tf.global_variables_initializer()
sess.run(init)

#print(sess.run(linear_model, {x: [1, 2, 3, 4]})) # Outputs [ 0.  0.30000001  0.60000002  0.90000004]
