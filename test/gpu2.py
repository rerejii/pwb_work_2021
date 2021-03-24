import tensorflow as tf

x = tf.constant(1, name='x')
y = tf.constant(2, name='y')

add_op = tf.add(x, y)

print(add_op)

input("2: Press Enter to continue...")