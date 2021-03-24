import tensorflow as tf

@tf.function
def task():
    x = [[0,0,0,], [0,0,0]]
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]

    x = tf.expand_dims(x, 0)
    t1 = tf.expand_dims(t1, 0)
    t2 = tf.expand_dims(t2, 0)

    x = tf.concat([x, t1], 0)
    x = tf.concat([x, t2], 0)

    return x[1:, :, :]

    # print(x)
    # print(x[1:, :, :])


x = task()
print(x)