import tensorflow as tf
import multiprocessing as mp

core_num = mp.cpu_count()
print (core_num)
config = tf.compat.v1.ConfigProto(
  inter_op_parallelism_threads=core_num,
  intra_op_parallelism_threads=core_num)
sess = tf.compat.v1.Session(config=config)

hello = tf.compat.v2.constant('hello, tensorflow!')
print (sess.run(hello))

a = tf.compat.v2.constant(10)
b = tf.compat.v2.constant(32)
print (sess.run(a + b))