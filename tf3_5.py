#两层简单神经网络（全连接）

import tensorflow as tf

#定义输入和输出参数
#用placeholder定义输入（sess.run喂多组数据）
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random.normal([3, 1], stddev=1, seed=1))

#定义向前传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#调用会话计算结果
with tf.compat.v1.Session() as sess:
	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)
	print("the result of tf3_5.py is:\n", sess.run(y, feed_dict={x: [[0.7, 0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
	print("w1:\n", sess.run(w1))
	print("w2:\n", sess.run(w2))

