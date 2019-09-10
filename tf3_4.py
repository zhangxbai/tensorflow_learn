import tensorflow as tf

#定义输入和输出参数
#用placeholder实现输入定义（sess.run()喂一组数据）
x = tf.compat.v1.placeholder(tf.float32, shape=(1, 2))
w1 = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random.normal([3, 1], stddev=1, seed=1))


#定义向前传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


#用会话计算结果

with tf.compat.v1.Session() as sess:
	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)
	print("y in tf3_4 is:\n", sess.run(y, feed_dict={x: [[0.7,0.5]]}))
