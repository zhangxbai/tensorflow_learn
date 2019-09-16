# 设损失函数为（w+1)^2, 令w初值为5.反向传播就是求最优w，即求最小loss对应的w值
import tensorflow as tf

w = tf.Variable(tf.constant(5, dtype=tf.float32))

loss = tf.square(w+1)
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.compat.v1.Session() as sess:
	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)
	for i in range(40):
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print("After %s steps: w is %f, loss is %f." %(i, w_val, loss_val))
