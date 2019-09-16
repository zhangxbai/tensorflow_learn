#酸奶成本 1元， 利润9元
#预测少了损失过大

import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
PROFIT = 9
COST = 1
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10-0.05)] for(x1, x2) in X]

#1 定义神经网络的输入， 参数、输出， 定义前向传播的过程
x = tf.compat.v1.placeholder(tf.float32, shape=(None,2))
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None,1))
w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

#2 定义损失函数及反向传播方法

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#3 生成会话，训练STEPS轮
with tf.compat.v1.Session() as sess:
	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)
	STEPS = 20000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = (i*BATCH_SIZE) % 32 +BATCH_SIZE
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})

		if i % 500 == 0:
			print("After %d training steps, w1 is:" %(i))
			print(sess.run(w1), "\n")
	print("Final w1 is :\n", sess.run(w1))
