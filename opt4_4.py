# 设损失函数为（w+1)^2, 令w初值为5.反向传播就是求最优w，即求最小loss对应的w值
import tensorflow as tf
LEARNING_RATE_BASE = 0.1 #最初学习率
LEARNING_RATE_DECAY = 0.9 #学习衰减率
LEARNING_RATE_STEP = 1 #喂入多少轮BATCH_SIZE后， 更新一次学习率， 一般设为：总样本/BATCH_SIZE

global_step = tf.Variable(0, trainable=False) #运行轮数计数器，不被训练
learning_rate = tf.compat.v1.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)

w = tf.Variable(tf.constant(5, dtype=tf.float32))

loss = tf.square(w+1)
train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.compat.v1.Session() as sess:
	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)
	for i in range(40):
		sess.run(train_step)
		learning_rate_val =sess.run(learning_rate)
		global_step_val = sess.run(global_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print("After %s steps: global_step is %f, learnng_rate is %f, w is %f, loss is %f." %(i, global_step_val, learning_rate_val, w_val, loss_val))
