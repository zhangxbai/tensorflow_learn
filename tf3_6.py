#导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
#一次喂入神经网络的数据
BATCH_SIZE = 8
seed = 23455

#基于seed生成随机数
rng = np.random.RandomState(seed)
#随机数返回32行两列矩阵 表示32组 体积和重量 作为输入数据集
X = rng.rand(32,2)
#从32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果和不小于1 给Y赋值0
# 作为输入数据集的标签(正确答案）
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print("X:\n",X)
print("Y:\n",Y)

#定义神经网络的输入、参数和输出，定义前向传播的过程
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random.normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#2 定义损失函数以及反向传播方法

loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#3生成对话，训练STEPS轮
with tf.compat.v1.Session() as sess:
	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)
	# 输出目前（未经训练）的参数取值
	print("w1:\n", sess.run(w1))
	print("w2:\n", sess.run(w2))
	print("\n")


#训练模型

	STEPS = 13000
	for i in range(STEPS):
		start = (i*BATCH_SIZE) % 32
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
		if i % 500 == 0:
			total_loss = sess.run(loss, feed_dict={x: X, y_:Y})
			print("after %d training steps, loss on all data is %g" %(i, total_loss))
	print("\n")
	print("w1:\n", sess.run(w1))
	print("w2:\n", sess.run(w2))
