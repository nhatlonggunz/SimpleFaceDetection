import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_CIFAR10 
import input_Yale

# train_images, train_labels = input_CIFAR10.read_data_sets('./CIFAR10_data')
# # train_images = input_CIFAR10.convert_to_grayscale(train_images)
# train_labels = input_CIFAR10.toLogit(train_labels)
# train_images, train_labels = input_CIFAR10.standardize(train_images, 
# 													   train_labels)

# test_images, test_labels = input_CIFAR10.read_test_set('./CIFAR10_data')
# # test_images = input_CIFAR10.convert_to_grayscale(test_images)
# test_labels = input_CIFAR10.toLogit(test_labels)
# test_images, test_labels = input_CIFAR10.standardize(test_images,
# 													 test_labels)


train_images, train_labels, train_names = input_Yale.load_training_data('./CroppedYale')

print(train_images[0])
print(train_images.shape)
print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

plt.imshow(train_images[1].reshape([168, 168, 3]))
plt.show()


# general
learning_rate = 0.0005
img_sz = 168
img_chan = 1
batch_sz = 64

# convolution 1
num_fil_1 = 32
fil_sz_1 = 5
pool_sz_1 = 2

# convolution 2
num_fil_2 = 64
fil_sz_2 = 5
pool_sz_2 = 2

# convolution 3
num_fil_3 = 128
fil_sz_3 = 5
pool_sz_3 = 2

# convolution 4
num_fil_4 = 128
fil_sz_4 = 5
pool_sz_4 = 2

# convolution 5
num_fil_5 = 256
fil_sz_5 = 5
pool_sz_5 = 2

# convolution 6
num_fil_6 = 512
fil_sz_6 = 5
pool_sz_6 = 2

# fully connected 1
neuron_fc1 = 1024
keep_prob = 0.6

# output layer
neuron_out = 10




def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.truncated_normal(shape, dtype=tf.float32)
	return tf.Variable(initial)

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, sz):
	return tf.nn.max_pool(x, ksize=[1, sz, sz, 1],
							 strides=[1, sz, sz, 1],
							 padding='SAME')

with tf.name_scope('input') as scope:
	# x = tf.constant(mnist.train.images[0], dtype=tf.float32)
	# y_correct = tf.constant(mnist.train.labels[0], dtype=tf.float32)

	x = tf.placeholder(shape=[None, img_sz**2, img_chan], dtype=tf.float32)
	y_correct = tf.placeholder(shape=[None, 10], dtype=tf.float32)

	x_image = tf.reshape(x, [-1, img_sz, img_sz, img_chan])


with tf.name_scope('Convolution_and_max_pooling_1') as scope:
	w_conv1 = weight_variable([fil_sz_1, fil_sz_1, img_chan, num_fil_1])
	b_conv1 = bias_variable([num_fil_1])
	h_conv1 = tf.nn.elu(conv2d(x_image, w_conv1) + b_conv1)
	h_pool1 = max_pool(h_conv1, pool_sz_1)

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# _ = sess.run(h_pool1)
# print(_.shape)

with tf.name_scope('Convolution_and_max_pooling_2') as scope:
	w_conv2 = weight_variable([fil_sz_2, fil_sz_2, num_fil_1, num_fil_2])
	b_conv2 = bias_variable([num_fil_2])
	h_conv2 = tf.nn.elu(conv2d(h_pool1, w_conv2) + b_conv2)
	h_pool2 = max_pool(h_conv2, pool_sz_2)

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# _ = sess.run(h_pool2)
# print(_.shape)

with tf.name_scope('Convolution_and_max_pooling_3') as scope:
	w_conv3 = weight_variable([fil_sz_3, fil_sz_3, num_fil_2, num_fil_3])
	b_conv3 = bias_variable([num_fil_3])
	h_conv3 = tf.nn.elu(conv2d(h_pool2, w_conv3) + b_conv3)
	h_pool3 = max_pool(h_conv3, pool_sz_3)

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# _ = sess.run(h_pool3)
# print(_.shape)

with tf.name_scope('Convolution_and_max_pooling_4') as scope:
	w_conv4 = weight_variable([fil_sz_4, fil_sz_4, num_fil_3, num_fil_4])
	b_conv4 = bias_variable([num_fil_4])
	h_conv4 = tf.nn.elu(conv2d(h_pool3, w_conv4) + b_conv4)
	h_pool4 = max_pool(h_conv4, pool_sz_4)

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# _ = sess.run(h_pool4)
# print(_.shape)

with tf.name_scope('Convolution_and_max_pooling_5') as scope:
	w_conv5 = weight_variable([fil_sz_5, fil_sz_5, num_fil_4, num_fil_5])
	b_conv5 = bias_variable([num_fil_5])
	h_conv5 = tf.nn.elu(conv2d(h_pool4, w_conv5) + b_conv5)
	h_pool5 = max_pool(h_conv5, pool_sz_5)

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# _ = sess.run(h_pool5)
# print(_.shape)

with tf.name_scope('Convolution_and_max_pooling_6') as scope:
	w_conv6 = weight_variable([fil_sz_6, fil_sz_6, num_fil_5, num_fil_6])
	b_conv6 = bias_variable([num_fil_6])
	h_conv6 = tf.nn.elu(conv2d(h_pool5, w_conv6) + b_conv6)
	h_pool6 = max_pool(h_conv6, pool_sz_6)
	h_pool_flat = tf.reshape(h_pool6, shape=[-1, tf.size(h_pool6[0])])


# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# _ = sess.run(h_pool6)
# print(_.shape)


with tf.name_scope('Fully_connected_layer_1') as scope:
	w_fc1 = weight_variable([tf.size(h_pool6[0]), neuron_fc1])
	b_fc1 = bias_variable([neuron_fc1])
	h_fc1 = tf.nn.elu(tf.matmul(h_pool_flat, w_fc1) + b_fc1)

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('Output_layer') as scope:
	w_out = weight_variable(shape=[neuron_fc1, neuron_out])
	b_out = bias_variable(shape=[neuron_out])
	y = tf.matmul(h_fc1_drop, w_out) + b_out

with tf.name_scope('cost') as scope:
	cost = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(
			 logits=y,
 			 labels=y_correct))

with tf.name_scope('train') as scope:
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.name_scope('accuracy') as scope:
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_correct, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

validation_size = 200

for i in range(20):

	l = 0

	while l < train_images.shape[0]:
		sz = min(batch_sz, 2**np.log2(train_images.shape[0] - l))
		#print("{} - {}".format(l, l + sz))

		sess.run(train_step, feed_dict={ 
								x : train_images[l : int(l + sz)],
								y_correct : train_labels[l : int(l + sz)] })
		l  = l + sz

	accur = sess.run(accuracy, feed_dict={
								x : test_images[: 200],
								y_correct : test_labels[: 200]})

	print("step {}: {}".format(i + 1, accur))