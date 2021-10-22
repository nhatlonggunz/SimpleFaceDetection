import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_CIFAR10 
import input_main as inp
import cv2 

data, lab_name, lab_ind = inp.create_list_data('./datasets')
# data = inp.random_list_data(data)

num_img = len(data)
num_lab = max(lab_ind.values()) + 1

# print(num_img)
# print(num_lab)

train_ind = (0, num_img//5 * 3)
valid_ind = (train_ind[-1], train_ind[-1] + num_img//5)
test_ind = (valid_ind[-1], num_img)

train_data = data[train_ind[0] : train_ind[-1]]
valid_data = data[valid_ind[0] : valid_ind[-1]]
test_data = data[test_ind[0] : test_ind[-1]]

num_train = train_ind[-1] - train_ind[0] + 1
num_valid = valid_ind[-1] - valid_ind[0] + 1
num_test = test_ind[-1] - test_ind[0] + 1

# print(train_data)
# print(valid_data)
# print(test_data)

# train_data = (0, 800)
# valid_data = (800, 900)
# test_data = (900, 1000)

# test_img, test_lab = inp.read_batch(data, lab_ind, test_data[0], test_data[-1])
# valid_img, valid_lab = inp.read_batch(data, lab_ind, valid_data[0], valid_data[-1])
# print(test_lab.shape)

# general
learning_rate = 0.0001
img_sz_1 = 160
img_sz_2  = 120
img_chan = 1
batch_sz = 64

# convolution 1
num_fil_1 = 32
fil_sz_1 = 5
pool_sz_1 = 2

# convolution 2
num_fil_2 = 32
fil_sz_2 = 5
pool_sz_2 = 2

# convolution 3
num_fil_3 = 64
fil_sz_3 = 5
pool_sz_3 = 2

# convolution 4
num_fil_4 = 64
fil_sz_4 = 5
pool_sz_4 = 2

# convolution 5
num_fil_5 = 128
fil_sz_5 = 5
pool_sz_5 = 2

# convolution 6
num_fil_6 = 128
fil_sz_6 = 5
pool_sz_6 = 2

# fully connected 1
neuron_fc1 = 4096
keep_prob = 0.6

# output layer
neuron_out = num_lab

# --------------------------

# --------------------------

# testing = cv2.imread('test.jpg')
# testing = inp.normalize(testing)
# testing = inp.convert_to_grayscale(testing)
# cv2.imshow('gg', testing)
# cv2.waitKey(0)

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
	# x = tf.constant(testing, dtype=tf.float32)
	# y_correct = tf.constant(mnist.train.labels[0], dtype=tf.float32)

	x = tf.placeholder(shape=[None, img_sz_1, img_sz_2], dtype=tf.float32)
	y_correct = tf.placeholder(shape=[None, num_lab], dtype=tf.float32)

	x_image = tf.reshape(x, [-1, img_sz_1, img_sz_2, img_chan])


with tf.name_scope('Convolution_and_max_pooling_1') as scope:
	w_conv1 = weight_variable([fil_sz_1, fil_sz_1, img_chan, num_fil_1])
	b_conv1 = bias_variable([num_fil_1])
	h_conv1 = tf.nn.elu(conv2d(x_image, w_conv1) + b_conv1)
	h_pool1 = max_pool(h_conv1, pool_sz_1)

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# _ = sess.run(h_pool1)
# cv2.imshow('gg', _[0, :, :, 20])
# cv2.waitKey(0)

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
	y_predict = tf.nn.softmax(y)

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

def calc_accuracy(sess, data):
	size = len(data)
	num_avg = 10
	gg = []
	print('cc1')

	for i in range(num_avg):
		data = inp.random_list_data(data)
		print('cc2')

		l = 0
		r = size
		
		cnt = 0
		tmp = 0.0

		while l < r:
			print('cc3')
			cnt += 1
			sz = min(batch_sz, 2**np.log2(r - l))

			img, lab = inp.read_batch(data, lab_ind, l, l + sz)
			accur = sess.run(accuracy, feed_dict={ x : img ,
												   y_correct : lab })
			tmp += accur
			l += sz

		tmp = tmp / cnt 
		gg.append(tmp)

	res = 0.0
	for i in gg:
		res += i
	print('cc4')
	return res / num_avg


def train():
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	for training in range(20):
		train_dat = inp.random_list_data(train_data)
		l_train, r_train = 0, len(train_dat)

		while l_train < r_train:
			sz_train = min(batch_sz, 2**np.log2(r_train - l_train))
			train_img, train_lab = inp.read_batch(train_dat, lab_ind, l_train, l_train + sz_train)

			sess.run(train_step, feed_dict={ x : train_img ,
											 y_correct : train_lab })
			# print('dmmm')
			l_train += sz_train

		valid_accur = calc_accuracy(sess, valid_data)
		print('step {}: {}'.format(training + 1, valid_accur))

	accur = calc_accuracy(sess, test_data)
	print('Accuracy: {}'.format(accur))

	save_path = saver.save(sess, 'main/main.ckpt')
	sess.close()

def predict_img(img):
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, 'main/main.ckpt')

	if img.shape[0] < img_sz_1 or img.shape[1] < img_sz_2:
		print('picture is too small')
		pass

	# print(img.ndim)

	if img.shape[0] != img_sz_1 or img.shape[1] != img_sz_2:
		img = cv2.resize(img, (img_sz_2, img_sz_1))	
	if np.max(img) > 1:
		img = inp.normalize(img)
	if img.ndim > 2:
		img = inp.convert_to_grayscale(img)

	# cv2.imshow('dm', img)
	# cv2.waitKey(0)
	# print('dcmm')
	# print(img)

	predict = sess.run(y_predict, feed_dict={x : [img]})

	ret = []
	for i in range(num_lab):
		if predict[0,i] > 0:
			ret.append([lab_name[i], predict[0,i] * 100])

	sess.close()
	return ret

if __name__ == "__main__":
	choose = {'1' : 'Train data',
			  '2' : 'Detect Face'}

	print("Choose what to do:")
	for i in choose:
		print("{}. {}.".format(i, choose[i]))

	user_choice = input()
	if(user_choice == '2'):
		import cv2

		face_cascade = cv2.CascadeClassifier('cascade_data/haarcascades/haarcascade_frontalface_default.xml')
		eye_cascade = cv2.CascadeClassifier('cascade_data/haarcascades/haarcascade_eye.xml')

		#this is the cascade we just made. Call what you want
		# watch_cascade = cv2.CascadeClassifier('watchcascade10stage.xml')

		cap = cv2.VideoCapture(0)

		while 1:
		    ret, img = cap.read()
		    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		     

		    for (_x,_y,_w,_h) in faces:
		        cv2.rectangle(img,(_x,_y),(_x+_w,_y+_h),(255,0,0),2)

		        tmp = img[_y:_y+_h, _x:_x+_w]
		        print(tmp.shape)
		        cv2.imshow('img', tmp)
		        cv2.waitKey(0)
		        cv2.destroyAllWindows()

		        pred = predict_img(tmp)
		        print(pred)
		        
		        roi_gray = gray[_y:_y+_h, _x:_x+_w]
		        roi_color = img[_y:_y+_h, _x:_x+_w]

		    cv2.imshow('img',img)
		    k = cv2.waitKey(30) & 0xff
		    if k == 27:
		        break

		cap.release()
		cv2.destroyAllWindows()


# train()

# img = test_img[-1]
# lab = test_lab[-1]

# img = cv2.imread('test.jpg')

# cv2.imshow('img', img)
# cv2.waitKey(0)

# pred = predict_img(img)
# for i in range(len(pred)):
# 	print("{}: {}%".format(pred[i][0], pred[i][-1]))




