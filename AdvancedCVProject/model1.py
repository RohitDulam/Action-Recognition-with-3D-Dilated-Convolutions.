'''The model is trained for 3 epochs on 3600 randomly selected videos out 
of the dataset. For the testing process, only 10 random samples have been picked at a time from a total of 1740 videos to give out the accuracy 
for that batch. The highest accuracy the model has achieved is 30%. The explanation for this dissapointing results has been discussed in the conclusion 
section of the paper.'''


import tensorflow as tf
import numpy as np
import random
import time
import cv2

### The blow 3 lines are to be changed. txt_dir and testing_txt_dir are the paths to the Textfiles containing the Training and Testing splits respectively. 
### The file_dir should contain the path to the categories in the dataset. The below given example should give a good idea how the path should be. 
file_dir = "C:/Users/dulam/Desktop/Advanced_CV/UCF50/UCF50/"
txt_dir = "C:/Users/dulam/Desktop/Rewriting.txt"
testing_txt_dir = "C:/Users/dulam/Desktop/RewritingTest.txt"

X = tf.placeholder(tf.float32, [None, 60, 240, 320, 1])
labels = tf.placeholder(tf.float32, [None , 50])
batch_size = 1 # The can be changed according to the Computation power at hand. 

def get_label(numb):
	res = np.zeros([50])
	res[numb-1] = 1
	return res

def get_video_data(file_path):
	video_links = []
	with open(file_path, "r") as f:
		m = True
		counter = 0
		for line in f:
			x = line.split()
			if x != []:
				video_links.append(x)
	f.close()
	return video_links

def extract_videos_to_frames(data_dir):
	list_of_frames = []
	count = 0
	file_ = file_dir + data_dir
	cap = cv2.VideoCapture(file_)
	frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	while(cap.isOpened()):
		_, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		list_of_frames.append(frame)
		count += 1
		if count > 59:
			cap.release()
	return np.array(list_of_frames)


def extract_frames_and_labels(data):
	frames_total = []
	labels_total = []
	frames_recon = []
	for i in data:
		frames_total.append(extract_videos_to_frames(str(i[0])))
		labels_total.append(get_label(int(i[1])))
	return frames_total, labels_total

def conv3d_block(sequence_input, filter, strides, dilations, pooling = True):

	L_conv = tf.nn.conv3d(sequence_input, filter, strides = strides, dilations = dilations, padding = 'VALID')#add padding
	L_act = tf.nn.leaky_relu(L_conv, alpha = 0.15)
	return L_act

def architectures(X_,arch2 = True):

	strides_dilation = [1,1,1,1,1]
	strides = [1, 3, 2, 2, 1] 
	strides_1 = [1, 2, 2, 2, 1]
	strides_final = [1, 5, 1, 1, 1]
	dilations = [1, 2, 2, 2, 1]
	dilation_none = [1,1,1,1,1]

	initializer = tf.contrib.layers.xavier_initializer()

	# Architecture - 2
	with tf.variable_scope("Dilated3D"):
		if arch2:

			filter2_1 = tf.get_variable("filter2_1", [3, 3, 3, 1, 16], dtype = tf.float32, initializer = initializer)
			filter2_2 = tf.get_variable("filter2_2", [3, 3, 3, 16, 32], dtype = tf.float32, initializer = initializer)
			filter2_3 = tf.get_variable("filter2_3", [3, 3, 3, 32, 64], dtype = tf.float32, initializer = initializer)
			filter2_4 = tf.get_variable("filter2_4", [3, 3, 3, 64, 128], dtype = tf.float32, initializer = initializer)
			filter2_5 = tf.get_variable("filter2_5", [3, 3, 3, 128, 128], dtype = tf.float32, initializer = initializer)
			filter2_6 = tf.get_variable("filter2_6", [3, 3, 3, 128, 256], dtype = tf.float32, initializer = initializer)
			filter2_7 = tf.get_variable("filter2_7", [3, 1, 1, 256, 64], dtype = tf.float32, initializer = initializer)
			filter2_8 = tf.get_variable("filter2_8", [1, 3, 3, 64, 16], dtype = tf.float32, initializer = initializer)
			filter2_9 = tf.get_variable("filter2_9", [1, 3, 3, 16, 1], dtype = tf.float32, initializer = initializer)		
	
			conv2_1 = conv3d_block(X, filter2_1, strides = strides_dilation, dilations = dilations) # Output 16 channels with rest of them remaining same.
			print("conv2_1",conv2_1.get_shape().as_list())
			conv2_2 = conv3d_block(conv2_1, filter2_2, strides = [1, 2, 1, 1, 1], dilations = dilation_none) # Outputs 32 channels.
			print("conv2_2",conv2_2.get_shape().as_list())
			conv2_3 = conv3d_block(conv2_2, filter2_3, strides = strides_dilation, dilations = dilations)# Outputs 64 channels.
			print("conv2_3",conv2_3.get_shape().as_list())
			conv2_4 = conv3d_block(conv2_3, filter2_4, strides = [1, 2, 3, 4, 1], dilations = dilation_none)# Outputs 128 channels.
			print("conv2_4", conv2_4.get_shape().as_list())
			conv2_5 = conv3d_block(conv2_4, filter2_5, strides = strides_dilation, dilations = dilations)# No change in channel size.
			print("conv2_5",conv2_5.get_shape().as_list())
			conv2_6 = conv3d_block(conv2_5, filter2_6, strides = strides_1, dilations = dilation_none)#Outputs 256 channels.
			print("conv2_6",conv2_6.get_shape().as_list())
			conv2_7 = conv3d_block(conv2_6, filter2_7, strides = [1, 3, 1, 1, 1], dilations = dilation_none)# 1*1 convolution with 64 output channels
			print("conv2_7",conv2_7.get_shape().as_list())
			conv2_8 = conv3d_block(conv2_7, filter2_8, strides = [1, 1, 1, 1, 1], dilations = dilation_none)# outputs 16 channels.
			print("conv2_8",conv2_8.get_shape().as_list()) 
			conv2_9 = conv3d_block(conv2_8, filter2_9, strides = [1, 1, 1, 1, 1], dilations = dilation_none)# Outputs a single channel. Aggregating all the previous acquired information.
			print("conv2_7",conv2_7.get_shape().as_list())
			d = conv2_9.get_shape().as_list()
			print(d)
			conv_reshape = tf.reshape(conv2_9, [-1, d[2] * d[3]])
			weight = tf.Variable(initializer([d[2] * d[3], 50]))
			bias = tf.Variable(initializer([50]))
			output = tf.nn.sigmoid(tf.add(tf.matmul(conv_reshape, weight),bias))

	return output

output = architectures(X, arch2 = True)
Y_ = tf.nn.softmax(output)
correct = tf.equal(tf.argmax(labels,1) , tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))
conv_cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = output)
optimize = tf.train.AdamOptimizer().minimize(conv_cost)

initialize = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(initialize)
	get_data = get_video_data(txt_dir)
	for i in range(3):
		get_data = random.sample(get_data, len(get_data))
		get_data = get_data[:3600]
		for j in range(0, len(get_data), batch_size):
			data = []
			for m in range(batch_size):
				if m + j >= len(get_data):
					continue
				else:
					data.append(get_data[m+j])
			if len(data) < batch_size:
				batch_size = len(data)
			frames_ , labels_ = extract_frames_and_labels(data)
			x_reshape = np.reshape(frames_, (batch_size, 60, 240, 320, 1))
			labl = np.reshape(labels_ , (batch_size, 50))
			data_dict = {X : x_reshape, labels : labl}
			start = time.time()
			session.run(optimize, feed_dict = data_dict)
			end = time.time()
			print("time cost:%.2f"%(end-start))
		print("End of 1 epoch")
	get_test_data = get_video_data(testing_txt_dir)
	get_test_data = get_test_data[:1740]
	batch_size = 10 # this can be changed according to the computation power at hand. 
	get_test_data = random.sample(get_test_data, len(get_test_data))
	start = time.time()
	for j in range(0, len(get_data), batch_size):
		data = []
		for m in range(batch_size):
			if m + j >= len(get_data):
				continue
			else:
				data.append(get_data[m+j])
		if len(data) < batch_size:
			batch_size = len(data)
		frames_ , labels_ = extract_frames_and_labels(data, testing = True)
		x_reshape = np.reshape(frames_, (batch_size, 60, 240, 320, 1))
		labl = np.reshape(labels_ , (batch_size, 50))
		data_dict = {X : x_reshape, labels : labl}
		print("Accuracy for each batch : ",session.run(accuracy, feed_dict = data_dict))
	end = time.time()
	print("Total testing time cost:%.2f"%(end-start))
