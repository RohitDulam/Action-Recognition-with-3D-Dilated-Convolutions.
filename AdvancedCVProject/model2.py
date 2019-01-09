''' The model contains the code for the second model talked about in the Experiments section of the paper.
The variables file_dir, txt_dir and testing_txt_dir values should be changed to the directory containing the UCF50 Action Recognition dataset, 
the training data text file and the testing data text file respectively. The model is trained for 3 epochs on 3600 randomly selected videos out 
of the dataset. For the testing process, only 10 random samples have been picked at a time from a total of 1740 videos to give out the accuracy 
for that batch. The highest accuracy the model has achieved is 30%. The explanation for this dissapointing results has been discussed in the conclusion 
section of the paper.'''


import tensorflow as tf
import numpy as np
import random
import cv2
import time

### The blow 3 lines are to be changed. txt_dir and testing_txt_dir are the paths to the Textfiles containing the Training and Testing splits respectively. 
### The file_dir should contain the path to the categories in the dataset. The below given example should give a good idea how the path should be. 

file_dir = "C:/Users/dulam/Desktop/Advanced_CV/UCF50/UCF50/"
txt_dir = "C:/Users/dulam/Desktop/Rewriting.txt"
testing_txt_dir = "C:/Users/dulam/Desktop/RewritingTest.txt"
X_recon = tf.placeholder(tf.float32, [None, 16, 120, 160, 1])
X = tf.placeholder(tf.float32, [None, 60, 120, 160, 1])
labels = tf.placeholder(tf.float32, [None , 50])
alpha = tf.placeholder(tf.float32)
batch_size = 3 # this can be changed according to the computation power at hand. 
recon_data = []

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

def extract_videos_to_frames(data_dir): # Extracting videos with more than 60 frames. This check isn't required as the text files in the folder contain videos with more than 60 frames. 
	list_of_frames = []
	count = 0
	file_ = file_dir + data_dir
	cap = cv2.VideoCapture(file_)
	frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	while(cap.isOpened()):
		_, frame = cap.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
		list_of_frames.append(small)
		count += 1
		if count > 59:
			cap.release()
	return np.array(list_of_frames)

def extract_videos_to_frames_recon(data_dir, counter): # Extracting videos for reconstruction. Done separately in order for easier readability and also to extract just 16 frames for reconstruction loss. 
	list_of_frames = []
	count = 0
	file_ = file_dir + data_dir
	cap = cv2.VideoCapture(file_)
	frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	while(cap.isOpened()):
		_, frame = cap.read()
		if counter%3 == 0:
			if len(list_of_frames) < 16:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
				list_of_frames.append(small)
		counter += 1
		count += 1
		if count > 59:
			cap.release()
	return np.array(list_of_frames)

def extract_frames_and_labels(data,testing = False):
	frames_total = []
	labels_total = []
	frames_recon = []
	if testing:
		for i in data:
			print(i)
			frames_total.append(extract_videos_to_frames(str(i[0])))
			labels_total.append(get_label(int(i[1])))
		return frames_total, labels_total
	else:
		for i in data:
			frames_total.append(extract_videos_to_frames(str(i[0])))
			frames_recon.append(extract_videos_to_frames_recon(str(i[0]), counter = 0))
			labels_total.append(get_label(int(i[1])))
		return frames_total, frames_recon, labels_total

def Classification(X_):
	with tf.variable_scope("Conv3D"):
		strides = [1, 1, 1, 1, 1]
		initializer = tf.contrib.layers.xavier_initializer()
		filter1_1 = tf.get_variable("filter1_1", [7, 7, 7, 1, 32], dtype = tf.float32, initializer = initializer)
		filter1_2 = tf.get_variable("filter1_2", [3, 3, 3, 32, 64], dtype = tf.float32, initializer = initializer)
		filter1_3 = tf.get_variable("filter1_3", [3, 3, 3, 64, 128], dtype = tf.float32, initializer = initializer)
		filter1_4 = tf.get_variable("filter1_4", [3, 3, 3, 128, 256], dtype = tf.float32, initializer = initializer)
		filter1_5 = tf.get_variable("filter1_5", [3, 3, 3, 256, 512], dtype = tf.float32, initializer = initializer)
		filter1_6 = tf.get_variable("filter1_6", [3, 3, 3, 512, 768], dtype = tf.float32, initializer = initializer)
		filter1_7 = tf.get_variable("filter1_7", [3, 3, 3, 768, 1024], dtype = tf.float32, initializer = initializer)


		conv_1 = tf.nn.relu(tf.nn.conv3d(X_, filter1_1, strides = strides, padding = 'SAME'))
		print("conv_1", conv_1.get_shape().as_list())
		conv_2 = tf.nn.relu(tf.nn.conv3d(conv_1, filter1_2, strides = strides, padding = 'SAME'))
		print("conv_2", conv_2.get_shape().as_list())
		max_pool_1 = tf.nn.relu(tf.nn.max_pool3d(conv_2, ksize = [1, 2, 2, 2, 1], strides = [1, 1, 2, 2, 1], padding = 'SAME'))
		conv_3 = tf.nn.relu(tf.nn.conv3d(max_pool_1, filter1_3, strides = [1, 2, 1, 1, 1], padding = 'SAME'))
		print("conv_3", conv_3.get_shape().as_list())
		max_pool_2 = tf.nn.relu(tf.nn.max_pool3d(conv_3, ksize = [1,2,2,2,1], strides = [1, 1, 2, 2, 1], padding = 'SAME'))
		conv_4 = tf.nn.relu(tf.nn.conv3d(max_pool_2, filter1_4, strides = [1, 2, 1, 1, 1], padding = 'SAME'))
		print("conv_4", conv_4.get_shape().as_list())
		conv_5 = tf.nn.relu(tf.nn.conv3d(conv_4, filter1_5, strides = [1, 2, 2, 2, 1], padding = 'SAME'))
		print("conv_5", conv_5.get_shape().as_list())
		conv_6 = tf.nn.relu(tf.nn.conv3d(conv_5, filter1_6, strides = [1, 2, 2, 2, 1], padding = 'SAME'))
		print("conv_6", conv_6.get_shape().as_list())
		conv_7 = tf.nn.relu(tf.nn.conv3d(conv_6, filter1_7, strides = [1, 2, 2, 2, 1], padding = 'SAME'))
		print("conv_7", conv_7.get_shape().as_list())
		recon_data.append(conv_5)
		d = conv_7.get_shape().as_list()
		conv_reshape = tf.reshape(conv_7, [-1, d[1] * d[2] * d[3] * d[4]])
		weight = tf.Variable(initializer([d[1] * d[2] * d[3] * d[4], 1000]))
		bias = tf.Variable(initializer([1000]))
		output_1 = tf.add(tf.matmul(conv_reshape, weight),bias)
		weight_2 = tf.Variable(initializer([1000, 50]))
		bias_2 = tf.Variable(initializer([50]))
		output_2 = tf.nn.sigmoid(tf.add(tf.matmul(output_1, weight_2),bias_2))
		return output_2

def Reconstruction(recon):
	initializer = tf.contrib.layers.xavier_initializer()
	with tf.variable_scope("Conv3D" , reuse = True):
		filter_r1 = tf.get_variable("filter1_5")
		filter_r2 = tf.get_variable("filter1_4")
		filter_r3 = tf.get_variable("filter1_3")
		filter_r4 = tf.get_variable("filter1_2")
		filter_r6 = tf.Variable(initializer([3, 3, 3, 32, 1]))

		deconv_r_1 = tf.nn.leaky_relu(tf.nn.conv3d_transpose(recon[0] , filter_r1, output_shape = [batch_size , 8, 30, 40, 256], strides = [1, 1, 2, 2, 1], padding = 'SAME'))
		print("deconv_r_1", deconv_r_1.get_shape().as_list())
		deconv_r_2 = tf.nn.leaky_relu(tf.nn.conv3d_transpose(deconv_r_1 , filter_r2, output_shape = [batch_size , 8, 60, 80, 128], strides = [1, 1, 2, 2, 1], padding = 'SAME'))
		print("deconv_r_2", deconv_r_2.get_shape().as_list())
		deconv_r_3 = tf.nn.leaky_relu(tf.nn.conv3d_transpose(deconv_r_2 , filter_r3, output_shape = [batch_size , 16, 60, 80, 64], strides = [1, 2, 1, 1, 1], padding = 'SAME'))
		print("deconv_r_3", deconv_r_3.get_shape().as_list())
		deconv_r_4 = tf.nn.leaky_relu(tf.nn.conv3d_transpose(deconv_r_3 , filter_r4, output_shape = [batch_size , 16, 120, 160, 32], strides = [1, 1, 2, 2, 1], padding = 'SAME'))
		print("deconv_r_4", deconv_r_4.get_shape().as_list())
		deconv_final = tf.nn.leaky_relu(tf.nn.conv3d(deconv_r_4, filter_r6, strides = [1,1,1,1,1], padding = 'SAME'))
		return deconv_final

output = Classification(X)
recon_frame = Reconstruction(recon_data)
Y_ = tf.nn.softmax(output)
correct = tf.equal(tf.argmax(labels,1) , tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct , tf.float32))
conv_cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = output)
recon_loss = tf.reduce_mean(tf.squared_difference(recon_frame, X_recon))
actual_loss = conv_cost + alpha * recon_loss
optimize = tf.train.AdamOptimizer().minimize(actual_loss)

init=tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)
	get_data = get_video_data(txt_dir)
	for i in range(3):
		get_data = random.sample(get_data, len(get_data)) # Videos are shuffled first, before selecting from them.
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
			frames_ ,frames_recon_, labels_ = extract_frames_and_labels(data)
			x_reshape = np.reshape(frames_, (batch_size, 60, 120, 160, 1))
			x_reshape_recon = np.reshape(frames_recon_, (batch_size, 16, 120, 160, 1))
			labl = np.reshape(labels_ , (batch_size, 50))
			data_dict = {X : x_reshape, X_recon : x_reshape_recon, labels : labl, alpha : 2.0}
			start = time.time()
			session.run(optimize, feed_dict = data_dict)
			end = time.time()
			print("time cost:%.2f"%(end-start))
		print("End of 1 epoch")
	
	get_test_data = get_video_data(testing_txt_dir)
	get_test_data = random.sample(get_test_data, len(get_test_data)) # Videos are shuffled first before selecting from them.
	get_test_data = get_test_data[:1740]
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
		x_reshape = np.reshape(frames_, (batch_size, 60, 120, 160, 1))
		labl = np.reshape(labels_ , (batch_size, 50))
		data_dict = {X : x_reshape, labels : labl}
		print("Accuracy for each batch : ",session.run(accuracy, feed_dict = data_dict))
	end = time.time()
	print("Total testing time cost:%.2f"%(end-start))
	
	


