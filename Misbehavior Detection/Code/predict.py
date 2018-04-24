import os,sys
import numpy as np 
import pandas as pd 
import tensorflow as tf 

dir_path = ""

#Preparing Training data
train_file_path = dir_path + "train.csv"
train_file = pd.read_csv(train_file_path,skiprows=1,header=None)

test_file_path = dir_path + "test2.csv"
test_file = pd.read_csv(test_file_path,skiprows=1,header=None)

train_file = train_file.drop(train_file.columns[0],axis=1)
train_file = train_file.values

test_file = test_file.drop(test_file.columns[0],axis=1)
test_file = test_file.values

train_X_temp = train_file[5:50000,:-1]
train_Y = train_file[6:50001,-1]

#Combining previous 5 time step data into one row
train_X = np.zeros((train_X_temp.shape[0],8*5))
for i in range(train_X_temp.shape[0]):
	for j in range(5):
		for k in range(8):
			train_X[i][j*8+k] = train_X_temp[i-j][k]

print("Data Preprocessing Done!")
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

class Neural_net():
	def __init__(self,sess):
		self.learning_rate = 0.0008
		self.hidden_layers = 3
		self.layer_size = [250,500,100]
		self.input_size = 40
		self.output_size = 1
		self.epochs=800
		self.stddev = 0.1
		self.tryCount = 1
		self.batch_size = 1000
		self.sess = sess
		self.checkpoint_dir = "checkpoint"
		self.model_dir = "./SavedMode"
		self.name = "Misbehaviour_detection"
		self.input = tf.placeholder(tf.float32,[None,self.input_size],name="input_tensor")
		self.label = tf.placeholder(tf.float32,[None,self.output_size],name="label")
		
		return self.build_model()


	def MLP_layer(self,input_is,layer_index,name="MLP_layer"):
		previous_layer_size = self.input_size
		current_layer_size = self.output_size

		if layer_index != self.hidden_layers+1 and layer_index >0:
			current_layer_size = self.layer_size[layer_index-1]

		if layer_index != 1 and layer_index > 0:
			previous_layer_size = self.layer_size[layer_index-2]

		with tf.variable_scope(name) as scope:
			weights = tf.get_variable('weights',[previous_layer_size,current_layer_size],
				initializer=tf.random_normal_initializer(stddev=self.stddev))
			biases = tf.get_variable('biases',[current_layer_size],
				initializer=tf.constant_initializer(0.0))

			return tf.add(tf.matmul(input_is,weights),biases)

	#MLP Model training
	def build_model(self):
		with tf.variable_scope(self.name) as name:
			if self.hidden_layers == 3:
				layer_1 = tf.nn.relu(self.MLP_layer(self.input,1,name="MLP_layer_1"))
				layer_2 = tf.nn.relu(self.MLP_layer(layer_1,2,name="MLP_layer_2"))
				layer_3 = tf.nn.relu(self.MLP_layer(layer_2,3,name="MLP_layer_3"))
				self.output_prediction = self.MLP_layer(layer_3,4,name="MLP_layer_4")
				self.loss = tf.losses.mean_squared_error(self.label,self.output_prediction)
				self.loss_summary = tf.summary.scalar("loss",self.loss)
				# self.loss_histogram = tf.summary.histogram("histogram",self.loss)
				self.saver = tf.train.Saver()

	#save trained model so as to not repeat training
	def save(self,checkpoint_dir,step):
		model_name = "{}/MLP.model".format(self.tryCount)
		checkpoint_dir = os.path.join(self.model_dir,checkpoint_dir)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,os.path.join(checkpoint_dir,model_name),global_step=step)

	#Load already trained model
	def load(self):
		checkpoint_dir = self.checkpoint_dir+"/{}".format(self.tryCount)
		checkpoint_dir = os.path.join(self.model_dir,checkpoint_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess,os.path.join(checkpoint_dir,ckpt_name))
			# counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			return True
		else:
			return False

		self.saver.restore(self.sess,os.path.join(checkpoint_dir,model_name))


	def next_batch(self,batch_size,input_is,label):
		idx = np.arange(0,len(input_is))
		np.random.shuffle(idx)
		idx = idx[:batch_size]
		data_shuffle = [input_is[i] for i in idx]
		labels_shuffle = [label[i] for i in idx]

		return np.asarray(data_shuffle), np.asarray(labels_shuffle)


	def train(self):
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		self.optimize = self.optimizer.minimize(self.loss)

		init = tf.global_variables_initializer()
		self.sess.run(init)
		self.merger_summary = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter("./NN_logs/{}".format(self.tryCount),self.sess.graph)

		print("here")
		count = 1
		for epoch in range(self.epochs):
			for iteration in range(30000/self.batch_size):
				batch_input, batch_label = self.next_batch(self.batch_size,train_X,train_Y)
				batch_label = np.reshape(batch_label,(self.batch_size,1))
				_,summary = self.sess.run([self.optimize,self.merger_summary],feed_dict={self.input:batch_input,self.label:batch_label})
				# print("it is:", iteration)
				self.writer.add_summary(summary,count)
				count = count + 1
			print("Current Epoch is: ",epoch+1)

		self.save(self.checkpoint_dir,1)
		
	def test(self):
		while(1):
			test_file_name = raw_input("command: ")
			if test_file_name == "exit":
				break
			else:
				print(test_file_name)
				test_file_name = dir_path + test_file_name
				test_file = pd.read_csv(test_file_name,header=None)
				test_file = test_file.values
				test_X = np.array(test_file[-1,:-1])
				test_X = np.reshape(test_X,(-1,40))
				actual_halt = np.array([test_file[-1,-1]],ndmin=2)
				print(actual_halt)
				pred_halt = self.sess.run(self.output_prediction,feed_dict={self.input:test_X,self.label:actual_halt})
				print(pred_halt)
				if abs(pred_halt-actual_halt) >=50:
					print("Detector Misbehaviour")
				else:
					print("Accurate Prediction: ",pred_halt)
print("start")
# neural_net.fit(train_X_new,train_Y)
# print("save the model")

# with open(dir_path+"model.pkl",'wb') as p_file:
# 	cPickle.dump(neural_net,p_file)


def main(_):
	config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
	with tf.Session(config=config) as sess:
		net = Neural_net(sess)
		print("model build!")
		net.train()
		# print("loading model!")
		# net.load()
		print("model loaded!")
		net.test()

if __name__ == '__main__':
	tf.app.run()
# print("load the model")

# with open(dir_path+"model.pkl",'rb') as p_file:
# 	loaded_neural_net = cPickle.load(p_file)

# # model.partial_fit(train_X,train_Y)
# # test_X = np.array(test_file[-1,:])

# # print(test_X.shape)

# # print(model.predict(test_X))

# # print(test_X)
# # print(test_file[5,-1])
# # print(model.get_params())

# while(1):
# 	test_file_name = raw_input("command: ")
# 	if test_file_name == "exit":
# 		break
# 	else:
# 		print(test_file_name)
# 		test_file_name = dir_path + test_file_name
# 		test_file = pd.read_csv(test_file_name,header=None)
# 		test_file = test_file.values
# 		test_X = np.array(test_file[-1,:-1])
# 		test_X = np.reshape(test_X,(-1,40))
# 		pred_halt = loaded_neural_net.predict(test_X)
# 		actual_halt = test_file[-1,-1]
# 		if abs(pred_halt-actual_halt) >=50:
# 			print("Detector Misbehaviour")
# 		else:
# 			print("Accurate Prediction: ",pred_halt)		
