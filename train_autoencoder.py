from common import *

class c_autoencoder:
	def __init__(self, hidden_unit_count):
		self.hidden_unit_count = hidden_unit_count

		self.W1 = get_weight(28 * 28, self.hidden_unit_count)
		self.b1 = get_bias(self.hidden_unit_count) 

		self.W2 = get_weight(self.hidden_unit_count, 10)
		self.b2 = get_bias(10) 
		
		self.W3 = get_weight(10, self.hidden_unit_count)
		self.b3 = get_bias(self.hidden_unit_count) 

		self.W4 = self.W1.T
		self.b4 = get_bias(28 * 28)

		self.x = theano.tensor.fmatrix()

		self.h1 = theano.dot(self.x, self.W1) + self.b1
		self.h2 = theano.tensor.nnet.softmax(theano.dot(self.h1, self.W2) + self.b2)
		self.h3 = theano.dot(self.h2, self.W3) + self.b3
		self.y = theano.dot(self.h3, self.W4) + self.b4

		self.loss = theano.tensor.mean((self.x - self.y)**2)
		self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.b4]

def setup_autoencoder(training_info, param):
	# model parameters
	log10_learning_rate = numpy.random.uniform(-1, -4)
	log2_hidden_unit_count = numpy.random.uniform(4, 8) 	

	learning_rate = pow(10, log10_learning_rate)
	decay_rate = numpy.random.uniform(0.1, 0.9)
	hidden_unit_count = int(pow(2, log2_hidden_unit_count))

	training_info.hyper_parameter['hidden_unit_count'] = hidden_unit_count
	training_info.hyper_parameter['learning_rate'] = learning_rate
	training_info.hyper_parameter['decay_rate'] = decay_rate

	# build model
	model = c_autoencoder(hidden_unit_count)
	updates = rms_prop(model.loss, model.params, learning_rate, decay_rate) 

	theano_train = theano.function(inputs=[model.x], outputs=[], updates=updates, allow_input_downcast=True)
	theano_eval_loss = theano.function(inputs=[model.x], outputs=model.loss, allow_input_downcast=True)
	return model, theano_train, theano_eval_loss, None

if __name__ == "__main__":
	data = get_mnist_data()

	# train without labels
	data['train'][1] = [] 
	data['test'][1] = [] 
	data['validation'][1] = [] 

	experiment = c_experiment("train_autoencoder", "Train network A using RMSProp and early stopping", setup_autoencoder)
	experiment.run(data, 100)


