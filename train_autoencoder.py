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
		self.y = self.get_output(theano.dot(self.h3, self.W4) + self.b4)

		self.loss = theano.tensor.mean((self.x - self.y)**2)
		self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.b4]

	def get_output(self, y):
		return 255.0 * theano.tensor.nnet.sigmoid(y)

def setup_autoencoder(training_info, user_param):
	log10_learning_rate = numpy.random.uniform(-1, -5)
	learning_rate = pow(10, log10_learning_rate)
	decay_rate = numpy.random.uniform(0.9, 0.99)
	# hidden_unit_count = int(numpy.random.uniform(10, 512))
	hidden_unit_count = 100

	training_info.hyper_parameter['hidden_unit_count'] = hidden_unit_count
	training_info.hyper_parameter['learning_rate'] = learning_rate
	training_info.hyper_parameter['decay_rate'] = decay_rate

	# build model
	model = c_autoencoder(hidden_unit_count)
	unit_variance_init(user_param, model.W1, model.x, model.h1)
	unit_variance_init(user_param, model.W2, model.x, model.h2)
	unit_variance_init(user_param, model.W3, model.x, model.h3)
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

	experiment = c_experiment("train_autoencoder", "Train autoencoder with sigmoidal output layer", setup_autoencoder)
	experiment.run(data, 100, data['train'])


