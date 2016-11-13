from common import *

class c_classifier: 
	def __init__(self, hidden_unit_count): 
		self.W1 = get_weight(28 * 28, hidden_unit_count)
		self.b1 = get_bias(hidden_unit_count)

		self.W2 = get_weight(hidden_unit_count, 10)
		self.b2 = get_bias(10)

		self.x = theano.tensor.fmatrix()
		self.y = theano.tensor.ivector()

		self.h1 = theano.tensor.dot(self.x, self.W1) + self.b1
		self.z = theano.tensor.nnet.softmax(theano.tensor.dot(self.h1, self.W2) + self.b2) 

		self.params = [self.W1, self.b1, self.W2, self.b2]

		self.loss = -theano.tensor.mean(theano.tensor.log(self.z)[theano.tensor.arange(self.y.shape[0]), self.y])	
		self.performance = 1. - theano.tensor.mean(theano.tensor.neq(theano.tensor.argmax(self.z, axis=1), self.y))

def setup(training_info, user_param): 
	# model parameters
	log10_learning_rate = numpy.random.uniform(-1, -5)

	learning_rate = pow(10, log10_learning_rate)
	hidden_unit_count = 100
	decay_rate = 0.9

	training_info.hyper_parameter['hidden_unit_count'] = hidden_unit_count
	training_info.hyper_parameter['learning_rate'] = learning_rate
	training_info.hyper_parameter['decay_rate'] = decay_rate

	# build model
	model = c_classifier(hidden_unit_count)
	updates = rms_prop(model.loss, model.params, learning_rate, decay_rate)

	theano_train = theano.function(inputs=[model.x, model.y], outputs=[], updates=updates, allow_input_downcast=True)
	theano_eval_loss = theano.function(inputs=[model.x, model.y], outputs=model.loss, allow_input_downcast=True)
	theano_eval_performance = theano.function([model.x, model.y], outputs=model.performance, allow_input_downcast=True)
	return model, theano_train, theano_eval_loss, theano_eval_performance

if __name__ == "__main__": 
	experiment = c_experiment("train_classifier", "Train basic classifier for network C", setup)
	experiment.run(get_mnist_data(), 10)

