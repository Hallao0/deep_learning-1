from common import *
from train_autoencoder import c_autoencoder

class c_classifier: 
	def __init__(self, autoencoder): 
		self.W1 = autoencoder.W1
		self.b1 = autoencoder.b1

		self.W2 = get_weight(autoencoder.hidden_unit_count, 10)
		self.b2 = get_bias(10)

		self.x = theano.tensor.fmatrix()
		self.y = theano.tensor.ivector()

		self.h1 = theano.tensor.dot(self.x, self.W1) + self.b1
		self.z = theano.tensor.nnet.softmax(theano.tensor.dot(self.h1, self.W2) + self.b2) 

		self.params = [self.W2, self.b2]

		self.loss = -theano.tensor.mean(theano.tensor.log(self.z)[theano.tensor.arange(self.y.shape[0]), self.y])	
		self.performance = 1. - theano.tensor.mean(theano.tensor.neq(theano.tensor.argmax(self.z, axis=1), self.y))

def setup(training_info, user_param): 
	# model parameters
	log10_learning_rate = numpy.random.uniform(-1, -4)
	log2_hidden_unit_count = numpy.random.uniform(4, 9) 	

	learning_rate = pow(10, log10_learning_rate)
	hidden_unit_count = int(pow(2, log2_hidden_unit_count))
	decay_rate = 0.9

	training_info.hyper_parameter['hidden_unit_count'] = hidden_unit_count
	training_info.hyper_parameter['learning_rate'] = learning_rate
	training_info.hyper_parameter['decay_rate'] = decay_rate

	# build model
	autoencoder = c_experiment.get_model(user_param)
	model = c_classifier(autoencoder)
	updates = rms_prop(model.loss, model.params, learning_rate, decay_rate)

	theano_train = theano.function(inputs=[model.x, model.y], outputs=[], updates=updates, allow_input_downcast=True)
	theano_eval_loss = theano.function(inputs=[model.x, model.y], outputs=model.loss, allow_input_downcast=True)
	theano_eval_performance = theano.function([model.x, model.y], outputs=model.performance, allow_input_downcast=True)
	return model, theano_train, theano_eval_loss, theano_eval_performance

if __name__ == "__main__": 
	experiment = c_experiment("train_autoencoder_classifier", "Train classifier using a pretrained autoencoder", setup)
	experiment.run(get_mnist_data(), 10, "train_autoencoder")
	# experiment.run(get_mnist_data(), 10, "train_autoencoder_with_pretraining")

