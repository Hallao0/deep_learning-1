from common import *
from train_autoencoder import c_autoencoder

class c_classifier:
	def __init__(self, autoencoder): 
		self.W1 = autoencoder.W1
		self.b1 = autoencoder.b1

		self.W2 = autoencoder.W2
		self.b2 = autoencoder.b2

		self.x = theano.tensor.fmatrix()
		self.y = theano.tensor.ivector()

		self.h1 = theano.dot(self.x, self.W1) + self.b1
		self.z = theano.tensor.nnet.softmax(theano.dot(self.h1, self.W2) + self.b2) 

		# fine-tune params
		self.params = [self.W1, self.b1, self.W2, self.b2]

		self.loss = -theano.tensor.mean(theano.tensor.log(self.z)[theano.tensor.arange(self.y.shape[0]), self.y])	
		self.performance = 1. - theano.tensor.mean(theano.tensor.neq(theano.tensor.argmax(self.z, axis=1), self.y))	

def setup(training_info, user_param): 
	# model parameters
	log10_learning_rate = numpy.random.uniform(-1, -5)
	learning_rate = pow(10, log10_learning_rate)

	training_info.hyper_parameter['learning_rate'] = learning_rate

	# build model
	autoencoder = c_experiment.get_model(user_param[0])
	model = c_classifier(autoencoder)
	updates = adam(model.loss, model.params, learning_rate)

	theano_train = theano.function(inputs=[model.x, model.y], outputs=[], updates=updates, allow_input_downcast=True)
	theano_eval_loss = theano.function(inputs=[model.x, model.y], outputs=model.loss, allow_input_downcast=True)
	theano_eval_performance = theano.function([model.x, model.y], outputs=model.performance, allow_input_downcast=True)
	return model, theano_train, theano_eval_loss, theano_eval_performance

if __name__ == "__main__": 
	data = get_mnist_data()
	experiment = c_experiment("train_autoencoder_classifier", "Train classifier using a pretrained autoencoder", setup)
	experiment.run(data, 10, ("train_autoencoder_search", data['train']))
	# experiment.run(get_mnist_data(), 10, "train_autoencoder_with_pretraining")

