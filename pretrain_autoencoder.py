from common import *
from train_autoencoder import c_autoencoder

def setup_pretraining(training_info, param):
	log10_learning_rate = numpy.random.uniform(-1, -4)
	log2_hidden_unit_count = numpy.random.uniform(4, 8) 	

	learning_rate = pow(10, log10_learning_rate)
	decay_rate = numpy.random.uniform(0.1, 0.9)
	hidden_unit_count = int(pow(2, log2_hidden_unit_count))

	training_info.hyper_parameter['hidden_unit_count'] = hidden_unit_count
	training_info.hyper_parameter['learning_rate'] = learning_rate
	training_info.hyper_parameter['decay_rate'] = decay_rate

	model = c_autoencoder(hidden_unit_count) 
	h1 = theano.dot(model.x, model.W1) + model.b1
	h2 = theano.dot(h1, model.W4) + model.b4
	loss = theano.tensor.mean((h2 - model.x) ** 2)
	params = [model.W1, model.b1, model.b4]
	updates = rms_prop(loss, params, learning_rate, decay_rate) 

	theano_train = theano.function(inputs=[model.x], outputs=[], updates=updates, allow_input_downcast=True)
	theano_eval_loss = theano.function(inputs=[model.x], outputs=model.loss, allow_input_downcast=True)
	return model, theano_train, theano_eval_loss, None

def setup_autoencoder_with_pretraining(training_info, param): 
	model = c_experiment.get_model(param)
	info = c_experiment.get_training_info(param)

	learning_rate = info.hyper_parameter['learning_rate']
	hidden_unit_count = info.hyper_parameter['hidden_unit_count']
	decay_rate = info.hyper_parameter['decay_rate']

	training_info.hyper_parameter['hidden_unit_count'] = hidden_unit_count
	training_info.hyper_parameter['learning_rate'] = learning_rate
	training_info.hyper_parameter['decay_rate'] = decay_rate

	updates = rms_prop(model.loss, [model.W2, model.b2, model.W3, model.b3], learning_rate, decay_rate) 

	theano_train = theano.function(inputs=[model.x], outputs=[], updates=updates, allow_input_downcast=True)
	theano_eval_loss = theano.function(inputs=[model.x], outputs=model.loss, allow_input_downcast=True)
	return model, theano_train, theano_eval_loss, None	

if __name__ == "__main__":
	data = get_mnist_data()

	# train without labels
	data['train'][1] = [] 
	data['test'][1] = [] 
	data['validation'][1] = [] 

	pretrain = c_experiment("pretrain_autoencoder", "Use greedy layerwise pre-training", setup_pretraining)
	pretrain.run(data, 10)

	train = c_experiment("train_autoencoder_with_pretraining", "Use pretrained network to train autoencoder", setup_autoencoder_with_pretraining)
	train.run(data, 10, "pretrain_autoencoder")



