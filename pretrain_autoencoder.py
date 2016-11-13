from common import *
from train_autoencoder import c_autoencoder

def setup_pretraining(training_info, user_param):
	# 100  hidden units and learning rate range were obtained via random search
	log10_learning_rate = numpy.random.uniform(-4, -5)
	learning_rate = pow(10, log10_learning_rate)
	decay_rate = numpy.random.uniform(0.5, 0.99)
	hidden_unit_count = 100

	training_info.hyper_parameter['hidden_unit_count'] = hidden_unit_count
	training_info.hyper_parameter['learning_rate'] = learning_rate
	training_info.hyper_parameter['decay_rate'] = decay_rate

	model = c_autoencoder(hidden_unit_count) 
	h1 = theano.dot(model.x, model.W1) + model.b1
	h2 = theano.dot(h1, model.W4) + model.b4
	y = model.get_output(h2)
	loss = theano.tensor.mean((y - model.x) ** 2)

	unit_variance_init(user_param, model.W1, model.x, h1)

	params = [model.W1, model.b1, model.b4]
	updates = rms_prop(loss, params, learning_rate, decay_rate) 

	theano_train = theano.function(inputs=[model.x], outputs=[], updates=updates, allow_input_downcast=True)
	theano_eval_loss = theano.function(inputs=[model.x], outputs=loss, allow_input_downcast=True)
	return model, theano_train, theano_eval_loss, None

def setup_autoencoder_with_pretraining(training_info, user_param): 
	model = c_experiment.get_model(user_param[0])
	info = c_experiment.get_training_info(user_param[0])

	log10_learning_rate = numpy.random.uniform(-1, -5)
	learning_rate = pow(10, log10_learning_rate)
	decay_rate = numpy.random.uniform(0.5, 0.99)	
	hidden_unit_count = info.hyper_parameter['hidden_unit_count']

	training_info.hyper_parameter['hidden_unit_count'] = hidden_unit_count
	training_info.hyper_parameter['learning_rate'] = learning_rate
	training_info.hyper_parameter['decay_rate'] = decay_rate

	unit_variance_init(user_param[1], model.W2, model.x, model.h2)
	unit_variance_init(user_param[1], model.W3, model.x, model.h3)

	updates = rms_prop(model.loss, [model.W2, model.b2, model.W3, model.b3], learning_rate, decay_rate) 

	theano_train = theano.function(inputs=[model.x], outputs=[], updates=updates, allow_input_downcast=True)
	theano_eval_loss = theano.function(inputs=[model.x], outputs=model.loss, allow_input_downcast=True)
	return model, theano_train, theano_eval_loss, None	

def continue_pretraining(training_info, user_param):
	model = c_experiment.get_model(user_param)
	info = c_experiment.get_training_info(user_param)	

	hidden_unit_count = info.hyper_parameter['hidden_unit_count']
	learning_rate = info.hyper_parameter['learning_rate']
	decay_rate = info.hyper_parameter['decay_rate']

	training_info.hyper_parameter['hidden_unit_count'] = hidden_unit_count
	training_info.hyper_parameter['learning_rate'] = learning_rate * numpy.random.uniform(0.1, 10)
	training_info.hyper_parameter['decay_rate'] = decay_rate

	h1 = theano.dot(model.x, model.W1) + model.b1
	h2 = theano.dot(h1, model.W4) + model.b4
	y = model.get_output(h2)
	loss = theano.tensor.mean((y - model.x) ** 2)

	params = [model.W1, model.b1, model.b4]
	updates = rms_prop(loss, params, learning_rate, decay_rate) 

	theano_train = theano.function(inputs=[model.x], outputs=[], updates=updates, allow_input_downcast=True)
	theano_eval_loss = theano.function(inputs=[model.x], outputs=loss, allow_input_downcast=True)
	return model, theano_train, theano_eval_loss, None

def main():
	if len(sys.argv) != 2:
		return 

	data = get_mnist_data()
	data['train'][1] = [] 
	data['test'][1] = [] 
	data['validation'][1] = [] 

	if sys.argv[1] == "pretrain":
		print "pretrain"
		pretrain = c_experiment("pretrain_autoencoder", "Use greedy layerwise pre-training", setup_pretraining)
		pretrain.run(data, 20, data['train'])
	elif sys.argv[1] == "train":
		print "train"
		train = c_experiment("train_autoencoder_with_pretraining", "Use pretrained network to train autoencoder", setup_autoencoder_with_pretraining)
		train.run(data, 50, ("pretrain_autoencoder", data['train']))
	elif sys.argv[1] == "continue_pretraining":
		pretrain = c_experiment("pretrain_autoencoder_continue", "Continue greedy layerwise pre-training for the best seen network", continue_pretraining)
		pretrain.run(data, 10, "pretrain_autoencoder")


if __name__ == "__main__":
	main()




