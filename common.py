import cPickle
import numpy
import theano.tensor
import theano
import matplotlib.pyplot as plot
import seaborn
import MNIST
import os
import shutil
import sys

theano.config.floatX = 'float32'
html_image_size = 500
html_small_image_size = 400
# theano.config.exception_verbosity = 'high'
# theano.config.optimizer = 'fast_compile'

def get_mnist_data(validation_size = 10000):
	data = { 'train': [[], []], 'validation': [[], []], 'test': [[], []] }

	training_set = MNIST.MNIST("data").load_training()
	test_set = MNIST.MNIST("data").load_testing()

	train_size = len(training_set[0]) - validation_size
	test_size = len(test_set[0])

	data['train'][0] = numpy.asarray(training_set[0][0:train_size], dtype='float32')
	data['train'][1] = numpy.asarray(training_set[1][0:train_size])
	data['validation'][0] = numpy.asarray(training_set[0][train_size:], dtype='float32')
	data['validation'][1] = numpy.asarray(training_set[1][train_size:])
	data['test'][0] = numpy.asarray(test_set[0], dtype='float32')
	data['test'][1] = numpy.asarray(test_set[1])

	return data

def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(numpy.float32(1), 'float32')
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape), 'float32')
        v_previous = theano.shared(numpy.zeros(theta_previous.get_value().shape), 'float32')

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (theano.tensor.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

def rms_prop(cost, params, learning_rate=0.01, decay_rate=0.9):
	updates = []
	grads = theano.tensor.grad(cost=cost, wrt=params)
	for p, g in zip(params, grads):
		r = theano.shared(p.get_value()*0., 'float32')
		r_new = decay_rate * r + (1 - decay_rate) * g**2 
		updates.append([r, r_new])
		updates.append([p, p - learning_rate * g / theano.tensor.sqrt(r_new + 1e-8)])
	return updates

def gradient_descent(cost, params, learning_rate = 0.01):
	updates = []
	grads = theano.tensor.grad(cost=cost, wrt=params)
	for p, g in zip(params, grads):
		updates.append([p, p - g * learning_rate])
	return updates

def get_weight(size_in, size_out): 
	scale = numpy.sqrt(6.0 / (size_in + size_out))
	w = theano.shared(numpy.asarray(((numpy.random.rand(size_in, size_out) * 2. - 1.) * scale)), 'float32')
	return w

def get_bias(size):
	return theano.shared(numpy.zeros(size), 'float32')

def training_info_to_html(info, dir, image_size): 
	# Generate html
	html = "<table><tr><td>"
  	html += "<table>"

  	for param in info.hyper_parameter.keys():
		html += "<tr><td>" + param + " </td><td>" + str(info.hyper_parameter[param]) + "</td> <td></td> </tr>"
	html += "<tr><td></td><td></td><td></tr></tr>"
	html += "<tr><td>Training loss/perf:</td><td>" + str(info.train_loss[info.stop_time]) + "</td> <td>" + str(info.train_perf[info.stop_time]) + "</td> </tr>"
	html += "<tr><td>Validation loss/perf:</td><td>" + str(info.validation_loss[info.stop_time]) + "</td> <td>" + str(info.validation_perf[info.stop_time]) + "</td> </tr>"
	html += "<tr><td>Test loss/perf:</td><td>" + str(info.test_loss[info.stop_time]) + "</td> <td>" + str(info.test_perf[info.stop_time]) + "</td> </tr>"
	html += "<tr><td>Stop time:</td><td>" + str(info.stop_time) + "</td> <td></td> </tr>"
	html += "</table>"

	html += "</td>"

	html += "<td><img width=" + str(image_size) + " src=learning_loss_" + str(info.index) + ".png></img></td>"
	html += "<td><img width=" + str(image_size) + " src=learning_perf_" + str(info.index) + ".png></img></td>"
	html += "</tr></table>\n"

	# Generate images
	x_axis = numpy.arange(len(info.train_loss))
	plot.clf()
	plot.plot(x_axis, info.train_loss, label="Train loss")
	plot.plot(x_axis, info.validation_loss, label="Validation loss")
	plot.plot(x_axis, info.test_loss, label="Test loss")
	plot.xlabel("Epoch")
	plot.ylabel("Loss")
	plot.legend()
	plot.savefig(dir + "/learning_loss_" + str(info.index) + ".png")

	x_axis = numpy.arange(len(info.train_perf))
	plot.clf()
	plot.plot(x_axis, info.train_perf, label="Train performance")
	plot.plot(x_axis, info.validation_perf, label="Validation performance")
	plot.plot(x_axis, info.test_perf, label="Test performance")
	plot.xlabel("Epoch")
	plot.ylabel("Performance")
	plot.legend()
	plot.savefig(dir + "/learning_perf_" + str(info.index) + ".png")

	return html

def create_training_report(dir, name, description, best_model, best_model_info, training_info_array): 
	# save model 
	model_file = open(dir + "/model.pickle", "w")
	model_file.write(cPickle.dumps(best_model))
	model_file.close() 

	model_info_file = open(dir + "/model_info.pickle", "w")
	model_info_file.write(cPickle.dumps(best_model_info))
	model_info_file.close() 
	html = "<html><body><h1>" + name + "</h1>\n"

	html += "<p>" + description + "</p>"

	# best model section
	html += "<h2>Best model</h2>"
	html += training_info_to_html(best_model_info, dir, html_image_size)

	# hyperparameter search section
	html += "<h2>Hyperparameter search</h2>"
	html += "<table>\n"

	hyper_parameter_names = best_model_info.hyper_parameter.keys()
	hyper_parameter_count = len(best_model_info.hyper_parameter)
	for i in range(hyper_parameter_count): 
		if i % 2 == 1: 
			continue 

		if i + 1 < hyper_parameter_count:
			html += "<tr>"
			html += "<td><img width=" + str(html_image_size) + " src=" + hyper_parameter_names[i] + "_vs_error.png></img></td>"
			html += "<td><img width=" + str(html_image_size) + " src=" + hyper_parameter_names[i + 1] + "_vs_error.png></img></td>"
			html += "</tr>\n"
		else:
			html += "<tr>"
			html += "<td><img width=" + str(html_image_size) + " src=" + hyper_parameter_names[-1] + "_vs_error.png></img></td>"
			html += "<td>"
			html += "</tr>\n"
	
	html += "</table>\n"

	# sorted list of other runs
	html += "<h2>Best 100 runs</h2>"
	sort_by_validation_error = sorted(training_info_array, key=lambda info: info.validation_loss[info.stop_time])
	sort_by_validation_error = sort_by_validation_error[0:100]
	for info in sort_by_validation_error: 
		html += training_info_to_html(info, dir, html_small_image_size)

	html += "</body></hmtl>"

	html_file = open(dir + "/index.html", "w") 
	html_file.write(html)
	html_file.close()

	for param in best_model_info.hyper_parameter.keys():
		param_vs_error = [] 

		for info in training_info_array:
			param_vs_error.append((info.hyper_parameter[param], info.train_loss[info.stop_time], info.validation_loss[info.stop_time], info.test_loss[info.stop_time]))

		param_vs_error = numpy.asarray(sorted(param_vs_error))

		plot.clf()
		plot.plot(param_vs_error[:, 0], param_vs_error[:, 1], label="Train loss")
		plot.plot(param_vs_error[:, 0], param_vs_error[:, 2], label="Validation loss")
		plot.xlabel(param)
		plot.ylabel("Loss")
		plot.legend()
		plot.savefig(dir + "/" + param + "_vs_error.png")

	print "Generated training report to " + dir

class c_training_info:
	def __init__(self, index): 
		# training run index
		self.index = index

		# hyperparameters
		self.hyper_parameter = {} 

		# training params
		self.max_epoch_count = 100
		self.minibatch_size = 100
		self.early_stop_patience = 10
		self.early_stop_begin = 20

		# training stats
		self.train_loss = [] 
		self.validation_loss = []
		self.test_loss = []
		self.train_perf = [] 
		self.validation_perf = []
		self.test_perf = []

		self.stop_time = 0 

class c_experiment: 
	def __init__(self, name, description, setup): 
		self.name = name
		self.description = description
		self.setup = setup
		self.hyper = [] 
		self.data = {"train": [], "validation": [], "test": [] }
		self.best_model = None
		self.best_training_info = None
		self.training_info_array = [] 

	@staticmethod
	def get_result_dir(name):
		return "results/" + name

	@staticmethod
	def get_model(name):
		file = c_experiment.get_result_dir(name) + "/model.pickle" 
		return cPickle.load(open(file, "r"))

	@staticmethod	
	def get_training_info(name):
		file = c_experiment.get_result_dir(name) + "/model_info.pickle" 
		return cPickle.load(open(file, "r"))

	@staticmethod
	def clean(result_dir):
		# clean old results
		if not os.path.isdir("results"):
			os.mkdir("results")
		if os.path.isdir(result_dir):
			shutil.rmtree(result_dir)
		os.mkdir(result_dir)

	def train(self, data, training_info, theano_train, theano_eval_loss, theano_eval_performance): 
		min_loss = None
		use_x_only = True 

		if len(data['train'][1]) > 0:
			use_x_only = False

		for epoch in range(training_info.max_epoch_count): 
			train_loss = 0
			validation_loss = 0
			test_loss = 0

			train_perf = 0
			validation_perf = 0
			test_perf = 0

			if use_x_only:
				for i in range(0, len(data['train'][0]), training_info.minibatch_size):
					theano_train(data['train'][0][i : i + training_info.minibatch_size])
					sys.stdout.write("%d/%d\r" % (i / training_info.minibatch_size + 1, data['train'][0].shape[0] / training_info.minibatch_size))
					sys.stdout.flush()	

				train_loss = theano_eval_loss(data['train'][0])
				validation_loss = theano_eval_loss(data['validation'][0])
				test_loss = theano_eval_loss(data['test'][0])

				if theano_eval_performance != None:
					train_perf = theano_eval_perf(data['train'][0])
					validation_perf = theano_eval_perf(data['validation'][0])
					test_perf = theano_eval_perf(data['test'][0])
			else:
				for i in range(0, len(data['train'][0]), training_info.minibatch_size):
					theano_train(data['train'][0][i : i + training_info.minibatch_size], data['train'][1][i : i + training_info.minibatch_size])
					sys.stdout.write("%d/%d\r" % (i / training_info.minibatch_size + 1, data['train'][0].shape[0] / training_info.minibatch_size))
					sys.stdout.flush()	

				train_loss = theano_eval_loss(data['train'][0], data['train'][1])
				validation_loss = theano_eval_loss(data['validation'][0], data['validation'][1])
				test_loss = theano_eval_loss(data['test'][0], data['test'][1])

				if theano_eval_performance != None:
					train_perf = theano_eval_performance(data['train'][0], data['train'][1])
					validation_perf = theano_eval_performance(data['validation'][0], data['validation'][1])
					test_perf = theano_eval_performance(data['test'][0], data['test'][1])

			print "[" + str(epoch) + "]" + " train loss " + str(train_loss) + " validation loss " + str(validation_loss) + " test loss " + str(test_loss)
			str_space = "    "	
			if epoch >= 10: 
				str_space += " "
			if epoch >= 100:
				str_space += " "
			print str_space + "train perf " + str(train_perf) + " validation perf " + str(validation_perf) + " test perf " + str(test_perf)

			training_info.train_loss.append(train_loss)
			training_info.validation_loss.append(validation_loss)
			training_info.test_loss.append(test_loss)

			training_info.train_perf.append(train_perf)
			training_info.validation_perf.append(validation_perf)
			training_info.test_perf.append(test_perf)

			# early stopping
			if epoch == training_info.early_stop_begin:
				min_loss = validation_loss 
				training_info.stop_time = epoch

			if epoch > training_info.early_stop_begin:
				if validation_loss < min_loss:
					min_loss = validation_loss
					training_info.stop_time = epoch 
				else:
					time_since_improvement = epoch - training_info.stop_time
					if epoch > training_info.early_stop_begin and time_since_improvement > training_info.early_stop_patience: 
						print "Early stop"
						return training_info

		return training_info 	

	def run(self, data, sample_count, user_param = None): 
		result_dir = c_experiment.get_result_dir(self.name)
		c_experiment.clean(result_dir)

		self.best_training_info = None
		self.best_model = None
		self.training_info_array = []

		for i in range(sample_count): 
			training_info = c_training_info(i)
			model, theano_train, theano_eval_loss, theano_eval_performance = self.setup(training_info, user_param)

			print "\nTraining run " + str(i) + ":"
			for param in training_info.hyper_parameter:
				print param + " : " + str(training_info.hyper_parameter[param])

			# train
			info = self.train(data, training_info, theano_train, theano_eval_loss, theano_eval_performance)
			self.training_info_array.append(info)

			if self.best_model == None or info.validation_loss[info.stop_time] < self.best_training_info.validation_loss[self.best_training_info.stop_time]: 
				self.best_model = model
				self.best_training_info = training_info

			create_training_report(result_dir, self.name, self.description, self.best_model, self.best_training_info, self.training_info_array)
