from common import * 
from common_visualization import * 
from train_autoencoder import c_autoencoder
from train_autoencoder_classifier import c_classifier_with_specialization_layer
from train_classifier import c_classifier


def visualize_classifier(model, path, data, samples):
 	if not os.path.exists(path_s):
	  os.makedirs(path_s)
	for i in xrange(0, len(samples)):
	  x = data['validation'][0][samples[i]].reshape(1, 28*28)
	  visualize_activation_W1_W2(model, x, path=path_s, posfix='_act'+str(i))
	visualize_b1(model.b1, path=path)
	visualize_b2(model.b2, path=path)
	visualize_W2(model.W2, path=path)
	visualize_W1(model.W1, path=path)
	visualize_W2_linear(model.W1, model.W2, 10, path=path)
	X_val = data['validation'][0]
	Y_val = data['validation'][1]
	X_train = data['train'][0]
	Y_train = data['train'][1]
	X_test = data['test'][0]
	Y_test = data['test'][1]

	plot_confusion(model, X_val, Y_val, path=path, posfix='_val')
	plot_confusion(model, X_train, Y_train, path=path, posfix='_train')
	plot_confusion(model, X_test, Y_test, path=path, posfix='_test')
	
def visualize_autoencoder(model, path, data, samples):
 	if not os.path.exists(path_s):
	  os.makedirs(path_s)
	col_count=len(samples)
	theano_eval_autoencoder = theano.function(inputs=[autoencoder.x], outputs=autoencoder.y)
	for i in xrange(0, len(samples)):
		x = data['validation'][0][samples[i]].reshape(1, 28*28)
		x = data['train'][0][i].reshape(1, 28*28)
		y = theano_eval_autoencoder(x)

		plot.subplot(2, col_count, 2 * i + 1)
		plot.imshow(x.reshape(28, 28))

		plot.subplot(2, col_count, 2 * i + 2)
		plot.imshow(y.reshape(28, 28))
	plot.savefig(path+'response.png')
	visualize_b1(model.b1, path=path)
	visualize_b2(model.b2, path=path)
	visualize_b2(model.b3, path=path)
	visualize_W2(model.W2, path=path)
	visualize_W1(model.W1, path=path)
	visualize_W3(model.W3, path=path)
	visualize_W2_linear(model.W1, model.W2, 10, path=path)
	

if __name__ == "__main__":
	data = get_mnist_data()
	l = data['validation'][1].shape[0]
	ind = np.random.permutation(l)
	ind = ind[:10]

	autoencoder = c_experiment.get_model("train_classifier")
	path_s = 'results/visualization/pretrained_classifier/'
	visualize_classifier(autoencoder, path_s, data, ind)
	
	autoencoder = c_experiment.get_model("train_autoencoder_classifier")
	path_s = 'results/visualization/classifier/'
	visualize_classifier(autoencoder, path_s, data, ind)
	
	autoencoder = c_experiment.get_model("train_autoencoder")
	path_s = 'results/visualization/autoencoder/'
	visualize_autoencoder(autoencoder, path_s, data, ind)