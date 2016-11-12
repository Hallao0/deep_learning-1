from common import * 
from train_autoencoder import c_autoencoder

if __name__ == "__main__": 
	data = get_mnist_data()
	# autoencoder = c_experiment.get_model("train_autoencoder_with_pretraining")
	autoencoder = c_experiment.get_model("train_autoencoder")

	theano_eval_autoencoder = theano.function(inputs=[autoencoder.x], outputs=autoencoder.y)

	col_count = 10
	for i in range(col_count): 

		x = data['train'][0][i].reshape(1, 28*28)
		y = theano_eval_autoencoder(x)

		plot.subplot(2, col_count, 2 * i + 1)
		plot.imshow(x.reshape(28, 28))

		plot.subplot(2, col_count, 2 * i + 2)
		plot.imshow(y.reshape(28, 28))

	plot.show()
