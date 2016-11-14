import numpy as np
import theano.tensor
import theano
import PIL.Image as Image
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
import itertools

def visualize_W1(thW1, path=''):
	plot.clf()
	W1 = thW1.get_value(borrow=True)
	image = Image.fromarray(tile_raster_images(
	      X= W1.T,
	      img_shape=(28, 28), tile_shape=(10, 10),
	      tile_spacing=(1, 1)))
	image.save(path+'W1.png')
	
def visualize_W2(thW2, path=''):
	plot.clf()
	W2 = thW2.get_value(borrow=True)
	image = Image.fromarray(tile_raster_images(
	      X= W2.T,
	      img_shape=(10, 10), tile_shape=(2, 5),
	      tile_spacing=(1, 1)))
	image.save(path+'W2.png')

def visualize_W3(thW3, path=''):
	plot.clf()
	W3 = thW3.get_value(borrow=True)
	image = Image.fromarray(tile_raster_images(
	      X= W3.T,
	      img_shape=(2, 5), tile_shape=(10, 10),
	      tile_spacing=(1, 1)))
	image.save(path+'W3.png')
	
def visualize_b1(thb1, path=''):
	plot.clf()
	b1 = np.array([thb1.get_value(borrow=True)])
	plot.imshow(b1.T)
	frame1 = plot.gca()
	frame1.axes.get_xaxis().set_visible(False)
	plot.savefig(path+'b1.png')

def visualize_b2(thb2, path=''):
	plot.clf()
	b2 = np.array([thb2.get_value(borrow=True)])
	plot.imshow(b2.T)
	frame1 = plot.gca()
	frame1.axes.get_xaxis().set_visible(False)
	plot.savefig(path+'b2.png')

def visualize_b3(thb3, path=''):
	plot.clf()
	b3 = np.array([thb3.get_value(borrow=True)])
	plot.imshow(b3.T)
	frame1 = plot.gca()
	frame1.axes.get_xaxis().set_visible(False)
	plot.savefig(path+'b3.png')
	
def visualize_W2_linear(thW1, thW2, num_weights, path=''):
	'''
	Lee, H., Ekanadham, C., & Ng, A. (2008). Sparse deep belief net model for visual area
	V2. In J. C. Platt, D. Koller, Y. Singer and S. Roweis (Eds.),
	Advances in neural information processing systems 2008. Cambridge, MA: MIT Press
	'''
	W1 = thW1.get_value()
	W2 = thW2.get_value()
	W2_ind = np.argsort(W2, axis=0)
	
	W2_new = np.zeros((W1.shape[0],W2.shape[1]))
	for i in xrange(0, W1.shape[0]):
	  for j in xrange(0, num_weights):
	    for k in xrange(0, W2.shape[1]):
	      W2_new[i,k] += W1[i, W2_ind[j,k]]*W2[W2_ind[j,k],k]
	plot.clf()
	image = Image.fromarray(tile_raster_images(
	      X= W2_new.T,
	      img_shape=(28, 28), tile_shape=(2, 5),
	      tile_spacing=(1, 1)))
	image.save(path+'W2_linear.png')
	
def visualize_activation_W1_W2(model, x, path='', posfix = '_act'):
	'''
	Visualize activation of weights W1 and W2 as a response to input x
	'''
	h1_o_fun = theano.function([model.x],model.h1)
	z_fun = theano.function([model.x],model.z)
	h1_i = np.multiply(x.T, model.W1.get_value())
	z_i = np.multiply(h1_o_fun(x).T, model.W2.get_value())
	plot.clf()
	plot.imshow(x.reshape(28, 28))
	plot.savefig(path+'x'+posfix+'.png')
	plot.clf()
	image = Image.fromarray(tile_raster_images(
	      X= h1_i.T,
	      img_shape=(28, 28), tile_shape=(10,10),
	      tile_spacing=(1, 1), scale_rows_to_unit_interval=False))
	image.save(path+'h1'+posfix+'.png')
	plot.clf()
	image = Image.fromarray(tile_raster_images(
	      X= z_i.T,
	      img_shape=(10, 10), tile_shape=(2,5),
	      tile_spacing=(1, 1), scale_rows_to_unit_interval=False))
	image.save(path+'z'+posfix+'.png')
	plot.clf()
	z=z_fun(x)
	z_new = np.ones((100,10))
	for i in xrange(0, 10):
	    z_new[:,i] = z[:,i]*z_new[:,i]
	image = Image.fromarray(tile_raster_images(
	      X= z_new.T,
	      img_shape=(10,10), tile_shape=(2,5),
	      tile_spacing=(1, 1), scale_rows_to_unit_interval=False))
	image.save(path+'z'+posfix+'.png')
	
def plot_confusion(model, x, y, path='', posfix=''):
    #to_label = theano.tensor.argmax(model.y, axis=1)
    #z_fun = theano.function([model.x], model.z)
    #from_target_data_to_label = function([model.y], to_label)
    pred = theano.tensor.argmax(model.z, axis=1)
    get_prediction = theano.function([model.x], pred)
    
    conf = confusion_matrix(y, get_prediction(x))
    save_confusion_matrix(conf, path+'confusion'+posfix+'.png')

def save_confusion_matrix(cnf_matrix, name):
    plot.clf()
    np.set_printoptions(precision=2)
    title = 'Normalized confusion matrix on MNIST test set'
    class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plot.imshow(cnf_matrix, interpolation='nearest', cmap=plot.cm.Blues)
    plot.title(title)
    plot.colorbar()

    tick_marks = np.arange(len(class_name))
    plot.xticks(tick_marks, class_name)
    plot.yticks(tick_marks, class_name)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plot.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plot.tight_layout()
    plot.ylabel('True digit')
    plot.xlabel('Predicted digit')
    plot.savefig(name)
	

def scale_to_unit_interval(ndar, eps=1e-8):
	""" Scales all values in the ndarray ndar to be between 0 and 1 """
	ndar = ndar.copy()
	ndar -= ndar.min()
	ndar *= 1.0 / (ndar.max() + eps)
	return ndar
 
#Copyright previous team
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).
    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats
    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not
    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
	return out_array