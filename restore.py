import tensorflow as tf
import numpy as np
import time as t
import random
import os
import matplotlib.pyplot as plt

def recup():
  directory = 'fic/'
  recupe = []
  X, y = [], []
  for num,noms in enumerate(['s_bord1','s_bord2','s_bord3','s_bord4','s_centre','s_coin1',\
            's_coin2','s_coin3','s_coin4','vide','t_bord1','t_bord2','t_bord3',\
            't_centre','t_bord4','t_coin1','t_coin2','t_coin3','t_coin4']):
    file = open(directory + noms,'r')
    ligne = file.readline()
    classe = np.zeros(2)
    if num < 10:
      classe[0] = 1
    else:
      classe[1] = 1
    compteur = 0
    while ligne != "" and compteur <= 499:
      recupe.append([np.array(ligne.split(','),dtype='float32'),classe])
      #y.append(classe)
      #y.append(num)
      ligne = file.readline()
      compteur += 1
    file.close()
  random.shuffle(recupe)
  recupe = np.array(recupe)
  print(np.shape(recupe))
  return np.array(list(recupe[:,0]), dtype='float32'), np.array(list(recupe[:,1]), dtype='float32')
  #np.array(X, dtype='float32'), y#np.array(y, dtype='float64')

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights
	
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

X_recup, y_recup = recup()
lenX = int(len(X_recup)*0.001)
leni = len(X_recup)
X_train, X_test, y_train, y_test = X_recup[:lenX],X_recup[lenX:],\
								   y_recup[:lenX], y_recup[lenX:]
print(lenX)
X_recup, y_recup = 0, 0
input("recuperation done")

# Convolutional Layer 1.
filter_size1 = 2
num_filters1 = 4


n_classes = 2
batch_size = 64

x = tf.placeholder('float', [None, 3072])
x_image = tf.reshape(x, [-1, 32, 32, 3])
y = tf.placeholder('float')
keep_prob = tf.placeholder(tf.float32)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=3,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv1)
layer_flat_drop = tf.nn.dropout(layer_flat,keep_prob)

"""layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)"""

layer_fc2 = new_fc_layer(input=layer_flat_drop,
                         num_inputs=num_features,
                         num_outputs=n_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

print(layer_conv1)
print(layer_flat)
print(layer_fc2)

rate = tf.placeholder(tf.float32, shape=[])
l_rate = 0.001#5e-4
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(layer_fc2,y))
optimizer = tf.train.AdamOptimizer(rate).minimize(cost)

correct = tf.equal(tf.argmax(layer_fc2, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

saver = tf.train.Saver()
save_dir = 'final_model/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_model')

hm_epochs = 50
compteur = 0
prec = 10e100
with tf.Session() as sess:
	saver.restore(sess, save_path)
	
	for f in range(len(X_test)):
		if sess.run(correct,feed_dict={keep_prob: 1, x: X_test[f:f+1], y:y_test[f:f+1]}):
			compteur += 1
		else:
			img = X_test[f].reshape((32,32,3))
			plt.imshow(img)
			plt.show()
	print(f+1,compteur,compteur/(f+1))
			

"""t0 = t.time()
c, res = 0, 0
for g in range(0,len(X_train),2048):
	res += accuracy.eval({keep_prob: 1, x:X_train[g:g+2048], y:y_train[g:g+2048]})
	c += 1
print('Accuracy Train :',res/c,'sur',lenX,'images en', t.time() - t0,'sec')

t0 = t.time()
c, res = 0, 0
for g in range(0,len(X_test),2048):
	res += accuracy.eval({keep_prob: 1, x:X_test[g:g+2048], y:y_test[g:g+2048]})
	c += 1
print('Accuracy Test :',res/c,'sur',leni-lenX,'images en', t.time() - t0,'sec')
"""

