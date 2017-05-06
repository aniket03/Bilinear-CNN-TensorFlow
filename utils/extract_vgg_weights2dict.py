import tensorflow as tf
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
import os
from tflearn.data_utils import shuffle
import numpy as np
import pickle 

def outer_product(inputs):
  n1 =  inputs[0]
  n2 = inputs[1]
  product = tf.batch_matmul(tflearn.reshape(n1,[-1,512,784]), n2)
  print product.get_shape()
  return product

def signed_square_root(layer):
  layer = tf.sign(layer)*tf.sqrt(tf.abs(layer))
  return layer

def vgg16_base(input):

    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1',trainable=False)
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    #x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8')
    x = tflearn.fully_connected(x, 100, activation='softmax', scope='fc8', restore=False)
    return x

def vgg16(input,label):

    x = tflearn.conv_2d(input, 64, 3, activation='relu', scope='conv1_1'+label, trainable=False)
    #temp_layer = tflearn.variables.get_layer_variables_by_name('conv1_1')
    #with model.session.as_default():
    #temp_layer = tflearn.variables.get_layer_variables_by_name('conv1_1')
    #tflearn.variables.set_value(x,tflearn.variables.get_value(temp_layer))

    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2'+label, trainable=False)
    #temp_layer = tflearn.variables.get_layer_variables_by_name('conv1_2')
    #with model.session.as_default():
    #  tflearn.variables.set_value(x,tflearn.variables.get_value(temp_layer))

    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1'+label)

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1'+label, trainable=False)
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2'+label, trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2'+label)

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1'+label, trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2'+label, trainable=False)
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3'+label, trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3'+label)

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1'+label, trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2'+label, trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3'+label, trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4'+label)

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1'+label, trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2'+label, trainable=False)
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3'+label, trainable=False)
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    #x = tflearn.fully_connected(x, num_class, activation='softmax', scope='fc8')
    x = tflearn.fully_connected(x, 100, activation='softmax', scope='fc8', restore=False)
    return x




data_dir = "/path/to/your/data"
model_path = "/path/to/your/vgg_model"
# the file gen by generated by gen_files_list.py
files_list = "/home/adoke/tf_tutorial/aircrafts_new/new_train_val/train_dummy.txt"
files_list_val = "/home/adoke/tf_tutorial/aircrafts_new/new_train_val/val_dummy.txt"

from tflearn.data_utils import image_preloader
'''
X, Y = image_preloader(files_list, image_shape=(224, 224), mode='file',
                       categorical_labels=True, normalize=False,
                       files_extension=['.jpg', '.png'], filter_channel=True)

X_val, Y_val = image_preloader(files_list_val, image_shape=(224, 224), mode='file',
                       categorical_labels=True, normalize=False,
                       files_extension=['.jpg', '.png'], filter_channel=True)
# or use the mode 'floder'
# X, Y = image_preloader(data_dir, image_shape=(224, 224), mode='folder',
#                        categorical_labels=True, normalize=True,
#                        files_extension=['.jpg', '.png'], filter_channel=True)

X, Y = shuffle(X, Y)'''
num_classes = 100 # num of your dataset

# VGG preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=[123.68, 116.779, 103.939],
                                     per_channel=True)

# VGG Network
x = tflearn.input_data(shape=[None, 224, 224, 3], name='input',
                       data_preprocessing=img_prep)
softmax = vgg16_base(x)

sgd = tflearn.SGD(learning_rate=0.001, lr_decay=0.96, decay_step=500)
regression = tflearn.regression(softmax, optimizer=sgd,
                                loss='categorical_crossentropy')

model = tflearn.DNN(regression, checkpoint_path='vgg_dummy',
                    best_checkpoint_path='vgg_dummy',max_checkpoints=3, tensorboard_verbose=2,
                    tensorboard_dir="./logs")

#model_file = os.path.join(model_path, "vgg16.tflearn")
#model.load("/home/adoke/tf_tutorial/aircrafts/vgg16.tflearn", weights_only=True)
model.load("/home/adoke/tf_tutorial/aircrafts_new/new_train_val/vgg16.tflearn", weights_only=True)
#first_layer = tflearn.variables.get_layer_variables_by_name('conv1_1')
#with model.session.as_default():
#  print tflearn.variables.get_value(first_layer[0])

vgg_weights_dict = {}
vgg_layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7']

for layer_name in vgg_layers:
  print layer_name
  base_layer = tflearn.variables.get_layer_variables_by_name(layer_name)
  '''if 'conv' in layer_name:
    vgg_weights_dict[layer_name] = [np.copy(model.get_weights(base_layer[0])),np.copy(model.get_weights(base_layer[1]))]
  else:
    print model.get_weights(base_layer)
    vgg_weights_dict[layer_name] = [np.copy(model.get_weights(base_layer[0]))]'''
  vgg_weights_dict[layer_name] = [model.get_weights(base_layer[0]),model.get_weights(base_layer[1])]
    
pickle.dump( vgg_weights_dict, open( "./vgg_weights.p", "wb" ) )  


'''

#VGG Network
x_1 = tflearn.input_data(shape=[None, 448, 448, 3], name='input_bcnn_1',
                       data_preprocessing=img_prep)
x_2 = tflearn.input_data(shape=[None, 448, 448, 3], name='input_bcnn_2', data_preprocessing=img_prep)

net1 = vgg16(x_1,'1')
#first_layer = tflearn.variables.get_layer_variables_by_name('conv1_11')
#with model.session.as_default():
#print tflearn.variables.get_value(first_layer[0])


#net1.load("/home/adoke/tf_tutorial/aircrafts_new/new_train_val/vgg16.tflearn", weights_only=True)
net2 = vgg16(x_2,'2')
#net2.load("/home/adoke/tf_tutorial/aircrafts_new/new_train_val/vgg16.tflearn", weights_only=True)
#print net1.get_shape()
#print net2.get_shape()

net1 = tflearn.reshape(net1,new_shape=[-1,784,512])
net2 = tflearn.reshape(net2,new_shape=[-1,784,512])

#print net1.get_shape()
#print tensorflow.transpose(net2).get_shape()
bcnn = tflearn.custom_layer([net1,net2],outer_product)
#bcnn = outer_product(tflearn.reshape(net1,[-1,512,784]),net2)
#print bcnn.get_shape()

phi_I = tflearn.flatten(bcnn)
#print phi_I.get_shape()

y = signed_square_root(phi_I)
#z = tflearn.layers.normalization.l2_normalize(y, dim=0)
z = tf.nn.l2_normalize(y, dim=0)

bcnn_softmax = tflearn.fully_connected(z, num_classes, activation='softmax', scope='fc8_bcnn')

sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=500)
regression = tflearn.regression(bcnn_softmax, optimizer=sgd,
                                loss='categorical_crossentropy')

bcnn_model = tflearn.DNN(regression, checkpoint_path='bcnn',
                    best_checkpoint_path='bcnn_best_val',max_checkpoints=3, tensorboard_verbose=2,
                    tensorboard_dir="./logs")

################### Setting up weights of bcnn from vgg ##################

vgg_layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']

for layer_name in vgg_layers:
  bcnn_layer_1 = tflearn.variables.get_layer_variables_by_name(layer_name+'1')
  bcnn_layer_2 = tflearn.variables.get_layer_variables_by_name(layer_name+'2')
  base_layer = tflearn.variables.get_layer_variables_by_name(layer_name)
  #print type(model.get_weights(base_layer[0]))
  bcnn_model.set_weights(bcnn_layer_1[0],np.copy(model.get_weights(base_layer[0])))
  bcnn_model.set_weights(bcnn_layer_1[1],np.copy(model.get_weights(base_layer[1])))
  bcnn_model.set_weights(bcnn_layer_2[0],np.copy(model.get_weights(base_layer[0])))
  bcnn_model.set_weights(bcnn_layer_2[1],np.copy(model.get_weights(base_layer[1])))
  print(bcnn_model.get_weights(bcnn_layer_1[0]).shape)

##########################################################################

bcnn_model.fit([X,X], [Y,Y], n_epoch=10, validation_set=([X_val,X_val],[Y_val,Y_val]), shuffle=True,show_metric=True, batch_size=64, snapshot_step=200, snapshot_epoch=False, validation_batch_size=200, run_id='bilinear_cnn')

bcnn_model.save('bcnn_01_500.tflearn')
X_test, Y_test = image_preloader('/home/adoke/tf_tutorial/aircrafts_new/from_start/a3_variants_test.txt', image_shape=(448, 448), mode='file',
                       categorical_labels=True, normalize=False,
                       files_extension=['.jpg', '.png'], filter_channel=True)

score = bcnn_model.evaluate(X_test, Y_test)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))

'''
'''
sgd = tflearn.SGD(learning_rate=0.001, lr_decay=0.96, decay_step=500)
regression = tflearn.regression(softmax, optimizer=sgd,
                                loss='categorical_crossentropy')

model = tflearn.DNN(regression, checkpoint_path='new-vgg-finetuning_01_500_with_best_val_001_500',
                    best_checkpoint_path='new-vgg-finetuning_01_500_with_best_val_001_500_BEST',max_checkpoints=3, tensorboard_verbose=2,
                    tensorboard_dir="./logs")

model_file = os.path.join(model_path, "vgg16.tflearn")
#model.load("/home/adoke/tf_tutorial/aircrafts/vgg16.tflearn", weights_only=True)
model.load("/home/adoke/tf_tutorial/aircrafts_new/new_train_val/new-vgg-variants_01_500.tflearn", weights_only=True)

`# Start finetuning
model.fit(X, Y, n_epoch=10, validation_set=(X_val,Y_val), shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200, snapshot_epoch=False, validation_batch_size=200, run_id='new-vgg-finetuning_01_500_best_val_001_500')


model.save('new-vgg-variants_01_500_best_val_001_500.tflearn')

X_test, Y_test = image_preloader('/home/adoke/tf_tutorial/aircrafts_new/from_start/a3_variants_test.txt', image_shape=(224, 224), mode='file',
                       categorical_labels=True, normalize=False,
                       files_extension=['.jpg', '.png'], filter_channel=True)

score = model.evaluate(X_test, Y_test)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
'''
