"""Digit classification in TensorFlow"""

#Import modules
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

#Dataset
mnist = input_data.read_data_sets("/tmp/data/",one_hot=True)

#Model parameters
learning_rate = .1
num_epochs = 501
batch_size = 128
display_epoch = 50

#Network parameters
input_size = 784
num_hidden_1 = 256
num_hidden_2 = 256
num_classes = 10

#input /output placeholders (none represents batch size - will be determined on graph execution)
X = tf.placeholder(tf.float32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, num_classes])

#set weights / biases
weights = {
        'w1': tf.Variable(tf.random_normal([input_size, num_hidden_1], stddev = 0.1)),
        'w2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], stddev = 0.1)),
        'output': tf.Variable(tf.random_normal([num_hidden_2, num_classes], stddev = 0.1))
        }

biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[num_hidden_1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[num_hidden_2])),
        'output': tf.Variable(tf.constant(0.1, shape=[num_classes]))
        }

#build the computational graph 
def assemble_network():
    layer_1 = tf.add(tf.matmul(X, weights['w1']),biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']),biases['b1'])
    return tf.add(tf.matmul(layer_2, weights['output']), biases['output'])

#define loss
prediction = assemble_network()
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits= prediction ))

#training operation
train = tf.train.AdamOptimizer().minimize(loss)

# 1 if prediction is correct, 0 if not
prediction_correct = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))

#accuracy metric
accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))

#run tensorflow session

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(1, num_epochs):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train, feed_dict = {X: batch_x, Y: batch_y})
    if epoch == 1 or epoch % display_epoch == 0:
        batch_loss, model_accuracy = sess.run([loss, accuracy], feed_dict = {X: batch_x, Y: batch_y})
        print("EPOCH " + str(epoch) + ", BATCH LOSS: " + \
        "{:.4f}".format(batch_loss) + ", TRAINING ACCURACY: "+ \
        "{:.3f}".format(model_accuracy)
        )

def testCustomImage(path):
    img = np.invert(Image.open(path).convert('L')).ravel()
    predicted_num = sess.run(tf.argmax(prediction, 1), feed_dict={X: [img]})
    print("Image prediction: ", np.squeeze(predicted_num))


testCustomImage("test_image.png")


"""QUESTIONS"""

"""What are the advantages of the different gradient descent algorithms?"""
"""How do we decide upon a loss function?"""
"""When would we use metrics other than 'accuracy'?"""
"""Good amount of epochs to start with?"""
"""Where do the Y labels come from?"""
"""Why did adding a standard deviation of .1 get better results for me?"""
