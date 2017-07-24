import tensorflow as tf
import os

tmpdir = os.path.dirname(os.path.realpath(__file__))
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

W = tf.Variable(tf.zeros([784, 10]), name="var_W")
b = tf.Variable(tf.zeros([10]), name="var_b")
tf.global_variables_initializer().run()

y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.train.write_graph(sess.graph_def, tmpdir, 'train.pb', as_text=False)