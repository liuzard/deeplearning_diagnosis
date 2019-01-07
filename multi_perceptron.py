import tensorflow as tf

class multi_perceptron(object):
    def __init__(self, n_input, n_hidden,n_output, transfer_function = tf.nn.tanh, optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output=n_output
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y_=tf.placeholder(tf.float32,[None,self.n_output])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x,self.weights['w1']),self.weights['b1']))
        self.y = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=tf.argmax(self.y_, 1))
        self.loss= tf.reduce_mean(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.optimizer = optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)



    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(tf.truncated_normal(shape=[self.n_input, self.n_hidden],dtype=tf.float32,stddev=0.1),name="w1")
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.truncated_normal(shape=[self.n_hidden, self.n_input],dtype=tf.float32,stddev=0.1),name="w2")
        # regularizer=tf.contrib.layers.l1_regularizer(0.0001)
        # tf.add_to_collection("losses",regularizer(all_weights['w1']))
        # tf.add_to_collection("losses", regularizer(all_weights['w2']))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights

    def partial_fit(self, X,Y):
        cost, opt = self.sess.run((self.loss, self.optimizer), feed_dict = {self.x: X,self.y_:Y})
        return cost
    def calc_accuracy(self,X,Y):
        return self.sess.run(self.accuracy,feed_dict = {self.x: X,self.y_:Y})
    def calc_total_cost(self, X,Y):
        return self.sess.run(self.loss, feed_dict = {self.x: X,self.y_:Y})

    def transform(self, X,):
        return self.sess.run(self.hidden, feed_dict = {self.x: X})

    # def generate(self, hidden=None):
    #     if hidden is None:
    #         hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
    #     return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
    #
    # def reconstruct(self, X):
    #     return self.sess.run(self.reconstruction, feed_dict = {self.x: X,
    #                                                            self.scale: self.training_scale
    #                                                            })

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

