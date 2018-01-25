
import tensorflow as tf
import numpy as np
# ===========================
#   Actor and Critic DNNs
# ===========================


FIRST_LAYER = 150
SECOND_LAYER = 150

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.sess = sess
            self.s_dim = state_dim
            self.a_dim = action_dim
            self.learning_rate = learning_rate

            # Actor Network
            self.inputs, self.out = self.create_actor_network(scope)

            self.network_params = tf.trainable_variables()

            # Or loss and apply gradients
            self.td_error = tf.placeholder(dtype=tf.float32, name="advantage")
            self.action_history = tf.placeholder(dtype=tf.int64, name="action_history")
            
            self.picked_action_prob = tf.gather(self.out, self.action_history)
            self.loss = -tf.log(self.picked_action_prob) * self.td_error
            self.optimizer = tf.train.AdamOptimizer(learning_rate)#tf.train.RMSPropOptimizer(learning_rate) 
            self.train_op = self.optimizer.minimize(self.loss)
            
            self.num_trainable_vars = len(
                self.network_params)             

    def create_actor_network(self,scope):
        with tf.variable_scope(scope):
            # weights initialization
            w1_initial = np.random.normal(size=(self.s_dim,FIRST_LAYER)).astype(np.float32)
            w2_initial = np.random.normal(size=(FIRST_LAYER,SECOND_LAYER)).astype(np.float32)
            #w3_initial = np.random.normal(size=(SECOND_LAYER,self.a_dim)).astype(np.float32)

            #w1_initial = np.random.uniform(size=(self.s_dim,FIRST_LAYER),low= -0.01, high=0.01 ).astype(np.float32)  
            #w2_initial = np.random.uniform(size=(FIRST_LAYER,SECOND_LAYER),low= -0.01, high=0.01 ).astype(np.float32)  
            w3_initial = np.random.uniform(size=(SECOND_LAYER,self.a_dim),low= -0.001, high=0.001 ).astype(np.float32)
            # Placeholders
            inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim])
            # Layer 1 without BN
            w1 = tf.Variable(w1_initial)
            b1 = tf.Variable(tf.zeros([FIRST_LAYER]))
            z1 = tf.matmul(inputs,w1)+b1
            l1 = tf.nn.tanh(z1)
            # Layer 2 without BN
            w2 = tf.Variable(w2_initial)
            b2 = tf.Variable(tf.zeros([SECOND_LAYER]))
            z2 = tf.matmul(l1,w2)+b2
            l2 = tf.nn.tanh(z2)
            #output layer
            w3 = tf.Variable(w3_initial)
            b3 = tf.Variable(tf.zeros([self.a_dim]))
            z2 = tf.matmul(l2,w3) + b3
            out = tf.squeeze(tf.nn.softmax(z2))
            
            self.saver = tf.train.Saver()
            return inputs, out


    

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    

    def train(self, inputs, actions, td_error):
        self.sess.run([self.out, self.loss, self.train_op], feed_dict={
            self.inputs: inputs,
            self.td_error: td_error,
            self.action_history: actions            
        })

    
    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def save_actor(self):
        self.saver.save(self.sess,'actor_model.ckpt')
        #saver.save(self.sess,'actor_model.ckpt')
        print("Model saved in file: actor_model")

    
    def recover_actor(self):
        self.saver.restore(self.sess,'actor_model.ckpt')
        #saver.restore(self.sess,'critic_model.ckpt')
    