import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

learn=tf.contrib.learn

HIDDEN_SIZE = 30                            # LSTM中隐藏节点的个数。
NUM_LAYERS = 2                              # LSTM的层数。
TIMESTEPS = 10                              # 循环神经网络的训练序列长度。
TRAINING_STEPS = 10000                      # 训练轮数。
BATCH_SIZE = 32                             # batch大小。
TRAINING_EXAMPLES = 10000                   # 训练数据个数。
TESTING_EXAMPLES = 1000                     # 测试数据个数。
SAMPLE_GAP = 0.01                           # 采样间隔。

def generate_data(seq):
    X = []
    y = []
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入；第i + TIMESTEPS项作为输
    # 出。即用sin函数前面的TIMESTEPS个点的信息，预测第i + TIMESTEPS个点的函数值。
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X,y):
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*NUM_LAYERS)
    x_=tf.unstack(X,axis=1)

    output,_=tf.contrib.rnn.static_rnn(cell,x_,dtype=tf.float32)
    output=output[-1]
    prediction,loss=learn.models.linear_regression(output,y)
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)
    return prediction, loss, train_op

regressor=tf.contrib.learn.Estimator(model_fn=lstm_model)

# 用正弦函数生成训练和测试数据集合。
test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

regressor.fit(train_X,train_y,batch_size=BATCH_SIZE,steps=TRAINING_STEPS)

predicted=[[pred] for pred in regressor.predict(test_X)]
rmse=np.sqrt(((predicted-test_y)**2).mean())
print(rmse[0])