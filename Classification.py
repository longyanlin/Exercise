import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#number 1 to 10 data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
   # add one more layer and return the output of this layer
   Weights = tf.Variable(tf.random_normal([in_size, out_size]))
   biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
   Wx_plus_b = tf.matmul(inputs, Weights) + biases
   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))#预测值和真实值的差别，最大值的位置是不是等于真实值1 的位置，如果等于就是对的，不等于就预测错的
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#计算data base中有多少个是对的，多少个是错的
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# 2.定义节点准备接收数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])#28*28
ys = tf.placeholder(tf.float32, [None, 10])


# add output layer 输入值是隐藏层 xs，在预测层输出 10个结果
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)


# the error between prediciton and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                    reduction_indices=[1]))  #loss

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# important step 对所有变量进行初始化
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(1000):
    # 提取一部分的xs,ys样本
    batch_xs, batch_ys = mnist.train.next_batch(100)#从下载好的mnist=input_data.read_data_sets('MNIST_data',one_hot=True)，data base 里提取100个，这里用train data
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
       # to see the step improvement
       print(compute_accuracy(mnist.test.images,mnist.test.labels))#整个data分为train data和test data，在compute_accuracy里用test data

