import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#number 1 to 10 data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))#预测值和真实值的差别，最大值的位置是不是等于真实值1
    # 的位置，如果等于就是对的，不等于就预测错的
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#计算data base中有多少个是对的，多少个是错的
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

#定义Weight变量，输入shape，返回变量的参数。其中我们使用tf.truncated_normal产生随机变量来进行初始化
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#同样的定义biase变量，输入shape ,返回变量的一些参数。其中我们使用tf.constant常量函数来进行初始化
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#定义卷积，tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，x是图片的所有参数，W是此卷积层的权重，然后定义步长strides
# =[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步，padding
# 采用的方式是SAME
def conv2d(x,W):
    #stride[1,x_movement,y_movement,1]
    #Must have strides[0]=strides[3]=1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。pooling 有两种，一种是最大值池化，一种是平均值池化，
# 本例采用的是最大值池化tf.max_pool()。池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]
def max_pool_2x2(x):
    # stride[1,x_movement,y_movement,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#define placeholder for inputs to network
#首先定义输入的placeholder
xs=tf.placeholder(tf.float32,[None,784])#28x28
ys=tf.placeholder(tf.float32,[None,10])
#定义了dropout的placeholder，它是解决过拟合的有效手段
keep_prob=tf.placeholder(tf.float32)
#接着呢，我们需要处理我们的xs，把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，
# 因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
x_image=tf.reshape(xs,[-1,28,28,1])
#print(x_image.shape)#[n_samples,28,28,1]

##conv1 layer##
W_conv1=weight_variable([5,5,1,32])#patch 5x5(卷积核),in size 1（输入厚度为1）,out size 32
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output size 28x28x32
h_pool1=max_pool_2x2(h_conv1)#output size 14x14x32

##conv2 layer##
W_conv2=weight_variable([5,5,32,64])#patch 5x5(卷积核),in size 32（输入厚度为32）,out size 64
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output size 14x14x64
h_pool2=max_pool_2x2(h_conv2)#output size 7x7x64

##func1 layer##
#此时weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64， 后面的输出size我们继续扩大，定为1024
W_fc1 =weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
#进入全连接层时, 通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出
# 结果展平.
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#[n_samples,7,7,64]->>[n_samples,7*7*64]
#然后将展平后的h_pool2_flat与本层的W_fc1相乘（注意这个时候不是卷积了）
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#如果考虑过拟合问题，可以加一个dropout的处理
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

##func2 layer##
#最后一层的构建，输入是1024，最后的输出是10个 (因为mnist数据集就是[0-9]十个类)，prediction就是我们最后的预测值
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
#用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# the error between prediciton and real data
#利用交叉熵损失函数来定义我们的cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                    reduction_indices=[1]))  #loss
#用tf.train.AdamOptimizer()作为优化器进行优化，使我们的cross_entropy最小
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#定义Session
sess = tf.Session()
# important step 初始化变量
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(1000):
    # 提取一部分的xs,ys样本
    batch_xs, batch_ys = mnist.train.next_batch(100)#从下载好的mnist=input_data.read_data_sets('MNIST_data',one_hot=True)，
    # data base 里提取100个，这里用train data
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:1})
    if i % 50 == 0:
       # to see the step improvement
       print(compute_accuracy(mnist.test.images,mnist.test.labels))#整个data分为train data和test data，在compute_accuracy
       # 里用test data