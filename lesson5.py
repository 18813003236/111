
#---------------优化网络的几种方式：
#---------------1.更改数据集 
#---------------2.改代价函数 
#---------------3.更改神经网络，如：网络的层数，更改初始化方式(如正太化)，更改神经元个数，
#---------------------------------更改激活函数，加入dropout，防止过拟合
#-------------- 4.更改评价函数
#---------------5.优化器，（如：学习率，其他优化方式等）
#---------------6.迭代次数





##                          5-1MNIST数据集分类简单版   (继续优化结果)   
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
# #za载入数据集
#mnist=input_data.read_data_sets("MNIST_data",one_hot=True) 
##每个批次的大小(这里即是每次放100张图片到神经网络中训练)
#batch_size=100 
##计算一共有多少个批次
#n_batch=mnist.train.num_examples // batch_size   
##定义placeholder
##图片是28*28的，拉长之后就是784的向量
#x=tf.placeholder(tf.float32,[None,784])
#y=tf.placeholder(tf.float32,[None,10])
#keep_prob=tf.placeholder(tf.float32)
#lr=tf.Variable(0.001,dtype=tf.float32)
##创建神经网络
#W1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
#b1=tf.Variable(tf.zeros([500])+0.1)
#L1=tf.nn.tanh(tf.matmul(x,W1)+b1)
##定义学习率
#L1_drop=tf.nn.dropout(L1,keep_prob)
#
#W2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
#b2=tf.Variable(tf.zeros([300])+0.1)
#L2=tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
#L2_drop=tf.nn.dropout(L2,keep_prob)
#
#W3=tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
#b3=tf.Variable(tf.zeros([10])+0.1)
#
#prediction=tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)
#
####二次代价函数
##loss=tf.reduce_mean(tf.square(y-prediction))
##交叉熵代价函数
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#
####定义一个梯度下降法来进行训练的优化器
#train_step=tf.train.AdamOptimizer(lr).minimize(loss)
#
##chushihua bianliang 初始化变量
#init=tf.global_variables_initializer()
##准确率的判断，存放在一个布尔型列表中
##tf.equal（a,b）:求参数a和b是否相等，返回true和flase
##tf.argmax(y,1)：求一维张量中最大值的位置在哪
#correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
##求准确率
##tf.cast(correct_prediction,tf.float32)：把上面结果的布尔型转化为float32型的
##true转为1，flase转为0
##tf.reduce_mean：求平均值
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#
#with tf.Session() as sess:
#    sess.run(init)
#    for epoch in range(51):
#        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
#        for batch in range(n_batch):
#            #每个批次图片的数据保存在batch_xs中,标签保存在batch_ys中
#            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
#            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
#            
#        learning_rate=sess.run(lr)
#        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
#        print("Iter"+str(epoch)+",testing accuary:"+str(acc)+" ,learning rate="+str(learning_rate))
#    
    
    
    
    
#    
#                                    5-2 tensorboard的网络结构
        
#使用程序为3-2的MNIST数据集分类简单版      
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 #za载入数据集
mnist=input_data.read_data_sets("MNIST_data",one_hot=True) 
#每个批次的大小(这里即是每次放100张图片到神经网络中训练)
batch_size=50 
#计算一共有多少个批次
n_batch=mnist.train.num_examples // batch_size   


#---------------------操作tensorboard------
#命名空间
with tf.name_scope('input'):
    #定义两个placeholder
    #图片是28*28的，拉长之后就是784的向量
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    y=tf.placeholder(tf.float32,[None,10],name='y-input')
    
with tf.name_scope('layer'):
    #创建神经网络
    with tf.name_scope('weight'):
        W=tf.Variable(tf.zeros([784,10]),name='W')
    with tf.name_scope('biases'):
        b=tf.Variable(tf.zeros([10]),name='b')
    with tf.name_scope('wx_plus_b'):
        wx_plus_b=tf.matmul(x,W)+b
    with tf.name_scope('softmax'):
        prediction=tf.nn.softmax(wx_plus_b)
    
###二次代价函数
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.square(y-prediction))
    
with tf.name_scope('train'):    
    ###定义一个梯度下降法来进行训练的优化器
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#chushihua bianliang 初始化变量
init=tf.global_variables_initializer()
#准确率的判断，存放在一个布尔型列表中
#tf.equal（a,b）:求参数a和b是否相等，返回true和flase
#tf.argmax(y,1)：求一维张量中最大值的位置在哪
with tf.name_scope('accuracy'):
    
    with tf.name_scope('correct_prediction'):
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
#tf.cast(correct_prediction,tf.float32)：把上面结果的布尔型转化为float32型的
#true转为1，flase转为0
#tf.reduce_mean：求平均值
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))




with tf.Session() as sess:
    sess.run(init)
    
    writer=tf.summary.FileWriter('logs',sess.graph)
    
    for epoch in range(1):
        for batch in range(n_batch):
            #每个批次图片的数据保存在batch_xs中,标签保存在batch_ys中
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(epoch)+",testing accuary"+str(acc))  
    
    
    





    
##                                          5-3 查看tensorboard网络运行  
#    #使用程序为3-2的MNIST数据集分类简单版      
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
# #za载入数据集
#mnist=input_data.read_data_sets("MNIST_data",one_hot=True) 
##每个批次的大小(这里即是每次放100张图片到神经网络中训练)
#batch_size=50 
##计算一共有多少个批次
#n_batch=mnist.train.num_examples // batch_size   
#
##参数概要
#def variable_summaries(var):
#    with tf.name_scope('summaries'):
#        mean=tf.reduce_mean(var)
#        tf.summary.scalar('mean',mean)  #平均值
#        with tf.name_scope('stddev'):
#            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
#        tf.summary.scalar('stddev',stddev)     #标准差
#        tf.summary.scalar('max',tf.reduce_max(var))#最大值
#        tf.summary.scalar('min',tf.reduce_min(var))#最小值
#        tf.summary.histogram('histogram',var)#直方图
#
#
##---------------------操作tensorboard------
##命名空间
#with tf.name_scope('input'):
#    #定义两个placeholder
#    #图片是28*28的，拉长之后就是784的向量
#    x=tf.placeholder(tf.float32,[None,784],name='x-input')
#    y=tf.placeholder(tf.float32,[None,10],name='y-input')
#    
#with tf.name_scope('layer'):
#    #创建神经网络
#    with tf.name_scope('weight'):
#        W=tf.Variable(tf.zeros([784,10]),name='W')
#        variable_summaries(W)
#    with tf.name_scope('biases'):
#        b=tf.Variable(tf.zeros([10]),name='b')
#        variable_summaries(b)
#    with tf.name_scope('wx_plus_b'):
#        wx_plus_b=tf.matmul(x,W)+b
#    with tf.name_scope('softmax'):
#        prediction=tf.nn.softmax(wx_plus_b)
#    
####二次代价函数
#with tf.name_scope('loss'):
#    loss=tf.reduce_mean(tf.square(y-prediction))
#    #因为loss值不像W和b那么多，所以没有必要调用上面定义的函数
#    tf.summary.scalar('loss',loss)
   
    
#with tf.name_scope('train'):    
#    ###定义一个梯度下降法来进行训练的优化器
#    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
##chushihua bianliang 初始化变量
#init=tf.global_variables_initializer()
##准确率的判断，存放在一个布尔型列表中
##tf.equal（a,b）:求参数a和b是否相等，返回true和flase
##tf.argmax(y,1)：求一维张量中最大值的位置在哪
#with tf.name_scope('accuracy'):
#    
#    with tf.name_scope('correct_prediction'):
#        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
##求准确率
##tf.cast(correct_prediction,tf.float32)：把上面结果的布尔型转化为float32型的
##true转为1，flase转为0
##tf.reduce_mean：求平均值
#    with tf.name_scope('accuracy'):
#        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#
##合并所有的summary
#merged=tf.summary.merge_all()
#
#
#with tf.Session() as sess:
#    sess.run(init)
#    
#    writer=tf.summary.FileWriter('C://Users//zhanglian//logs',tf.get_default_graph())
#    
#    for epoch in range(51):
#        for batch in range(n_batch):
#            #每个批次图片的数据保存在batch_xs中,标签保存在batch_ys中
#            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
#            summary,_=sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
#            
#        writer.add_summary(summary,epoch)
#        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
#        
#        print("Iter"+str(epoch)+",testing accuary"+str(acc))  
    
    
    
    
    
    
    
    
    
#                                              5-4 tensorboard可视化  
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
#载入数据集
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#运行次数
max_step=1001

#图片数量
image_num=3000

#文件路径
DIR="D:/Tensorflow/"

#定义会话
sess=tf.Session()

#载入图片
embedding=tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False,name='embedding')

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)   #平均值
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)
        
#命名空间
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x-input')
    #正确的标签
    y=tf.placeholder(tf.float32,[None,10],name='y-input')
    
#显示图片
with tf.name_scope('input_reshape'):
    #tf.reshape(x,[-1,28,28,1])表示把x转化成[-1,28,28,1]这种形状
    #-1代表一个不确定的值，因为x的维度不确定28,28代表28行28列，
    #1代表维度是1，因为用的是给白图片，如果是彩色图片，则应该是3
    image_shaped_input=tf.reshape(x,[-1,28,28,1])
    
    tf.summary.image('input',image_shaped_input,10)#10代表方10张图片
    
#创建一个神经网络
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W=tf.Variable(tf.zeros([10]),name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b=tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b=tf.matmul(x,W)+b
    with tf.name_scope('softmax'):
        prediction=tf.nn.softmax(wx_plus_b)

#交叉熵代价函数        
with tf.name_scope('loss'):
    #交叉熵代价函数
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)

#使用梯度下降法
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
#初始化变量
    sess.run(tf.global_variables_initializer())
    with tf.name_scope('accuracy'):
        #结果存放在一个布尔型里表里面
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
        with tf.name_scope('accurary'):
            accurary=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            tf.summary.scalar('accurary',accurary)
            
#产生metadata文件
if tf.gfile.Exists(DIR+'projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR+'projector/projector/metadata.tsv')    
       
































 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    