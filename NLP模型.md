

## RNN

### 函数

torch.nn.RNNCell()

### 参数

CNN中和RNN中batchSize的默认位置是不同的。

```
CNN中：batchsize的位置是position 0.
RNN中：batchsize的位置是position 1.
```

torch.nn.RNN()可以接受一个序列的输入，默认会传入一个全0的隐藏状态，也可以自己申明隐藏状态传入。

输入大小是三维**tensor[seq_len, batch_size, input_dim]**

    input_dim是输入的维度，比如是128
    batch_size是一次往RNN输入句子的数目，比如是5。
    seq_len是一个句子的最大长度，比如15
```
# 构造RNN网络，x的维度5，隐层的维度10,网络的层数2
rnn_seq = nn.RNN(5,10,2)  
# 构造一个输入序列，句长为6，batch_size是3，每个单词使用长度是5的向量表示。
x = torch.randn(6, 3, 5)
# out,ht = rnn_seq(x,h0) 
out,ht = rnn_seq(x) #h0可以指定或者不指定
```

问题1：这里out、ht的size是多少呢？
回答：out: 6 * 3 * 10, **out的输出维度[seq_len, batch_size, output_dim]**，ht: 2 * 3 * 10，ht的维度**[num_layers * num_directions, batch_size, hidden_size]**,如果是单向单层的RNN那么一个句子只有一个hidden。
问题2：out[-1]和ht[-1]是否相等？
回答：相等，隐藏单元就是输出的最后一个单元，可以想象，每个的输出其实就是那个时间步的隐藏单元。

### 必需参数的深入理解

1、RNN、GRU、LSTM的构造函数的三个必须参数理解——第一步：构造循环层对象

在创建循环层的时候，第一步是构造循环层，如下操作：

lstm = nn.LSTM(10, 20, 2)
构造函数的参数列表为如下：

class LSTM(RNNBase):

    '''参数Args:
        input_size:
        hidden_size:         
        num_layers: 
        bias:       
        batch_first: 
        dropout: 
        bidirectional:
    '''
（1）input_size:指的是每一个单词的特征维度，比如我有一个句子，句子中的每一个单词都用10维向量表示，则input_size就是10；

（2）hidden_size：指的是循环层中每一个LSTM内部单元的隐藏节点数目，这个是自己定义的，随意怎么设置都可以；

（3）num_layers：循环层的层数，默认是一层，这个根据自己的情况来定。

2、通过第一步构造的对象构造前向传播的过程——第二步：调用循环层对象，传入参数，并得到返回值

一般如下操作：

output, (hn, cn) = lstm(input, (h0, c0))
这里是以LSTM为例子来说的，

（1）输入参数

**input：必须是这样的格式（seq,batch,input_dim）**。**第一个seq指的是序列的长度**，这是根据自己的数据来定的，比如我的一个句子最大的长度是20个单词组成，那这里就是20,上面的例子是假设句子长度为5；**第二个是batch**，这个好理解，就是一次使用几条样本，比如3组样本；**第三个input_dim指的是每一个单词的向量维度**，需要注意的是，这个必须要和构造函数的第一个参数input_size保持一样的，上面的例子中是10.

（h0,c0）：指的是每一个循环层的初始状态，可以不指定，不指定的情况下全部初始化为0，这里因为是LSTM有两个状态需要传递，所以有两个，像普通的RNN和GRU只有一个状态需要传递，则只需要传递一个h状态即可，如下：

```
output, hn = rnn(input, h0)  # 普通rnn
output, hn = gru(input, h0)  # gru
```

这里需要注意的是传入的状态参数的维度，依然以LSTM来说：

**h0和c0的数据维度均是(num_layers * num_directions, batch, hidden_size)**，这是什么意思呢？

第一个num_layer指的是到底有基层循环层，这好理解，几层就应该有几个初始状态；

第二个num_directions指的是这个循环层是否是双向的（在构造函数中通过bidirectional参数指定哦），如果不是双向的，则取值为1，如果是双向的则取值为2；

第三个batch指的是每次数据的batch，和前面的batch保持一致即可；

最后一个hidden_size指的是循环层每一个节点内部的隐藏节点数，这个需要很好地理解循环神经网络的整个运算流程才行哦！

（2）输出结果

其实输出的结果和输入的是相匹配的，分别如下：

```
output, hn = rnn(input, h0)  # 普通rnn
output, hn = gru(input, h0)  # gru
output, (hn, cn) = lstm(input, (h0, c0)) # lstm
```

这里依然以lstm而言：

**output的输出维度：(seq_len, batch, num_directions * hidden_size)**，在上面的例子中，应该为（5,3,20），我们通过验证的确如此，需要注意的是，第一个维度是seq_len，也就是说每一个时间点的输出都是作为输出结果的，这和隐藏层是不一样的；

**hn、cn的输出维度：为(num_layers * num_directions, batch, hidden_size)**，在上面的例子中为（2,3,20），也得到了验证，我们发现这个跟序列长度seq_len是没有关系的，为什么呢，输出的状态仅仅是指的是最后一个循环层节点输出的状态。

如下图所示：

下面的例子是以普通的RNN来画的，所以只有一个状态h，没有状态c。

![img](https://img-blog.csdnimg.cn/20190831112855793.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3ODI1NDUx,size_16,color_FFFFFF,t_70)

3、几个重要的属性理解

不管是RNN，GRU还是lstm，内部可学习的参数其实就是几个权值矩阵，包括了偏置矩阵，那怎么查看这些学习到的参数呢？就是通过这几个矩阵来实现的

（1）weight_ih_l[k]：这表示的是输入到隐藏层之间的权值矩阵，其中K表示的第几层循环层，

若K=0，表示的是最下面的输入层到第一个循环层之间的矩阵，维度为(hidden_size, input_size)，如果k>0则表示第一循环层到第二循环层、第二循环层到第三循环层，以此类推，之间的权值矩阵，形状为(hidden_size, num_directions * hidden_size)。

（2）weight_hh_l[k]: 表示的是循环层内部之间的权值矩阵，这里的K表示的第几层循环层，取值为0,1,2,3,4... ...。形状为(hidden_size, hidden_size)

注意：循环层的层数取值是从0开始，0代表第一个循环层，1代表第二个循环层，以此类推。

（3）bias_ih_l[k]: 第K个循环层的偏置项，表示的是输入到循环层之间的偏置，维度为 (hidden_size)

（4）bias_hh_l[k]:第K个循环层的偏置项，表示的是循环层到循环层内部之间的偏置，维度为 (hidden_size)。



## LSTM

下面两个图可以看出RNN与LSTM的区别：

（1）RNN

![img](https://img-blog.csdn.net/20150915110014414?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（2）LSTM

![img](https://img-blog.csdn.net/20150915110046267?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

PS：

（1）部分图形含义如下：

![img](https://img-blog.csdn.net/20150915110203606?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（2）RNN与LSTM最大的区别在于LSTM中最顶层多了一条名为“cell state”的信息传送带，其实也就是信息记忆的地方；

3.LSTM的核心思想：

（1）理解LSTM的核心是“cell state”，暂且名为细胞状态，也就是上述图中最顶的传送线，如下：

![img](https://img-blog.csdn.net/20150915110556609?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（2）cell state也可以理解为传送带，个人理解其实就是整个模型中的记忆空间，随着时间而变化的，当然，传送带本身是无法控制哪些信息是否被记忆，起控制作用的是下面将讲述的控制门（gate）；

（3）控制门的结构如下：主要由一个sigmoid函数跟点乘操作组成；sigmoid函数的值为0-1之间，点乘操作决定多少信息可以传送过去，当为0时，不传送，当为1时，全部传送；

![img](https://img-blog.csdn.net/20150915111500108?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（4）LSTM中有3个控制门：输入门，输出门，记忆门；

4.LSTM工作原理：

（1）forget gate：选择忘记过去某些信息：

![img](https://img-blog.csdn.net/20150915121435938?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（2）input gate：记忆现在的某些信息：

![img](https://img-blog.csdn.net/20150915121744590?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（3）将过去与现在的记忆进行合并：

![img](https://img-blog.csdn.net/20150915121822934?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（4）output gate：输出

![img](https://img-blog.csdn.net/20150915121942641?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

PS：以上是标准的LSTM的结构，实际应用中常常根据需要进行稍微改善；

5.LSTM的改善

（1）peephole connections：为每个门的输入增加一个cell state的信号

![img](https://img-blog.csdn.net/20150915122451448?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（2）coupled forget and input gates：合并忘记门与输入门

![img](https://img-blog.csdn.net/20150915122741911?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)





LSTM结构中是一个神经网络，即上图的结构就是一个LSTM单元，里面的每个黄框是一个神经网络，这个网络的隐藏单元个数我们设为hidden_size，那么这个LSTM单元里就有4*hidden_size个参数。每个LSTM输出的都是向量，包括 C_t和h_t，它们的长度都是当前LSTM单元的hidden_size。

------

### 函数

torch.nn.LSTM(*args, **kwargs)

公式推导https://blog.csdn.net/songhk0209/article/details/71134698

### 参数列表

input_size：x的特征维度
hidden_size：隐藏层的特征维度(即输出维度)
num_layers：LSTM隐层的层数，默认为1
bias：bias: LSTM层是否使用偏置矩阵 偏置权值为 `b_ih` and `b_hh`.
            Default: ``True``（默认是使用的）
batch_first：True则输入输出的数据格式为 (batch, seq, feature)
dropout: 是否使用dropout机制，默认是0，表示不使用dropout，如果提供一个非0的数字，则表示在每一个LSTM层之后默认使用dropout，但是最后一个层的LSTM层不使用dropout。

bidirectional：True则为双向lstm，默认为False
输入：input, (h0, c0)
输出：output, (hn, cn)

h_0和C_0是第一个LSTM cell的隐藏层状态。h_n和C_n是最后一个LSTM cell的隐藏层状态。



**LSTM模型推导**

![img](https://img-blog.csdn.net/20150917143809332?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

1.LSTM模型的思想是将RNN中的每个隐藏单元换成了具有记忆功能的cell（如上图所示），其余的跟RNN一样；

2.每个cell的组成如下：

（1）输入节点（gc）：与RNN中的一样，接受上一个时刻点的隐藏节点的输出以及当前的输入作为输入，然后通过一个tanh的激活函数；

（2）输入门（ic）：起控制输入信息的作用，门的输入为上一个时刻点的隐藏节点的输出以及当前的输入，激活函数为sigmoid（原因为sigmoid的输出为0-1之间，将输入门的输出与输入节点的输出相乘可以起控制信息量的作用）；

（3）内部状态节点（sc）：输入为被输入门过滤后的当前输入以及前一时间点的内部状态节点输出，如图中公式；

（4）忘记门（fc）：起控制内部状态信息的作用，门的输入为上一个时刻点的隐藏节点的输出以及当前的输入，激活函数为sigmoid（原因为sigmoid的输出为0-1之间，将内部状态节点的输出与忘记门的输出相乘可以起控制信息量的作用）；

（5）输出门（oc）：起控制输出信息的作用，门的输入为上一个时刻点的隐藏节点的输出以及当前的输入，激活函数为sigmoid（原因为sigmoid的输出为0-1之间，将输出门的输出与内部状态节点的输出相乘可以起控制信息量的作用）；

3.LSTM层的计算可以表示如下（若干个cell组成一个LSTM层）：

![img](https://img-blog.csdn.net/20150917151355985?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

PS：公式1 中的Wih应改为Wgh；圆圈表示点乘


## GRU

![image-20221214210734504](C:\Users\ykx\AppData\Roaming\Typora\typora-user-images\image-20221214210734504.png)