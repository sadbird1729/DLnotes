[ MachineLearingNotes(week1-4)](#head1)

- [ 基本概念](#head2)
- [ 单变量线性回归](#head3)
- [ 多变量线性回归](#head4)
- [ 逻辑回归](#head5)
- [ 神经网络](#head6)

# MachineLearingNotes(week1-4)

> 本篇总结参考吴恩达机器学习课程及各类博客，加上个人理解。方便日后复习。
>
> 图源吴恩达机器学习课程。

## 基本概念

**训练集，验证集，测试集。**

**监督学习：有数据，有标签。**

在训练集上训练出来算法模型，把测试集带入模型得到预测值，可以把测试集的预测值和测试集的真值（lable）做误差分析。

比如分类问题。

**无监督学习：有数据，无标签。**

比如聚类问题。



我们以一个房价预测例子展开学习。

## 单变量线性回归

单变量，即1个特征。假设以房子的面积来预测房价，面积为$x$，房价为$y$。

现在训练集有$m$个样本，$({{x}^{(i)}},{{y}^{(i)}})$表示第$i$个样本。

我们要做的是根据训练集，拟合出一条曲线：$h_\theta \left( x \right)=\theta_{0}+\theta_{1}x$。参数是$\theta_{0},\theta_{1}$。

![6168b654649a0537c67df6f2454dc9ba](.pic/MachineLearingNotes(week1-4)/6168b654649a0537c67df6f2454dc9ba.png)

为了评价这条曲线预测房价的性能，我们引入代价函数$J \left( \theta_0, \theta_1 \right) = \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}$，代价函数描述预测值$h_\theta \left( x \right)$与真实值$y$之间的误差，是误差的平方和。显而易见，为了让模型性能更好，我们目标是要让代价函数$J \left( \theta_0, \theta_1 \right)$最小。

接下来看下图$J \left( \theta_0, \theta_1 \right)$函数，在三维空间中我们可以找到一个使得$J \left( \theta_0, \theta_1 \right)$最小的点。

![27ee0db04705fb20fab4574bb03064ab](.pic/MachineLearingNotes(week1-4)/27ee0db04705fb20fab4574bb03064ab.png)

假设只有一个参数$\theta_{0}$，则模型为$h_\theta \left( x \right)=\theta_{1}x$，模型曲线和代价函数曲线（是二维的）如下图。
![2c9fe871ca411ba557e65ac15d55745d](.pic/MachineLearingNotes(week1-4)/2c9fe871ca411ba557e65ac15d55745d.png)

用梯度下降，从玫红色到浅蓝色，最后$J \left( \theta_1\right)$最小时，$h_\theta \left( x \right)$拟合效果最好。

梯度下降：${\theta_{j}}:={\theta_{j}}-\alpha \frac{\partial }{\partial {\theta_{j}}}J\left( \theta  \right)$。对$\theta_1$赋值，使$J \left( \theta\right)$得按梯度下降最快方向进行，一直迭代下去，最终得到局部最小值。

$\alpha$为学习率（learning rate），太高会无法收敛，一次一次越过最低点；太低会收敛很慢，一点一点挪动，需要很多步才能到达全局最低点。

## 多变量线性回归

多变量，即多个特征，$\left( {x_{1}},{x_{2}},...,{x_{n}} \right)$，现在特征有房子面积，房间数，楼层等等。

${x}_{j}^{\left( i \right)}$表示第$j$个特征的第$i$个样本。如

${x}^{(2)}\text{=}\begin{bmatrix} 1416\\\ 3\\\ 2\\\ 40 \end{bmatrix}$

表示共有4个特征，第2个样本中房子面积1416，房间数3，楼层2等等。

现在要支持多变量，拟合曲线为$h_{\theta} \left( x \right)={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$，其中$x_{0}=1$。

此时$h_{\theta} \left( x \right)={\theta^{T}}X$，$X$的维度是$m*(n+1)$，模型参数$\theta$是$n+1$维的向量。

多变量线性回归的代价函数是$J\left( {\theta_{0}},{\theta_{1}}...{\theta_{n}} \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( h_{\theta} \left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}}$，其中$h_{\theta}\left( x \right)=\theta^{T}X={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$。

```
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
```

梯度下降法为${{\theta}_{j}}={{\theta }_{j}}-\alpha\frac{\partial}{\partial {\theta_{j}}}\frac{1}{2m}\sum\limits_{i=1}^m\left(h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}={{\theta}_{j}}-\alpha\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta}}({{x}^{(i)}})-{{y}^{(i)}})}x_{j}^{(i)}$。

多项式回归：之前的线性回归拟合出来的曲线会都是直线/直面。。有时候我们需要曲线，比如二次方、三次方模型，如${{{h}}_{\theta}}(x)={{\theta}_{0}}+{{\theta}_{1}}(size)+{{\theta}_{2}}{{(size)}^{2}}+{{\theta}_{2}}{{(size)(floor)}}$，${{{h}}_{\theta}}(x)={{\theta}_{0}}+{{\theta}_{1}}(x_1)+{{\theta}_{2}}{{(x_2)}^{2}}+{{\theta}_{2}}{{(x_1)(x_2)}}$等。

![3a47e15258012b06b34d4e05fb3af2cf](.pic/MachineLearingNotes(week1-4)/3a47e15258012b06b34d4e05fb3af2cf.jpg)

## 逻辑回归

之前的单变量线性回归、多变量线性回归是回归类问题，下面开始说逻辑回归，二分类问题。预测变量$y$是离散的，在这里取0或者1。

那么再用$h_{\theta} \left( x \right)={\theta^{T}}X={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$作为拟合曲线就不合适了。由于$y$值为0或1，而线性回归的$h_{\theta} \left( x \right)$值范围很大，这样求误差$(h_{\theta} \left( x \right)-y)$是不太合适的，所以我们需要一个逻辑回归算法就是**S**形函数（**Sigmoid function**），公式为$g\left( z \right)=\frac{1}{1+{{e}^{-z}}}$，此时的拟合曲线为$h_\theta \left( x \right)=g\left(\theta^{T}X \right)$，即把$\theta^{T}X $的输入映射到0-1之间，作为$g\left( z \right)$的输入，这样求误差是更合理的。

![1073efb17b0d053b4f9218d4393246cc](.pic/MachineLearingNotes(week1-4)/1073efb17b0d053b4f9218d4393246cc.jpg)

$h_{\theta} \left( x \right)$的作用是，对于给定的输入变量，根据选择的参数计算输出变量=1的可能性（**estimated probablity**）即 $h_\theta \left( x \right)=P\left( y=1|x;\theta \right)$。例如，如果对于给定的$x$，通过已经确定的参数计算得出$h_\theta \left( x \right)=0.7$，则表示有70%的几率$y$为正向类，相应地为负向类的几率为1-0.7=0.3。

```
def sigmoid(z):  
   return 1 / (1 + np.exp(-z))
```

逻辑回归的代价函数：当我们将${h_\theta}\left( x \right)=\frac{1}{1+{e^{-\theta^{T}x}}}$带入到之前线性回归的代价函数中时，得到的代价函数将是一个非凸函数（**non-convexfunction**），有许多局部最小值，梯度下降算法不容易找到全局最小值。

![8b94e47b7630ac2b0bcb10d204513810](.pic/MachineLearingNotes(week1-4)/8b94e47b7630ac2b0bcb10d204513810.jpg)

那么重新定义逻辑回归的代价函数为$J\left(\theta\right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log\left({{h}_{\theta}}\left({{x}^{(i)}}\right)\right)-\left(1-{{y}^{(i)}}\right)\log\left(1-{{h}_{\theta}}\left({{x}^{(i)}}\right)\right)]}$ 。这是一个分段函数，当$y=1$时，即真值=1，我们想让预测值$${h_\theta}\left( x \right)$$趋向于1，则误差值越小，loss趋向于0，所以有下图左边，定义这部分代价函数为$(-{{y}^{(i)}}\log \left( {{h}_{\theta }}\left( {{x}^{(i)}} \right) \right)$。同理$y=0$如下图右边所示。

![ffa56adcc217800d71afdc3e0df88378](.pic/MachineLearingNotes(week1-4)/ffa56adcc217800d71afdc3e0df88378.jpg)

```
def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg
```

逻辑回归的梯度下降：$\theta_j:=\theta_j-\alpha\frac{\partial}{\partial\theta_j} J(\theta)=\theta_j-\alpha\frac{1}{m}\sum\limits_{i=1}^{m}{{\left({h_\theta}\left( \mathop{x}^{\left(i\right)}\right)-\mathop{y}^{\left(i\right)} \right)}}\mathop{x}_{j}^{(i)}$ ，这里需要带入${h_\theta}\left({{x}^{(i)}}\right)=\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}}$进行推导。

 

## 神经网络

因为线性回顾和逻辑回归都有一个缺点，当特征太多时，计算量会很大。

比如识别一张图片是否为汽车。假如我们只选用灰度图片，每个像素只有一个值，我们可以选取图片上的两个不同位置上的两个像素，即2个特征，然后训练一个逻辑回归算法利用这两个像素的值来判断图片上是否是汽车：

![2](.pic/MachineLearingNotes(week1-4)/2.jpg)

而现实情况肯定不可能只根据2个像素点就判断是否为汽车，那我们的范围再扩大一些，假使采用的局部的50x50像素的小图片，并将所有的像素视为特征，则会有 2500个特征，如果我们要进一步将两两特征组合构成一个多项式模型，则会有约${{2500}^{2}}/2$个（接近3百万个）特征。

普通的逻辑回归模型，不能有效地处理这么多的特征，这时候我们需要神经网络。

我们知道大脑中神经网络是大量神经元相互链接并通过电脉冲来交流的一个网络，根据神经元从树突接收信息、通过自己的算法模型做一些计算、再将信息从轴突发送出去的工作模式，我们建立了下图以逻辑回归模型作为自身学习模型的示例。

![c2233cd74605a9f8fe69fd59547d3853](.pic/MachineLearingNotes(week1-4)/c2233cd74605a9f8fe69fd59547d3853.jpg)

下图是3层神经网络，Layer1输入层，Layer2中间层，Layer3输出层。$a^{\left( j \right)}$是第$j$层矩阵，$a_{i}^{\left( j \right)}$是第$j$层的第$i$个激活单元。

${{\theta }^{\left( 1 \right)}}$代表从第一层映射到第二层的权重的矩阵。

${{z}^{\left( 2 \right)}}$是第2层$a^{\left( 2 \right)}$的输入矩阵，${{z}^{\left( 2 \right)}}={{\theta }^{\left( 1 \right)}}x$。${{\theta }^{\left( 1 \right)}}$是$3*4$维,$x$是$4*1$维,${{z}^{\left( 2 \right)}}$是$3*1$维。

则${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$，计算后添加$a_{0}^{\left( 2 \right)}=1$.

![5](.pic/MachineLearingNotes(week1-4)/5.png)

再令${{z}^{\left( 3 \right)}}={{\theta }^{\left( 2 \right)}}{{a}^{\left( 2 \right)}}$，则$h_\theta(x)={{a}^{\left( 3 \right)}}=g({{z}^{\left( 3 \right)}})$。

这里$x$是$4*1$维，即1个数据4个特征，要对整个数据集进行计算，则$X$是$4*m$维，${{z}^{\left( 2 \right)}}={{\Theta}^{\left(1\right)}}\times{{X}^{T}}$

![3](.pic/MachineLearingNotes(week1-4)/3.png)

对神经网络的理解：假如遮住第1层，只看第2层和第3层，这就像逻辑回归是一样的，以$a_0, a_1, a_2, a_3$作为输入，按照逻辑回归方式输出$h_\theta(x)$，只是把逻辑回归中的输入向量$\left[ x_1\sim {x_3} \right]$换成了中间层的$\left[ a_1^{(2)}\sim a_3^{(2)} \right]$，即$h_\theta(x)=g\left( \Theta_0^{\left( 2 \right)}a_0^{\left( 2 \right)}+\Theta_1^{\left( 2 \right)}a_1^{\left( 2 \right)}+\Theta_{2}^{\left( 2 \right)}a_{2}^{\left( 2 \right)}+\Theta_{3}^{\left( 2 \right)}a_{3}^{\left( 2 \right)} \right)$。

我们可以把$a_0, a_1, a_2, a_3$看作比$x_0, x_1, x_2, x_3$更厉害的输入，更高级的特征，$a_0, a_1, a_2, a_3$

是由$x$和$\theta$决定的，$\theta$是梯度下降的不断更新的，网络也是多层的。所以这些更高级的$a_0, a_1, a_2, a_3$比$x^{2}$更厉害，能更好的预测新数据，这就是神经网络比逻辑回归的优势，可以理解为套娃。



接下来讲神经网络里计算代价函数的方法，反向传播。