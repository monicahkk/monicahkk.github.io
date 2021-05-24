### 基于MLlib的机器学习
#### 概述
- MLlib 的设计理念非常简单：把数据以 RDD 的形式表示，然后在分布式数据集上调用各种算法
- MLlib 引入了一些数据类型（比如点和向量），不过归根结底，MLlib 就是 RDD 上一系列可供调用的函数的集合
	- MLlib 中只包含能够在集群上运行良好的并行算法（部分不能并行执行的经典的机器学习算法没有包含在其中）
	- 一些较新的研究得出的算法因为适用于集群，也被包含在 MLlib 中，例如分布式随机森林算法（distributed random forests）、K-means|| 聚类、交替最小二乘算法（alternating least squares）等
	- 如果你要在许多小规模数据集上训练各机器学习模型，最好还是在各节点上使用单节点的机器学习算法库(Weka，http://www.cs.waikato.ac.nz/ml/weka/ 或 SciKit-Learn，http://scikit-learn.org/stable/ )，比如可以用 Spark 的 map() 操作在各节点上并行使用
	- 我们在机器学习流水线中也常常用同一算法的不同参数对小规模数据集分别训练，来选出最好的一组参数
- 在 Spark 中，你可以通过把参数列表传给 parallelize() 来在不同的节点上分别运不同的参数，而在每个节点上则使用单节点的机器学习库来实现

#### 系统要求
- MLlib 需要你的机器预装一些线性代数的库
	- 安装 gfortran 运行库 (http://spark.apache.org/docs/latest/mllib-guide.html)
	- 你要在 Python 中使用 MLlib，你需要安装 NumPy（http://www.numpy.org/ ）

#### 机器学习基础
- 机器学习算法尝试根据训练数据（training data）使得表示算法行为的数学目标最大化，并以此来进行预测或作出决定
- 机器学习问题分为几种，包括分类、回归、聚类
- 所有的学习算法都需要定义每个数据点的特征（feature）集，也就是传给学习函数的值
- 大多数算法都只是专为数值特征（具体来说，就是一个代表各个特征值的数字向量）定义的，因此提取特征并转化为特征向量是机器学习过程中很重要的一步
- 当数据已经成为特征向量的形式后，大多数机器学习算法都会根据这些向量优化一个定义好的数学函数
- 最后，大多数机器学习算法都有多个会影响结果的参数，所以现实中的机器学习流水线会训练出多个不同版本的模型，然后分别对其进行评估（evaluate）
```python 3
# 示例：垃圾邮件分类
# 程序使用了MLlib中的两个函数：HashingTF 与LogisticRegressionWithSGD，前者从文本数据构建词频（term frequency）特征向量，后者使用随机梯度下降法（Stochastic Gradient Descent，简称 SGD）实现逻辑回归

# Python 版垃圾邮件分类器
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithSGD

spam = sc.textFile("spam.txt")
normal = sc.textFile("normal.txt")

# 创建一个HashingTF实例来把邮件文本映射为包含10000个特征的向量
tf = HashingTF(numFeatures = 10000)
# 各邮件都被切分为单词，每个单词被映射为一个特征
spamFeatures = spam.map(lambda email: tf.transform(email.split(" ")))
normalFeatures = normal.map(lambda email: tf.transform(email.split(" ")))

# 创建LabeledPoint数据集分别存放阳性（垃圾邮件）和阴性（正常邮件）的例子
positiveExamples = spamFeatures.map(lambda features: LabeledPoint(1, features))
negativeExamples = normalFeatures.map(lambda features: LabeledPoint(0, features))
trainingData = positiveExamples.union(negativeExamples)
trainingData.cache() # 因为逻辑回归是迭代算法，所以缓存训练数据RDD

# 使用SGD算法运行逻辑回归
model = LogisticRegressionWithSGD.train(trainingData)

# 以阳性（垃圾邮件）和阴性（正常邮件）的例子分别进行测试。首先使用
# 一样的HashingTF特征来得到特征向量，然后对该向量应用得到的模型
posTest = tf.transform("O M G GET cheap stuff by sending money to ...".split(" "))
negTest = tf.transform("Hi Dad, I started studying Spark the other ...".split(" "))
print "Prediction for positive test example: %g" % model.predict(posTest)
print "Prediction for negative test example: %g" % model.predict(negTest)
```

#### 数据类型
- MLlib 包含一些特有的数据类型，它们位于 org.apache.spark.mllib 包（Java/Scala）或 pyspark.mllib（Python）内
- 主要的几个如下所列:
	- Vector: 
		- MLlib 既支持稠密向量也支持稀疏向量, 前者表示向量的每一位都存储下来，后者则只存储非零位以节约空间
		- 向量可以通过 mllib.linalg.Vectors 类创建出来
	- LabeledPoint:
		- 在诸如分类和回归这样的监督式学习（supervised learning）算法中，LabeledPoint 用来表示带标签的数据点
		- 它包含一个特征向量与一个标签（由一个浮点数表示），位置在 mllib.regression 包中
	- Rating
		- 用户对一个产品的评分，在 mllib.recommendation 包中，用于产品推荐
	- 各种Model类
		- 每个 Model 都是训练算法的结果，一般有一个 predict() 方法可以用来对新的数据点或数据点组成的 RDD 应用该模型进行预测
- 大多数算法直接操作由 Vector、LabeledPoint 或 Rating 对象组成的 RDD
1. **操作向量**
MLlib 中最常用的数据类型:
	- 向量有两种：稠密向量与稀疏向量
		- 稠密向量把所有维度的值存放在一个浮点数数组中 —— 例如，一个 100 维的向量会存储 100 个双精度浮点数
		- 稀疏向量只把各维度中的非零值存储下来 —— 当最多只有 10% 的元素为非零元素时，我们通常更倾向于使用稀疏向量（不仅是出于对内存使用的考虑，也是出于对速度的考虑）
			- 许多特征提取技术都会生成非常稀疏的向量，所以这种方式常常是一种很关键的优化手段
	- 创建向量的方式在各种语言中有一些细微的差别
		- 在 Python 中，你在 MLlib 中任意地方传递的 NumPy 数组都表示一个稠密向量，你也可以使用 mllib.linalg.Vectors 类创建其他类型的向量
``` python 3
# 用 Python 创建向量
from numpy import array
from pyspark.mllib.linalg import Vectors
# 创建稠密向量<1.0, 2.0, 3.0>
denseVec1 = array([1.0, 2.0, 3.0]) # NumPy数组可以直接传给MLlib
denseVec2 = Vectors.dense([1.0, 2.0, 3.0]) # 或者使用Vectors类来创建
# 创建稀疏向量<1.0, 0.0, 2.0, 0.0>；该方法只接收
# 向量的维度（4）以及非零位的位置和对应的值
# 这些数据可以用一个dictionary来传递，或使用两个分别代表位置和值的list
sparseVec1 = Vectors.sparse(4, {0: 1.0, 2: 2.0})
sparseVec2 = Vectors.sparse(4, [0, 2], [1.0, 2.0])
```
- 为了让 MLlib 保持在较小规模内：
	- Java 和 Scala 中，MLlib 的 Vector 类只是用来为数据表示服务的，而没有在用户 API 中提供加法和减法这样的向量的算术操作
	- 在 Python 中，你可以对稠密向量使用 NumPy 来进行这些数学操作，也可以把这些操作传给 MLlib
	- 如果你想在你的程序中进行向量的算术操作，可以使用一些第三方的库，比如 Scala 中的 Breeze（https://github.com/scalanlp/breeze ）或者 Java 中的 MTJ（https://github.com/fommil/matrix-toolkits-java ），然后再把数据转为 MLlib 向量

#### 算法
介绍 MLlib 中主要的算法，以及它们的输入和输出类型
1. 特征提取
	- mllib.feature 包中包含一些用来进行常见特征转化的类
	- 这些类中有从文本（或其他表示）创建特征向量的算法，也有对特征向量进行正规化和伸缩变换的方法
**TF-IDF**
- 词频—逆文档频率（简称 TF-IDF）是一种用来从文本文档（例如网页）中生成特征向量的简单方法
	- 为文档中的每个词计算两个统计值：
		- 一个是词频（TF）: 每个词在文档中出现的次数
		- 一个是逆文档频率（IDF）: 用来衡量一个词在整个文档语料库中出现的（逆）频繁程度
	- TF × IDF，展示了一个词与特定文档的相关程度
- MLlib 有两个算法可以用来计算 TF-IDF：HashingTF 和 IDF，都在 mllib.feature 包内
	- HashingTF 从一个文档中计算出给定大小的词频向量
		- 为了将词与向量顺序对应起来，它使用了哈希法（hasing trick）
			- HashingTF 使用每个单词对所需向量的长度 S 取模得出的哈希值，把所有单词映射到一个 0 到 S-1 之间的数字上 —— 可以保证生成一个 S 维的向量
			- 推荐将 S 设置在 218 到 220 之间
			- HashingTF 可以一次只运行于一个文档中，也可以运行于整个 RDD 中
			- 它要求每个“文档”都使用对象的可迭代序列来表示——例如 Python 中的 list 或 Java 中的 Collection
```python 3
在 Python 中使用 HashingTF
>>> from pyspark.mllib.feature import HashingTF
>>> sentence = "hello hello world"
>>> words = sentence.split() # 将句子切分为一串单词
>>> tf = HashingTF(10000) # 创建一个向量，其尺寸S = 10,000
>>> tf.transform(words)
SparseVector(10000, {3065: 1.0, 6861: 2.0})
>>> rdd = sc.wholeTextFiles("data").map(lambda (name, text): text.split())
>>> tfVectors = tf.transform(rdd) # 对整个RDD进行转化操作

# 在真实流水线中，你可能需要在把文档传给 TF 之前，对文档进行预处理并提炼单词。例如，你可能需要把所有的单词转为小写、去除标点、去除 ing这样的后缀。为了得到最佳结果，你可以在 map() 中调用一些类似 NLTK（http://www.nltk.org/）这样的单节点自然语言处理库
```
- 构建好词频向量之后，就可以使用 IDF 来计算逆文档频率，然后将它们与词频相乘
来计算 TF-IDF
	- 首先要对 IDF 对象调用 fit() 方法来获取一个 IDFModel，它代表语料库中的逆文档频率
	- 接下来，对模型调用 transform() 来把 TF 向量转为 IDF 向量
```python 3
# 在 Python 中使用 TF-IDF
from pyspark.mllib.feature import HashingTF, IDF
# 将若干文本文件读取为TF向量
rdd = sc.wholeTextFiles("data").map(lambda (name, text): text.split())
tf = HashingTF()
tfVectors = tf.transform(rdd).cache()
# 计算IDF，然后计算TF-IDF向量
idf = IDF()
idfModel = idf.fit(tfVectors)
tfIdfVectors = idfModel.transform(tfVectors)

# 注意，我们对 RDDtfVectors 调用了 cache() 方法，因为它被使用了两次（一次是训练IDF 模型时，一次是用 IDF 乘以 TF 向量时）
```
	1. 缩放
	多数机器学习算法都要考虑特征向量中各元素的幅值，并且在特征缩放调整为平等对待时表现得最好（例如所有的特征平均值为 0，标准差为 1）
		- 当构建好特征向量之后，你可以使用 MLlib 中的 StandardScaler 类来进行这样的缩放，同时控制均值和标准差
			- 你需要创建一个 StandardScaler，对数据集调用 fit() 函数来获取一个 StandardScalerModel（也就是为每一列计算平均值和标准差）
			- 然后使用这个模型对象的 transform() 方法来缩放一个数据集
```python 3
# 在 Python 中缩放向量
from pyspark.mllib.feature import StandardScaler
vectors = [Vectors.dense([-2.0, 5.0, 1.0]), Vectors.dense([2.0, 0.0, 1.0])]
dataset = sc.parallelize(vectors)
scaler = StandardScaler(withMean=True, withStd=True)
model = scaler.fit(dataset)
result = model.transform(dataset)
# 结果：{[-0.7071, 0.7071, 0.0], [0.7071, -0.7071, 0.0]}
```
	2. 正规化
	在一些情况下，在准备输入数据时，把向量正规化为长度 1 也是有用的
		- 使用 Normalizer类可以实现，只要使用 Normalizer.transform(rdd) 就可以了
		- 默认情况下，Normalizer 使用 L2 范式（也就是欧几里得距离），不过你可以给 Normalizer 传递一个参数 p 来使用 Lp范式
	3. Word2Vec
	Word2Vec（https://code.google.com/p/word2vec/ ）是一个基于神经网络的文本特征化算法，可以用来将数据传给许多下游算法
		- Spark 在 mllib.feature.Word2Vec 类中引入了该算法的一个实现
2. 统计
MLlib 通过 mllib.stat.Statistics 类中的方法提供了几种广泛使用的统计函数，这些函数可以直接在 RDD 上使用
- Statistics.colStats(rdd)
	- 计算由向量组成的 RDD 的统计性综述，保存着向量集合中每列的最小值、最大值、平均值和方差
- Statistics.corr(rdd, method)
	- 计算由向量组成的 RDD 中的列间的相关矩阵，使用皮尔森相关（Pearson correlation）或斯皮尔曼相关（Spearman correlation）中的一种（method 必须是 pearson 或 spearman中的一个）
- Statistics.corr(rdd1, rdd2, method)
	- 计算两个由浮点值组成的 RDD 的相关矩阵，使用皮尔森相关或斯皮尔曼相关中的一种（method 必须是 pearson 或 spearman 中的一个）
- Statistics.chiSqTest(rdd)
	- 计算由 LabeledPoint 对象组成的 RDD 中每个特征与标签的皮尔森独立性测试（Pearson’sindependencetest）结果
	- 返回一个ChiSqTestResult对象，其中有p值（p-value）、测试统计及每个特征的自由度
	- 标签和特征值必须是分类的（即离散值）
- 数值 RDD 还提供几个基本的统计函数，例如 mean()、stdev() 以及 sum()
- RDD 还支持 sample() 和 sampleByKey()，使用它们可以构建出简单而分层的数据样本
3. 分类与回归
- 分类与回归是监督式学习的两种主要形式
	- 监督式学习指算法尝试使用有标签的训练数据（也就是已知结果的数据点）根据对象的特征预测结果
- 分类和回归都会使用 mllib.regression 包中的 LabeledPoint 类
	- 一个 LabeledPoint 其实就是由一个 label（label 总是一个 Double 值，不过可以为分类算法设为离散整数）和一个 features 向量组成
		- 对于二元分类，MLlib 预期的标签为 0 或 1
		- 对于多元分类，MLlib 预期标签范围是从 0 到 C-1，其中 C 表示类别数量
- MLlib 包含多种分类与回归的算法，其中包括简单的线性算法以及决策树和森林算法
	1. 线性回归
	MLlib 也支持 L1 和 L2 的正则的回归，通常称为 Lasso 和 ridge 回归
	- 线性回归算法可以使用的类包括 mllib.regression.LinearRegressionWithSGD、LassoWithSGD以及 RidgeRegressionWithSGD
		- 涉及多个算法的问题，在类名中使用“With”来表明所使用的算法
		- 几个可以用来对算法进行调优的参数
			- numIterations: 要运行的迭代次数（默认值：100）
			- stepSize: 梯度下降的步长（默认值：1.0）
			- intercept: 是否给数据加上一个干扰特征或者偏差特征——也就是一个值始终为 1 的特征（默认值：false）
			- regParam: Lasso 和 ridge 的正规化参数（默认值：1.0）
		- 调用算法的方式在不同语言中略有不同
			- 在 Java 和 Scala 中，你需要创建一个 LinearRegressionWithSGD 对象，调用它的 setter 方法来设置参数，然后调用 run() 来训练模型
			- 在 Python 中，你需要使用类的方法 LinearRegressionWithSGD.train()，并对其传递键值对参数
			- 在这两种情况中，你都需要传递一个由 LabeledPoint 组成的 RDD
```python 3
# Python 中的线性回归
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
points = # (创建LabeledPoint组成的RDD)
model = LinearRegressionWithSGD.train(points, iterations=200, intercept=True)
print "weights: %s, intercept: %s" % (model.weights, model.intercept)

# 注意，在 Java 中，需要通过调用 .rdd() 方法把 JavaRDD 转为 Scala 中的 RDD 类。这种模式在 MLlib 中随处可见，因为 MLlib 方法被设计为既可以用 Java 调用，也可以用 Scala 调用
```
		- 一旦训练完成，所有语言中返回的 LinearRegressionModel 都会包含一个 predict() 函数，可以用来对单个特征向量预测一个值
			- RidgeRegressionWithSGD 和 LassoWithSGD 的行为类似，并且也会返回一个类似的模型类
			- 事实上，这种通过 setter 方法调节算法参数，然后返回一个带有 predict() 方法的 Model 对象的模式，在 MLlib 中很常见
		2. 逻辑回归
		逻辑回归是一种二元分类方法，用来寻找一个分隔阴性和阳性示例的线性分割平面
			- 有两种可以用来解决逻辑回归问题的算法：SGD 和 LBFGS5
				- LBFGS 一般是最好的选择，但是在早期的 MLlib 版本（早于 Spark 1.2）中不可用
					- LBFGS 是牛顿法的近似，它可以在比随机梯度下降更少的迭代次数内收敛
				- 这些算法通过 mllib.classification.LogisticRegressionWithLBFGS 和 WithSGD 类提供给用户，接口和 LinearRegressionWithSGD相似
				- 它们接收的参数和线性回归完全一样
			- 这两个算法中得出的 LogisticRegressionModel 以为每个点求出一个在 0 到 1 之间的得分，之后会基于一个阈值返回 0 或 1：
				- 默认情况下，对于 0.5，它会返回 1
					- 你可以通过 setThreshold() 改变阈值，也可以通过 clearThreshold() 去除阈值设置，这样的话predict() 就会返回原始得分
				- 对于阴性阳性示例各半的均衡数据集，我们推荐保留 0.5 作为阈值
				- 对于不平衡的数据集，你可以通过提升阈值来减少假阳性数据的数量（也就是提高精确率，但是也降低了召回率），也可以通过降低阈值来减少假阴性数据的数量
			- 在使用逻辑回归时，将特征提前缩放到相同范围内通常比较重要
				- 你可以使用 MLlib 的 StandardScaler 来实现特征缩放
		3. 支持向量机
		支持向量机（简称 SVM）算法是另一种使用线性分割平面的二元分类算法，同样只预期 0 或者 1 的标签
			- 通过 SVMWithSGD 类，我们可以访问这种算法，它的参数与线性回归和逻辑回归的参数差不多
			- 返回的 SVMModel 与 LogisticRegressionModel 一样使用阈值的方式进行预测
		4. 朴素贝叶斯
		朴素贝叶斯（Naive Bayes）算法是一种多元分类算法，它使用基于特征的线性函数计算将一个点分到各类中的得分
			- 这种算法通常用于使用 TF-IDF 特征的文本分类，以及其他一些应用
		- MLlib 实现了多项朴素贝叶斯算法，需要非负的频次（比如词频）作为输入特征
			- 可以通过 mllib.classification.NaiveBayes 类来使用朴素贝叶斯算法
			- 它支持一个参数 lambda（Python 中是 lambda_），用来进行平滑化
			- 你可以对一个由 LabeledPoint 组成的 RDD 调用朴素贝叶斯算法，对于 C 个分类，标签值范围在 0 至 C-1 之间
		- 返回的 NaiveBayesModel 让我们可以使用 predict() 预测对某点最合适的分类，也可以访问训练好的模型的两个参数：
			- 各特征与各分类的可能性矩阵 theta（对于 C 个分类和 D 个特征的情况，矩阵大小为 C × D）
			- 表示先验概率的 C 维向量 pi
		5. 决策树与随机森林
		- 可以使用 mllib.tree.DecisionTree 类中的静态方法 trainClassifier() 和trainRegressor() 来训练决策树
			- 和其他有些算法不同的是，Java 和 Scala 的 API 也使用静态方法，而不使用 setter 方法定制的 DecisionTree 对象
		- 训练方法接收如下所列参数:
			- data: 由 LabeledPoint 组成的 RDD
			- numClasses(仅用于分类时): 要使用的类别数量
			- impurity: 节点的不纯净度测量；对于分类可以为 gini 或 entropy，对于回归则必须为 variance
			- maxDepth: 树的最大深度（默认值：5）
			- maxBins: 在构建各节点时将数据分到多少个箱子中（推荐值：32）
			- categoricalFeaturesInfo: 一个映射表，用来指定哪些特征是分类的，以及它们各有多少个分类
				- 例如，如果特征1 是一个标签为 0 或 1 的二元特征，特征 2 是一个标签为 0、1 或 2 的三元特征，你就应该传递 {1: 2, 2: 3}
				- 如果没有特征是分类的，就传递一个空的映射表
		- MLlib 的在线文档（http://spark.apache.org/docs/latest/mllib-decision-tree.html ）中包含了对此处所使用算法的详细解释
		- 算法的开销会随训练样本数目、特征数量以及 maxBins 参数值进行线性增长
			- 对于大规模数据集，你可能需要使用较低的 maxBins 值来更快地训练模型，尽管这也会降低算法的质量
			- train() 方法会返回一个 DecisionTreeModel 对象
				- 可以使用这个对象的 predict() 方法来对一个新的特征向量预测对应的值，或者预测一个向量 RDD
				- 也可以使用 toDebugString() 来输出这棵树
					- 这个对象是可序列化的，所以你可以用 Java 序列化将它保存，然后在另一个程序中读取出来
			- 在 Spark 1.2 中，MLlib 在 Java 和 Scala 中添加了试验性的 RandomForest 类，可以用来构建一组树的组合，也被称为随机森林
				- 它可以通过 RandomForest.trainClassifier 和 trainRegressor 使用
		- 除了刚才列出的每棵树对应的参数外，RandomForest 还接收如下参数:
			- numTrees: 要构建的树的数量
				- 提高 numTrees 可以降低对训练数据过度拟合的可能性
			- featureSubsetStrategy: 在每个节点上作决定时需要考虑的特征数量
				- 可以是 auto（让库来自动选择）、all、sqrt、log2 以及 onethird
				- 越大的值所花费的代价越大
			- seed: 所使用的随机数种子
		- 随机森林算法返回一个 WeightedEnsembleModel 对象，其中包含几个决策树（在 weakHypotheses 字段中，权重由 weakHypothesisWeights 决定），可以对 RDD 或 vector 调用 predict()
		- 它还有一个 toDebugString 方法，可以打印出其中所有的树
		4. 聚类
		聚类算法是一种无监督学习任务，用于将对象分到具有高度相似性的聚类中
		**KMeans**
		MLlib 包含聚类中流行的 K-means 算法，以及一个叫作 K-means|| 的变种，可以为并行环境提供更好的初始化策略
			- K-means|| 的初始化过程与 K-means++ 在配置单节点时所进行的初始化过程非常相似
		- K-means 中最重要的参数是生成的聚类中心的目标数量 K
			- 最佳实践是尝试几个不同的 K 值，直到聚类内部平均距离不再显著下降为止
		- 除了 K 以外，MLlib 中的 K-means 还接收以下几个参数:
			- initializationMode: 用来初始化聚类中心的方法
				- 可以是“k-means||”或者“random”；
				- k-means||（默认值）一般会带来更好的结果，但是开销也会略高一些
			- maxIterations: 运行的最大迭代次数（默认值：100）
			- runs: 算法并发运行的数目
				- MLlib 的 K-means 算法支持从多个起点并发执行，然后选择最佳结果，这也是获取较好的整体模型的一种不错的方法（K-means 的运行可以停止在本地最小值上）
		- 当你要调用 K-means 算法时，你需要创建 mllib.clustering.KMeans对象（ 在 Java/Scala 中 ）或者调用 KMeans.train（ 在 Python 中 ）
			- 接收一个 Vector 组成的 RDD 作为参数
		- K-means 返回一个 KMeansModel 对象
			- 该对象允许你访问其clusterCenters 属性（聚类中心，是一个向量的数组）或者调用 predict() 来对一个新的向量返回它所属的聚类
			- 注意，predict() 总是返回和该点距离最近的聚类中心，即使这个点跟所有的聚类都相距很远
		5. 协同过滤与推荐
		协同过滤是一种根据用户对各种产品的交互与评分来推荐新产品的推荐系统技术
		**交替最小二乘**
		MLlib 中包含交替最小二乘（简称 ALS）的一个实现，这是一个协同过滤的常用算法，可以很好地扩展到集群上
			- 它位于 mllib.recommendation.ALS 类中
			- ALS 会为每个用户和产品都设一个特征向量，这样用户向量与产品向量的点积就接近于他们的得分
		- 接收下面所列这些参数:
			- rank: 使用的特征向量的大小
				- 更大的特征向量会产生更好的模型，但是也需要花费更大的计算代价（默认值：10）
				- iterations: 要执行的迭代次数（默认值：10）
				- lambda: 正则化参数（默认值：0.01）
				- alpha: 用来在隐式 ALS 中计算置信度的常量（默认值：1.0）
				- numUserBlocks，numProductBlocks: 切分用户和产品数据的块的数目，用来控制并行度
					- 你可以传递 -1 来让 MLlib 自动决定（默认行为）
		- 要使用 ALS 算法，你需要有一个由 mllib.recommendation.Rating 对象组成的 RDD, 其中每个包含一个用户 ID、一个产品 ID 和一个评分（要么是显式的评分，要么是隐式反馈)
			- 每个 ID 都需要是一个 32 位的整型值
				- 如果你的 ID 是字符串或者更大的数字，我们推荐你直接在 ALS 中使用 ID 的哈希值
					- 使有两个用户或者两个产品映射到同一个 ID 上，总体结果依然会不错
				- 还有一种办法是broadcast() 一张从产品 ID 到整型值的表，来赋给每个产品独特的 ID
		- ALS 返回一个 MatrixFactorizationModel 对象来表示结果
			- 可以调用 predict() 来对一个由 (userID, productID) 对组成的 RDD 进行预测评分
			- 也可以使用 model.recommendProducts (userId, numProducts) 来为一个给定用户找到最值得推荐的前 numProduct 个产品
		- 和 MLlib 中的其他模型不同，MatrixFactorizationModel 对象很大，为每个用户和产品都存储了一个向量
			- 这样我们就不能把它存到磁盘上，然后在另一个程序中读取回来
			- 不过，你可以把模型中生成的特征向量 RDD，也就是 model.userFeatures 和 model.productFeatures 保存到分布式文件系统上
		- ALS 有两个变种：显式评分（默认情况）和隐式反馈（通过调用 ALS.trainImplicit()而非 ALS.train() 来打开）
			- 用于显式评分时，每个用户对于一个产品的评分需要是一个得分（例如 1 到 5 星），而预测出来的评分也是得分
			- 而用于隐式反馈时，每个评分代表的是用户会和给定产品发生交互的置信度（比如随着用户访问一个网页次数的增加，评分也会提高），预测出来的也是置信度
			- 关于对隐式反馈使用 ALS 算法，Hu 等人所撰写的“Collaborative Filtering for Implicit Feedback Datasets,” ICDM 2008 中有更为详细的介绍
		6. 降维
			1. 主成分分析
			主要的降维技术是主成分分析（简称 PCA，https://en.wikipedia.org/wiki/Principal_component_analysis ）
				- 构建出正规化的相关矩阵，并使用这个矩阵的奇异向量和奇异值
				- PCA 目前只在 Java 和 Scala（MLlib 1.2）中可用
					- 要调用 PCA，你首先要使用 mllib.linalg.distributed.RowMatrix 类来表示你的矩阵
					- 然后存储一个由 Vector 组成的 RDD，每行一个
				- computePrincipalComponents() 返回的是 mllib.linalg.Matrix 对象， 是一个和 Vector 相似的表示稀疏矩阵的工具类
					- 可以调用 toArray 方法获取底层的数据
```java
// Scala 中的 PCA
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix

val points: RDD[Vector] = // ...
val mat: RowMatrix = new RowMatrix(points)
val pc: Matrix = mat.computePrincipalComponents(2)

// 将点投影到低维空间中
val projected = mat.multiply(pc).rows

// 在投影出的二维数据上训练k-means模型
val model = KMeans.train(projected, 10)
```
			2. 奇异值分解
			MLlib 也提供了低层的奇异值分解（简称 SVD）原语
			- SVD 会把一个 m ×n 的矩阵 A 分解成三个矩阵 A ≈ UΣV T，其中：
				- U 是一个正交矩阵，它的列被称为左奇异向量；
				- Σ 是一个对角线上的值均为非负数并降序排列的对角矩阵，它的对角线上的值被称为奇异值；
				- V 是一个正交矩阵，它的列被称为右奇异向量
			- 对于大型矩阵，通常不需要进行完全分解，只需要分解出靠前的奇异值和与之对应的奇异向量即可
				- 这样可以节省存储空间、降噪，并有利于恢复低秩矩阵
				- 如果保留前 k 个奇异值，那么结果矩阵就会是
					- U : m × k
					- Σ : k × k 
					- V : n × k
			- 要进行分解，应调用 RowMatrix 类的 computeSVD 方法
```java
// Scala 中的 SVD
// 计算RowMatrix矩阵的前20个奇异值及其对应的奇异向量
val svd: SingularValueDecomposition[RowMatrix, Matrix] =
	mat.computeSVD(20, computeU=true)
val U: RowMatrix = svd.U // U是一个分布式RowMatrix
val s: Vector = svd.s // 奇异值用一个局部稠密向量表示
val V: Matrix = svd.V // V是一个局部稠密矩阵
```
		7. 模型评估
		MLlib 包含一组试验性模型评估函数: 这些函数在 mllib.evaluation 包中

#### 一些提示与性能考量
1. 准备特征
	- 缩放输入特征
	- 正确提取文本特征
	- 为分类标上正确的标签
2. 配置算法
	- 在正规化选项可用时，MLlib 中的大多数算法都会在正则化打开时表现得更好（在预测准确度方面）
	- 大多数基于 SGD 的算法需要大约 100 轮迭代来获得较好的结果
	- MLlib 尝试提供合适的默认值，但是你应该尝试增加迭代次数，来看看是否能够提高精确度
		- 例如，使用 ALS 算法时，rank 的默认值 10 相对较低，所以你应该尝试提高这个值。确保在评估这些参数变化时将测试数据排除在训练集之外
3. 缓存RDD以重复使用
	- MLlib 中的大多数算法都是迭代的，对数据进行反复操作
		- 在把输入数据集传给MLlib 前使用 cache() 将它缓存起来是很重要的
		- 即使数据在内存中放不下，你也应该尝试 persist(StorageLevel.DISK_ONLY)
	- 在 Python 中，MLlib 会把数据集在从 Python 端传到 Java 端时在 Java 端自动缓存，因此没有必要缓存你的 Python RDD，除非你在自己的程序中还要用到它
	- 而在 Scala 和 Java 中，则需要由你来决定是否执行缓存操作
4. 识别稀疏程度
	- 当你的特征向量包含很多零时，用稀疏格式存储这些向量会为大规模数据集节省巨大的时间和空间
		- 在空间方面，当至多三分之二的位为非零值时，MLlib 的稀疏表示比它的稠密表示要小
		- 在数据处理代价方面，当至多 10% 的位为非零值时，稀疏向量所要花费的代价
		也会更小 ——（这是因为使用稀疏表示需要对向量中的每个元素执行的指令比使用稠密向量表示时要多）
		- 如果使用稀疏表示能够让你缓存使用稠密表示时无法缓存的数据，即使数据本身比较稠密，你也应当选择稀疏表示
5. 并行度
	- 对于大多数算法而言，你的输入 RDD 的分区数至少应该和集群的 CPU 核心数相当，这样才能达到完全的并行
		- 默认情况下 Spark 会为文件的每个“块”创建一个分区，而块一般为 64 MB
		- 可以通过向 SparkContext.textFile() 这样的函数传递分区数的最小值来改变默认行为 — —例如 sc.textFile("data.txt", 10)
		- 另一种方法是对 RDD 调用repartition(numPartitions) 来将 RDD 分区成 numPartitions 个分区
	- 始终可以通过Spark 的网页用户界面看到每个 RDD 的分区数
	- 同时，注意不要使用太多分区，因为这会增加通信开销

#### 流水线API
- 流水线就是一系列转化数据集的算法（要么是特征转化，要么是模型拟合）
	- 流水线的每个步骤都可能有参数（例如逻辑回归中的迭代次数）
	- 流水线 API 通过使用所选的评估矩阵评估各个集合，使用网格搜索自动找到最佳的参数集
- 流水线 API 使用 Spark SQL 中的 SchemaRDD 作为统一的数据集表示形式
	- SchemaRDD 中有多个有名字的列，这样要引用数据的不同字段就会比较容易
	- 流水线的各步骤可能会给 SchemaRDD 加上新的列（例如提取了特征的数据）
```Java
// 在 Scala 中使用流水线 API 实现垃圾邮件分类
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

// 用来表示文档的类，会被转入SchemaRDD中
case class LabeledDocument(id: Long, text: String, label: Double)
val documents = // （读取LabeledDocument的RDD）

val sqlContext = new SQLContext(sc)
import sqlContext._

// 配置该机器学习流水线中的三个步骤：分词、词频计数、逻辑回归；每个步骤
// 会输出SchemaRDD的一个列，并作为下一个步骤的输入列
val tokenizer = new Tokenizer() // 把各邮件切分为单词
	.setInputCol("text")
	.setOutputCol("words")
val tf = new HashingTF() // 将邮件中的单词映射为包含10000个特征的向量
	.setNumFeatures(10000)
	.setInputCol(tokenizer.getOutputCol)
	.setOutputCol("features")
val lr = new LogisticRegression() // 默认使用"features"作为输入列
val pipeline = new Pipeline().setStages(Array(tokenizer, tf, lr))

// 使用流水线对训练文档进行拟合
val model = pipeline.fit(documents)

// 或者，不使用上面的参数只对训练集进行一次拟合，也可以通过交叉验证对一批参数进行网格搜索，来找到最佳的模型
val paramMaps = new ParamGridBuilder()
	.addGrid(tf.numFeatures, Array(10000, 20000))
	.addGrid(lr.maxIter, Array(100, 200))
	.build() // 构建参数的所有组合
val eval = new BinaryClassificationEvaluator()
val cv = new CrossValidator()
	.setEstimator(lr)
	.setEstimatorParamMaps(paramMaps)
	.setEvaluator(eval)
val bestModel = cv.fit(documents)
```

官方文档（http://spark.apache.org/documentation.html ）