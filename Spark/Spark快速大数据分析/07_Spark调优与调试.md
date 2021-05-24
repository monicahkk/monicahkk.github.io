### Spark调优与调试

#### 使用SparkConf配置Spark
- 当创建出一个 SparkContext 时，就需要创建出一个 SparkConf 的实例
```python 3
# 例 8-1：在 Python 中使用 SparkConf 创建一个应用
# 创建一个conf对象
conf = new SparkConf()
conf.set("spark.app.name", "My Spark App")
conf.set("spark.master", "local[4]")
conf.set("spark.ui.port", "36000") # 重载默认端口配置
# 使用这个配置对象创建一个SparkContext
sc = SparkContext(conf)

# SparkConf 实例包含用户要重载的配置选项的键值对
# 要使用创建出来的 SparkConf 对象，可以调用 set() 方法来添加配置项的设置，然后把这个对象传给 SparkContext 的构造方法
# SparkConf 类也包含了一小部分工具方法：可以调用 setAppName() 和 setMaster() 来分别设置spark.app.name 和 spark.master 的配置值
```
- Spark 允许通过 spark-submit 工具动态设置配置项
	spark-submit 工具为常用的 Spark 配置项参数提供了专用的标记，还有一个通用标记--conf 来接收任意 Spark 配置项的值
```linux
# 在运行时使用标记设置配置项的值
$ bin/spark-submit \
	--class com.example.MyApp \
	--master local[4] \
	--name "My Spark App" \
	--conf spark.ui.port=36000 \
	myApp.jar

# spark-submit 也支持从文件中读取配置项的值
# 默认情况下，spark-submit 脚本会在 Spark 安装目录中找到 conf/spark-defaults.conf 文件，尝试读取该文件中以空格隔开的键值对数据
# 也可以通过 spark-submit 的 --properties-File 标记，自定义该文件的路径

# 例 8-5：运行时使用默认文件设置配置项的值
$ bin/spark-submit \
	--class com.example.MyApp \
	--properties-file my-config.conf \
	myApp.jar

## Contents of my-config.conf ##
spark.master local[4]
spark.app.name "My Spark App"
spark.ui.port 36000
```
一旦传给了 SparkContext 的构造方法，应用所绑定的 SparkConf 就不可变了。_这意味着所有的配置项都必须在 SparkContext 实例化出来之前定下来_
- 同一个配置项可能在多个地方被设置了:Spark 有特定的优先级顺序来选择实际配置
	- 优先级最高的是在用户代码中显式调用 set() 方法设置的选项
	- 其次是通过 spark-submit 传递的参数
	- 再次是写在配置文件中的值，最后是系统的默认值

常用的Spark配置项的值

|选项 |默认值 |描述|
|:--:|:--:|:--:|
|spark.executor.memory(--executor-memory)|512m |为每个执行器进程分配的内存，格式与 JVM 内存字符串格式一样（例如 512m，2g）。关于本配置项的更多细节，请参阅 8.4.4 节|
|spark.executor.cores (--executor-cores) spark.cores.max(--total-executor-cores)|1（无）| 限制应用使用的核心个数的配置项。在 YARN 模式下，spark.executor.cores 会为每个任务分配指定数目的核心。在独立模式和 Mesos 模式下，spark.core.max 设置了所有执行器进程使用的核心总数的上限。参阅 8.4.4 节了解更多细节
|spark.speculation |false |设为 true 时开启任务预测执行机制。当出现比较慢的任务时，这种机制会在另外的节点上也尝试执行该任务的一个副本。打开此选项会帮助减少大规模集群中个别较慢的任务带来的影响
|spark.storage.blockManagerTimeoutIntervalMs|45000 |内部用来通过超时机制追踪执行器进程是否存活的阈值。对于会引发长时间垃圾回收（GC）暂停的作业，需要把这个值调到 100 秒（对应值为 100000）以上来防止失败。在 Spark 将来的版本中，这个配置项可能会被一个统一的超时设置所取代，所以请注意检索最新文档|
|spark.executor.extraJavaOptions
spark.executor.extraClassPath
spark.executor.extraLibraryPath|（空） |这三个参数用来自定义如何启动执行器进程的 JVM，分 别 用 来 添 加 额 外 的 Java 参 数、classpath 以 及 程 序库路径。使用字符串来设置这些参数（例如 spark.executor.extraJavaOptions="- XX:+PrintGCDetails-XX:+PrintGCTimeStamps"）。请注意，虽然这些参数可以让你自行添加执行器程序的 classpath，我们还是推荐使用spark-submit 的 --jars 标记来添加依赖，而不是使用这几个选项|
|spark.serializer |org.apache.spark.serializer.JavaSerializer|指定用来进行序列化的类库，包括通过网络传输数据或缓存数据时的序列化。默认的 Java 序列化对于任何可以被序列化的 Java 对象都适用，但是速度很慢。我们推荐在追求速度时使用 org.apache.spark.serializer.KryoSerializer 并且对 Kryo 进行适当的调优。该项可以配置为任何 org.apache.spark.Serializer 的子类|
|spark.[X].port |（任意值） |用来设置运行 Spark 应用时用到的各个端口。这些参数对于运行在可靠网络上的集群是很有用的。有效的 X 包括 driver、fileserver、broadcast、replClassServer、blockManager，以及 executor|
|spark.eventLog.enabled |false |设为 true 时，开启事件日志机制，这样已完成的 Spark作业就可以通过历史服务器（history server）查看。关于历史服务器的更多信息，请参考官方文档|
|spark.eventLog.dir |file:///tmp/spark-events|指开启事件日志机制时，事件日志文件的存储位置。这个值指向的路径需要设置到一个全局可见的文件系统中，比如 HDFS|

- 几乎所有的 Spark 配置都发生在 SparkConf 的创建过程中，但有一个重要的选项是个例外。
	- 你需要在 conf/spark-env.sh 中将环境变量 SPARK_LOCAL_DIRS 设置为用逗号隔开的存储位置列表，来指定 Spark 用来混洗数据的本地存储路径。这需要在独立模式和 Mesos 模式下设置

#### Spark执行的组成部分：作业、任务和步骤
Spark 执行的各个阶段
```txt
# 用作示例的源文件 input.txt
## input.txt ##
INFO This is a message with content
INFO This is some other content
（空行）
INFO Here are more messages
WARN This is a warning
（空行）
ERROR Something bad happened
WARN More details on the bad thing
INFO back to normal messages
```
```scala
// 在 Scala 版本的 Spark shell 中处理文本数据
// 读取输入文件
scala> val input = sc.textFile("input.txt")
// 切分为单词并且删掉空行
scala> val tokenized = input.
 | map(line => line.split(" ")).
 | filter(words => words.size > 0)
// 提取出每行的第一个单词（日志等级）并进行计数
scala> val counts = tokenized.
 | map(words = > (words(0), 1)).
 | reduceByKey{ (a,b) => a + b }
```
Spark 提供了 toDebugString() 方法来查看 RDD 的谱系
```scala
// 在 Scala 中使用 toDebugString() 查看 RDD
// 输出了 RDDinput 的相关信息。通过调用 sc.textFile() 创建出了这个 RDD
scala> input.toDebugString
res85: String =
(2) input.text MappedRDD[292] at textFile at <console>:13
 | input.text HadoopRDD[291] at textFile at <console>:13

scala> counts.toDebugString
res84: String =
(2) ShuffledRDD[296] at reduceByKey at <console>:17
 +-(2) MappedRDD[295] at map at <console>:17
 | FilteredRDD[294] at filter at <console>:15
 | MappedRDD[293] at map at <console>:15
 | input.text MappedRDD[292] at textFile at <console>:13
 | input.text HadoopRDD[291] at textFile at <console>:13
```
- 在调用行动操作之前，RDD 都只是存储着可以让我们计算出具体数据的描述信息
- 要触发实际计算，需要对 counts 调用一个行动操作，
```scala
// 如使用 collect() 将数据收集到驱动器程序中
scala> counts.collect()
res86: Array[(String, Int)] = Array((ERROR,1), (INFO,4), (WARN,2))
```
- Spark 调度器会创建出用于计算行动操作的 RDD 物理执行计划
	- RDD 的每个分区都会被物化出来并发送到驱动器程序中
	- 调度器从最终被调用行动操作的 RDD（在本例中是 counts）出发，向上回溯所有必须计算的 RDD
	- 递归向上生成计算所有必要的祖先 RDD 的物理计划
- 简单情况，调度器为有向图中的每个RDD 输出计算步骤，步骤中包括 RDD 上需要应用于每个分区的任务。然后以相反的顺序执行这些步骤，计算得出最终所求的 RDD
- 更复杂的情况，此时 RDD 图与执行步骤的对应关系并不一定是一一对应的
	- 当调度器进行流水线执行（pipelining），或把多个 RDD 合并到一个步骤中时（RDD 不需要混洗数据就可以从父节点计算出来时，调度器就会自动进行流水线执行）
		- 在物理执行时，执行计划输出的缩进等级与其父节点相同的 RDD 会与其父节点在同一个步骤中进行流水线执行
	- 除了流水线执行的优化，当一个 RDD 已经缓存在集群内存或磁盘上时，Spark 的内部调度器也会自动截短 RDD 谱系图
		- 在这种情况下，Spark 会“短路”求值，直接基于缓存下来的 RDD 进行计算
	- 还有一种截短 RDD 谱系图的情况发生在当 RDD 已经在之前的数据混洗中作为副产品物化出来时，哪怕该 RDD 并没有被显式调用 persist() 方法。
		- 这种内部优化是基于 Spark 数据混洗操作的输出均被写入磁盘的特性，同时也充分利用了 RDD 图的某些部分会被多次计算的事实
[counts的两个执行步骤](D:\MyDocuments\Typora\05-spark\Spark快速大数据分析\counts的2个执行步骤.png)
```scala
// 计算一个已经缓存过的 RDD
// 缓存RDD
scala> counts.cache()
// 第一次求值运行仍然需要两个步骤
scala> counts.collect()
res87: Array[(String, Int)] = Array((ERROR,1), (INFO,4), (WARN,2), (##,1), ((empty,2))
// 该次执行只有一个步骤
scala> counts.collect()
res88: Array[(String, Int)] = Array((ERROR,1), (INFO,4), (WARN,2), (##,1), ((empty,2))
```
特定的行动操作所生成的步骤的集合被称为一个作业
通过类似 count() 之类的方法触发行动操作，创建出由一个或多个步骤组成的作业
一旦步骤图确定下来，任务就会被创建出来并发给内部的调度器
调度器在不同的部署模式下会有所不同
物理计划中的步骤会依赖于其他步骤，如 RDD 谱系图所显示的那样。因此，这些步骤会以特定的顺序执行
一个物理步骤会启动很多任务，每个任务都是在不同的数据分区上做同样的事情。
任务内部的流程是一样的:
	1. 从数据存储（如果该 RDD 是一个输入 RDD）或已有 RDD（如果该步骤是基于已经缓存的数据）或数据混洗的输出中获取输入数据。
	2. 执行必要的操作来计算出这些操作所代表的 RDD。例如，对输入数据执行 filter() 和map() 函数，或者进行分组或归约操作
	3. 把输出写到一个数据混洗文件中，写入外部存储，或者是发回驱动器程序（如果最终RDD 调用的是类似 count() 这样的行动操作）
Spark 的大部分日志信息和工具都是以步骤、任务或数据混洗为单位的
**归纳**
Spark 执行时有下面所列的这些流程：
- **用户代码定义RDD的有向无环图**
RDD 上的操作会创建出新的 RDD，并引用它们的父节点，这样就创建出了一个图。
- **行动操作把有向无环图强制转译为执行计划**
当你调用 RDD 的一个行动操作时，这个 RDD 就必须被计算出来。这也要求计算出该RDD 的父节点。Spark 调度器提交一个作业来计算所有必要的 RDD。这个作业会包含一个或多个步骤，每个步骤其实也就是一波并行执行的计算任务。一个步骤对应有向无环图中的一个或多个 RDD，一个步骤对应多个 RDD 是因为发生了流水线执行。
- **任务于集群中调度并执行**
步骤是按顺序处理的，任务则独立地启动来计算出 RDD 的一部分。一旦作业的最后一个步骤结束，一个行动操作也就执行完毕了

#### 查找信息
Spark 在应用执行时记录详细的进度信息和性能指标
这些内容可以在两个地方找到：
_Spark 的网页用户界面_ 以及 _驱动器进程和执行器进程生成的日志文件中_
1. Spark网页用户界面
	- 默认情况下，它在驱动器程序所在机器的 4040 端口上
	- 对于 YARN 集群模式来说，应用的驱动器程序会运行在集群内部，你应该通过 YARN 的资源管理器来访问用户界面
	1.1. 作业页面：步骤与任务的进度和指标，以及更多内容
		- Jobs: 一个很重要的信息是正在运行的作业、步骤以及任务的进度情况。
			- 针对每个步骤，这个页面提供了一些帮助理解物理执行过程的指标
			- 经常用来评估一个作业的性能表现
			- 可以着眼于组成作业的所有步骤，看看是不是有一些步骤特别慢，或是在多次运行同一个作业时响应时间差距很大
		- Stages：定位性能问题
			- _数据倾斜_ 是导致性能问题的常见原因之一
				- 当看到少量的任务相对于其他任务需要花费大量时间的时候，一般就是发生了数据倾斜
				- 步骤页面可以帮助我们发现数据倾斜，只需要查看所有任务各项指标的分布情况就可以了
			- 还可以用来查看任务在其生命周期的各个阶段（读取、计算、输出）分别花费了多少时间
	1.2 存储页面：已缓存的RDD的信息
		- 当有人在一个 RDD 上调用了 persist() 方法，并且在某个作业中计算了该 RDD 时，这个 RDD 就会被缓存下来
		- 这个页面可以告诉我们到底各个 RDD 的哪些部分被缓存了，以及在各种不同的存储媒介（磁盘、内存等）中所缓存的数据量
		- 浏览这个页面并理解一些重要的数据集是否被缓存在了内存中，对我们是很有意义的
	1.3 执行器页面：应用中的执行器进程列表
		- 列出了应用中申请到的执行器实例，以及各执行器进程在数据处理和存储方面的一些指标
		- 用处之一在于确认应用可以使用你所预期使用的全部资源量
			- 调试问题时也最好先浏览这个页面，因为错误的配置可能会导致启动的执行器进程数量少于我们所预期的，显然也就会影响实际性能
			- 失败率很高的执行器节点可能表明这个执行器进程所在的物理主机的配置有问题或者出了故障
		- 另一个功能是使用线程转存（Thread Dump）按钮收集执行器进程的栈跟踪信息
			- 可视化呈现执行器进程的线程调用栈可以精确地即时显示出当前执行的代码
			- 在短时间内使用该功能对一个执行器进程进行多次采样，你就可以发现用户代码中消耗代价比较大的代码段
			- 这种信息分析通常可以检测出低效的用户代码
	1.4 环境页面：用来调试Spark配置项
		- 页面枚举了你的 Spark 应用所运行的环境中实际生效的配置项集合 —— 这里显示的配置项代表应用实际的配置情况。
		- 当你检查哪些配置标记生效时，这个页面很有用，尤其是当你同时使用了多种配置机制时。
		- 这个页面也会列出你添加到应用路径中的所有 JAR 包和文件，在追踪类似依赖缺失的问题时可以用到。
2. 驱动器进程和执行器进程的日志
	- 日志文件的具体位置取决于以下部署模式：
		- 在 Spark 独立模式下，所有日志会在独立模式主节点的网页用户界面中直接显示。这些日志默认存储于各个工作节点的 Spark 目录下的 work/ 目录中。
		- 在 Mesos 模式下，日志存储在 Mesos 从节点的 work/ 目录中，可以通过 Mesos 主节点用户界面访问。
		- 在 YARN 模式下
			- 最简单的收集日志的方法是使用 YARN 的日志收集工具（运行 yarn logs -applicationId <app ID>）来生成一个包含应用日志的报告 —— 只有在应用已经完全完成之后才能使用
			- 查看当前运行在 YARN 上的应用的日志，你可以从资源管理器的用户界面点击进入节点（Nodes）页面，然后浏览特定的节点，再从那里找到特定的容器。YARN 会提供对应容器中 Spark 输出的内容以及相关日志
	- 在默认情况下，Spark 输出的日志包含的信息量比较合适
	- 可以自定义日志行为，改变日志的默认等级或者默认存放位置
		- Spark 的日志系统是基于广泛使用的 Java 日志 log4j 实现的，使用 log4j 的配置方式进行配置：log4j 配置的示例文件位置 conf/log4j.properties.template
		- 要想自定义 Spark 的日志，首先需要把这个示例文件复制为 log4j.properties，然后就可以修改日志行为了
			- 修改根日志等级（即日志输出的级别门槛），默认值为 INFO
			- 更少的日志输出，可以把该值设为 WARN或者 ERROR
			- 设置了满意的日志等级或格式之后，你可以通过 spark-submit 的 --Files标记添加 log4j.properties 文件
			- 如果你在设置日志级别时遇到了困难，请首先确保你没有在应用中引入任何自身包含 log4j.properties 文件的 JAR 包。
				- Log4j 会扫描整个 classpath，以其找到的第一个配置文件为准，因此如果在别处先找到该文件，它就会忽略你自定义的文件

#### 关键性能考量
1. 并行度
输入 RDD 一般会根据其底层的存储系统选择并行度
并行度会从两方面 _影响程序的性能_：
	- 当并行度过低时，Spark 集群会出现资源闲置的情况
	- 当并行度过高时，每个分区产生的间接开销累计起来就会更大
评判并行度是否过高的标准包括任务是否是几乎在瞬间（毫秒级）完成的，或者是否观察到任务没有读写任何数据
两种方法来对操作的并行度进行 _调优_：
	- 在数据混洗操作时，使用参数的方式为混洗后的 RDD 指定并行度
	- 对于任何已有的 RDD，可以进行重新分区来获取更多或者更少的分区数
		- 重新分区操作通过 repartition() 实现, 会把 RDD 随机打乱并分成设定的分区数目
		- 如果确定要减少 RDD 分区，可以使用coalesce() 操作 (由于没有打乱数据，该操作比 repartition() 更为高效)
```python 3
# 假设我们从 S3 上读取了大量数据，然后马上进行 filter() 操作筛选掉数据集中的绝大部分数据。默认情况下，filter() 返回的 RDD 的分区数和其父节点一样，这样可能会产生很多空的分区或者只有很少数据的分区
# 以可以匹配数千个文件的通配字符串作为输入
>>> input = sc.textFile("s3n://log-files/2014/*.log")
>>> input.getNumPartitions()
35154
# 排除掉大部分数据的筛选方法
>>> lines = input.filter(lambda line: line.startswith("2014-10-17"))
>>> lines.getNumPartitions()
35154
# 在缓存lines之前先对其进行合并操作
>>> lines = lines.coalesce(5).cache()
>>> lines.getNumPartitions()
4
# 可以在合并之后的RDD上进行后续分析
>>> lines.count()
```
2. 序列化格式
当 Spark 需要通过网络传输数据，或是将数据溢写到磁盘上时，Spark 需要把数据序列化为二进制格式
序列化会在数据进行混洗操作时发生，此时有可能需要通过网络传输大量数据
	- 默认情况下，Spark 会使用 Java 内建的序列化库
	- Spark 也支持使用第三方序列化库 Kryo（https://github.com/EsotericSoftware/kryo ），可以提供比 Java 的序列化工具更短的序列化时间和更高压缩比的二进制表示，但不能直接序列化全部类型的对象
		- 使用即需要设置 spark.serializer 为 org.apache.spark.serializer.KryoSerializer
		- 为了获得最佳性能，还应该向 Kryo 注册你想要序列化的类
			- 注册类可以让 Kryo 避免把每个对象的完整的类名写下来，成千上万条记录累计节省的空间相当可观
			- 如果你想强制要求这种注册，可以把 spark.kryo.registrationRequired 设置为 true，这样 Kryo 会在遇到未注册的类时抛出错误
```java
// 使用 Kryo 序列化工具并注册所需类
val conf = new SparkConf()
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
// 严格要求注册类
conf.set("spark.kryo.registrationRequired", "true")
conf.registerKryoClasses(Array(classOf[MyClass], classOf[MyOtherClass]))
```
	- 不论是选用 Kryo 还是 Java 序列化，如果代码中引用到了一个没有扩展 Java 的 Serializable接口的类，你都会遇到 NotSerializableException
		- 很多 JVM 都支持通过一个特别的选项来帮助调试这一情况："-Dsun.io.serialization.extended DebugInfo=true”
			- 你可以通过设置 spark-submit 的 --driver-java-options 和 --executor-java-options 标记来打开这个选项
			- 一旦找到了有问题的类，最简单的解决方法就是把这个类改为实现了Serializable 接口的形式
			- 如果没有办法修改这个产生问题的类，你就需要采用一些高级的变通策略
				- 比如为这个类创建一个子类并实现 Java 的 Externalizable 接口（https://docs.oracle.com/javase/7/docs/api/java/io/Externalizable.html ）
				- 或者自定义 Kryo 的序列化行为
3. 内存管理
- 内存对 Spark 来说有几种不同的用途，理解并调优 Spark 的内存使用方法可以帮助优化Spark 的应用
	- RDD存储
		- 当调用 RDD 的 persist() 或 cache() 方法时，这个 RDD 的分区会被存储到缓存区中
		- Spark 会根据 spark.storage.memoryFraction 限制用来缓存的内存占整个 JVM 堆空间的比例大小
		- 如果超出限制，旧的分区数据会被移出内存
	- 数据混洗与聚合的缓存区
		- 当进行数据混洗操作时，Spark 会创建出一些中间缓存区来存储数据混洗的输出数据
		- 这些缓存区用来存储聚合操作的中间结果，以及数据混洗操作中直接输出的部分缓存数据
		- Spark 会尝试根据 spark.shuffle.memoryFraction 限定这种缓存区内存占总内存的比例
	- 用户代码
		- Spark 可以执行任意的用户代码，所以用户的函数可以自行申请大量内存
		- 如果一个用户应用分配了巨大的数组或者其他对象，那这些都会占用总的内存
		- 用户代码可以访问 JVM 堆空间中除分配给 RDD 存储和数据混洗存储以外的全部剩余空间
- 调整内存各区域比例：
	- 在默认情况下，Spark 会使用 60％的空间来存储 RDD，20% 存储数据混洗操作产生的数据，剩下的 20% 留给用户程序
	- 用户可以自行调节这些选项来追求更好的性能表现
	- 如果用户代码中分配了大量的对象，那么降低 RDD 存储和数据混洗存储所占用的空间可以有效避免程序内存不足的情况
- 还可以为一些工作负载改进缓存行为的某些要素
	- Spark默认的 cache() 操作会以 MEMORY_ONLY 的存储等级持久化数据 —— 如果缓存新的RDD 分区时空间不够，旧的分区就会直接被删除，当用到这些分区数据时，再进行重算
		- 有时以 MEMORY_AND_DISK 的存储等级调用 persist() 方法会获得更好的效果
			- 在这种存储等级下，内存中放不下的旧分区会被写入磁盘，当再次需要用到的时候再从磁盘上读取回来
			- 可能比重算各分区要低很多，也可以带来更稳定的性能表现
			- 当RDD 分区的重算代价很大（比如从数据库中读取数据）时，这种设置尤其有用
		- 另一个改进是缓存序列化后的对象而非直接缓存
			- 可以通过MEMORY_ONLY_SER 或者 MEMORY_AND_DISK_SER 的存储等级来实现这一点
			- 缓存序列化后的对象会使缓存过程变慢，因为序列化对象也会消耗一些代价，不过这可以显著减少 JVM 的垃圾回收时间，因为很多独立的记录现在可以作为单个序列化的缓存而存储。垃圾回收的代价与堆里的对象数目相关，而不是和数据的字节数相关
			- 这种缓存方式会把大量对象序列化为一个巨大的缓存区对象
			- 需要以对象的形式缓存大量数据（比如数 GB 的数据），或者是注意到了长时间的垃圾回收暂停，可以考虑配置这个选项
				- 这些暂停时间可以在应用用户界面中显示的每个任务的垃圾回收时间那一栏看到
4. 硬件供给
提供给 Spark 的硬件资源会显著影响应用的完成时间
- 影响集群规模的主要参数包括：
	- 分配给每个执行器节点的内存大小
	- 每个执行器节点占用的核心数
	- 执行器节点总数
	- 用来存储临时数据的本地磁盘数量
- 在各种部署模式下，执行器节点的内存都可以通过 spark.executor.memory 配置项或者 spark-submit 的 --executor-memory 标记来设置
- 执行器节点的数目以及每个执行器进程的核心数的配置选项则取决于各种部署模式
	- YARN 模式下，你可以通过 spark.executor.cores 或 --executor-cores 标记来设置执行器节点的核心数，通过 --num-executors 设置执行器节点的总数
	- 在 Mesos 和独立模式中，Spark 则会从调度器提供的资源中获取尽可能多的核心以用于执行器节点
	- Mesos 和独立模式也支持通过设置 spark.cores.max 项来限制一个应用中所有执行器节点所使用的核心总数
	- 本地磁盘则用来在数据混洗操作中存储临时数据
- 一般来说，更大的内存和更多的计算核心对 Spark 应用会更有用处
	- Spark 的架构允许线性伸缩；双倍的资源通常能使应用的运行时间减半
	- 在调整集群规模时，需要额外考虑的方面还包括是否在计算中把中间结果数据集缓存起来
		- 如果确实要使用缓存，那么内存中缓存的数据越多，应用的表现就会越好
		- Spark 用户界面中的存储页面会展示所缓存的数据中有哪些部分保留在内存中
		- 你可以从在小集群上只缓存一部分数据开始，然后推算缓存大量数据所需要的总内存量
- 除了内存和 CPU 核心，Spark 还要用到本地磁盘来存储数据混洗操作的中间数据，以及溢写到磁盘中的 RDD 分区数据
	- 使用大量的本地磁盘可以帮助提升 Spark 应用的性能
	- YARN 模式下，由于 YARN 提供了自己的指定临时数据存储目录的机制，Spark 的本地磁盘配置项会直接从 YARN 的配置中读取
	- 在独立模式下，我们可以在部署集群时，在 spark-env.sh 文件中设置环境变量 SPARK_LOCAL_DIRS，这样 Spark 应用启动时就会自动读取这个配置项的值
	- 运行的是 Mesos 模式，或者是在别的模式下需要重载集群默认的存储位置时，可以使用 spark.local.dir 选项来实现配置
	- 在所有情况下，本地目录的设置都应当使用由单个逗号隔开的目录列表。一般的做法是在磁盘的每个分卷中都为Spark 设置一个本地目录。写操作会被均衡地分配到所有提供的目录中。
	- 磁盘越多，可以提供的总吞吐量就越高
“越多越好”的原则在设置执行器节点内存时并不一定适用
	- 使用巨大的堆空间可能会导致垃圾回收的长时间暂停，从而严重影响 Spark 作业的吞吐量 —— 有时，使用较小内存（比如不超过 64GB）的执行器实例可以缓解该问题
	- 以使用较小内存的执行器实例不代表应用所使用的总资源一定会减少
	- 除了给单个执行器实例分配较小的内存，我们还可以用序列化的格式存储数据来减轻垃圾回收带来的影响

要深入了解 Spark 调优，请访问官方文档中的调优指南（http://spark.apache.org/docs/latest/tuning.html ）