### Spark Streaming
- 即时处理收到的数据, 允许用户使用一套和批处理非常接近的 API 来编写流式计算应用
- Spark Streaming 使用离散化流（discretized stream）作为抽象表示，叫作 DStream
	- DStream 是随时间推移而收到的数据的序列
	- 在内部，每个时间区间收到的数据都作为 RDD 存在，而 DStream 是由这些 RDD 所组成的序列（因此得名“离散化”）
	- 可以从各种输入源创建，比如 Flume、Kafka 或者 HDFS
	- 创建出来的 DStream支持两种操作:
		- 转化操作（transformation）： 会生成一个新的DStream
		- 输出操作（output operation）： 可以把数据写入外部系统中
	- 提供了许多与 RDD 所支持的操作相类似的操作支持，还增加了与时间相关的新操作，比如滑动窗口
- 和批处理程序不同，Spark Streaming 应用需要进行额外配置来保证 24/7 不间断工作
	- 检查点（checkpointing）机制 - 是把数据存储到可靠文件系统（比如 HDFS）上的机制
		- Spark Streaming 用来实现不间断工作的主要方式
	- 遇到失败时如何重启应用
	- 如何把应用设置为自动重启模式
- Spark 1.1 来说，Spark Streaming 只可以在 Java 和 Scala 中使用, 类似的概念对 Python 也是适用的

#### 　一个简单的例子
例子: 从一台服务器的 7777 端口上收到一个以换行符分隔的多行文本，要从中筛选出包含单词 error 的行，并打印出来
- Spark Streaming 程序最好以使用 Maven 或者 sbt 编译出来的独立应用的形式运行
	- Spark Streaming 在 Maven 中以独立工件的形式提供，需要在工程中添加一些额外的 import 声明
```java
// Spark Streaming 的 Maven 索引
groupId = org.apache.spark
artifactId = spark-streaming_2.10
version = 1.2.0

// Scala 流计算 import 声明
import org.apache.spark.streaming.StreamingContext  
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.Duration
import org.apache.spark.streaming.Seconds

// Java 流计算 import 声明
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.Duration;
import org.apache.spark.streaming.Durations;
```
- StreamingContext 是流计算功能的主要入口
	- StreamingContext 会在底层创建出 SparkContext，用来处理数据
	- 其构造函数还接收用来指定多长时间处理一次新数据的批次间隔（batch interval）作为输入，这里我们把它设为 1 秒
- 接着，调用 socketTextStream() 来创建出基于本地 7777 端口上收到的文本数据的 DStream
- 然后把 DStream 通过 filter() 进行转化，只得到包含“error”的行
- 最后，使用输出操作 print() 把一些筛选出来的行打印出来
```java
// 用 Scala 进行流式筛选，打印出包含“error”的行
// 从SparkConf创建StreamingContext并指定1秒钟的批处理大小
val ssc = new StreamingContext(conf, Seconds(1))
// 连接到本地机器7777端口上后，使用收到的数据创建DStream
val lines = ssc.socketTextStream("localhost", 7777)
// 从DStream中筛选出包含字符串"error"的行
val errorLines = lines.filter(_.contains("error"))
// 打印出有"error"的行
errorLines.print()

// Java 进行流式筛选，打印出包含“error”的行
// 从SparkConf创建StreamingContext并指定1秒钟的批处理大小
JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(1));
// 以端口7777作为输入来源创建DStream
JavaDStream<String> lines = jssc.socketTextStream("localhost", 7777);
// 从DStream中筛选出包含字符串"error"的行
JavaDStream<String> errorLines = lines.filter(new Function<String, Boolean>() {
 public Boolean call(String line) {
 return line.contains("error");
 }});
// 打印出有"error"的行
errorLines.print();
```
_这只是设定好了要进行的计算，系统收到数据时计算就会开始_
- 要开始接收数据，必须显式调用 StreamingContext 的 start() 方法
	- 这样，Spark Streaming 就会开始把 Spark 作业不断交给下面的 SparkContext 去调度执行
	- 执行会在另一个线程中进行，所以需要调用 awaitTermination 来等待流计算完成，来防止应用退出
```java
// 用 Scala 进行流式筛选，打印出包含“error”的行
// 启动流计算环境StreamingContext并等待它"完成"
ssc.start()
// 等待作业完成
ssc.awaitTermination()

// 用 Java 进行流式筛选，打印出包含“error”的行
// 启动流计算环境StreamingContext并等待它"完成"
jssc.start();
// 等待作业完成
jssc.awaitTermination();
```
**注意，一个 Streaming context 只能启动一次，所以只有在配置好所有 DStream 以及所需要的输出操作之后才能启动**
```Linux
<!-- 在 Linux/Mac 操作系统上运行流计算应用并提供数据 -->
$ spark-submit --class com.oreilly.learningsparkexamples.scala.StreamingLogInput \
$ASSEMBLY_JAR local[4]

$ nc localhost 7777 # 使你可以键入输入的行来发送给服务器
<此处是你的输入>

<!-- Windows 用户可以使用 ncat（http://nmap.org/ncat/）命令来替代这里的 nc 命令。ncat 是nmap（http://nmap.org/）工具的一部分 -->
```
<!-- 如果你需要生成一些假的日志，可以运行本书 Git 仓库中的脚本 ./bin/fakelogs.sh 或者 ./bin/fakelogs.cmd 来把日志发给 7777 端口 -->

#### 架构与抽象
- Spark Streaming 使用“微批次”的架构，把流式计算当作一系列连续的小规模批处理来对待
	- Spark Streaming 从各种输入源中读取数据，并把数据分组为小的批次
	- 新的批次按均匀的时间间隔创建出来
	- 在每个时间区间开始的时候，一个新的批次就创建出来，在该区间内收到的数据都会被添加到这个批次中;在时间区间结束时，批次停止增长
	- 时间区间的大小是由批次间隔这个参数决定的。批次间隔一般设在 500 毫秒到几秒之间，由应用开发者配置
	- 每个输入批次都形成一个 RDD，以 Spark 作业的方式处理并生成其他的 RDD
	- 处理的结果可以以批处理的方式传给外部系统
[Spark Streaming 的高层次构架](D:\MyDocuments\Typora\05-spark\Spark快速大数据分析\Spark Streaming 的高层次构架.PNG)
- Spark Streaming 的编程抽象是离散化流, 它是一个 RDD 序列，每个 RDD 代表数据流中一个时间片内的数据
[DStream 是一个持续的RDD序列](D:\MyDocuments\Typora\05-spark\Spark快速大数据分析\DStream 是一个持续的RDD序列.PNG)
- 可以从外部输入源创建 DStream，也可以对其他 DStream 应用进行转化操作得到新的DStream
	- DStream 支持许多 RDD 支持的转化操作
	- DStream 还有“有状态”的转化操作，可以用来聚合不同时间片内的数据
```log
<!-- 在列举的简单的例子中，我们以从套接字中收到的数据创建出 DStream，然后对其应用filter() 转化操作 -->
<!-- 日志输出 -->
-------------------------------------------
Time: 1413833674000 ms
-------------------------------------------
71.19.157.174 - - [24/Sep/2014:22:26:12 +0000] "GET /error78978 HTTP/1.1" 404 505
...
-------------------------------------------
Time: 1413833675000 ms
-------------------------------------------
71.19.164.174 - - [24/Sep/2014:22:27:10 +0000] "GET /error78978 HTTP/1.1" 404 505
...
```
[DStream 及其转化关系](D:\MyDocuments\Typora\05-spark\Spark快速大数据分析\DStream及其转化关系)
	- DStream 还支持输出操作
		- 输出操作和 RDD 的行动操作的概念类似
		- Spark 在行动操作中将数据写入外部系统中，而 Spark Streaming 的输出操作在每个时间区间中周期性执行，每个批次都生成输出
- Spark Streaming 为每个输入源启动对应的接收器
	- 接收器以任务的形式运行在应用的执行器进程中，从输入源收集数据并保存为 RDD
	- 它们收集到输入数据后会把数据复制到另一个执行器进程来保障容错性（默认行为）
	- 数据保存在执行器进程的内存中，和缓存 RDD 的方式一样
		- 在 Spark 1.2 中，接收器也可以把数据备份到 HDFS 
		- 对于一些输入源，比如 HDFS，天生就是多份存储，所以 Spark Streaming 不会再作一次备份
	- 驱动器程序中的StreamingContext 会周期性地运行 Spark 作业来处理这些数据，把数据与之前时间区间中的RDD 进行整合
[Spark Streaming 在 Spark 各组件中的执行过程](D:\MyDocuments\Typora\05-spark\Spark快速大数据分析\Spark_Streaming在Spark各组件中的执行过程)
- Spark Streaming 对 DStream 提供的容错性与 Spark 为 RDD 所提供的容错性一致
	- 只要输入数据还在，它就可以使用 RDD 谱系重算出任意状态（比如重新执行处理输入数据的操作）
		- 默认情况下，收到的数据分别存在于两个节点上，这样 Spark 可以容忍一个工作节点的故障
		- 如果只用谱系图来恢复的话，重算有可能会花很长时间，因为需要处理从程序启动以来的所有数据
	- 因此, Spark Streaming 也提供了检查点机制，可以把状态阶段性地存储到可靠文件系统中（例如 HDFS 或者 S3）
		- 一般来说，你需要每处理 5-10 个批次的数据就保存一次
		- 在恢复数据时，Spark Streaming 只需要回溯到上一个检查点即可

#### 转化操作
- DStream 的转化操作可以分为无状态（stateless）和有状态（stateful）两种:
	- 在无状态转化操作中，每个批次的处理不依赖于之前批次的数据。
		- 之前讲的常见的RDD转化操作，例如map()、filter()、reduceByKey()等，都是无状态转化操作。
	- 相对地，有状态转化操作需要使用之前批次的数据或者是中间结果来计算当前批次的数据
		- 有状态转化操作包括基于滑动窗口的转化操作和追踪状态变化的转化操作
1. 无状态转化操作
无状态转化操作就是把简单的 RDD 转化操作应用到每个批次上，也就是转化 DStream 中的每一个 RDD
_针对键值对的 DStream 转化操作（比如 reduceByKey()）要添加 import StreamingContext._ 才能在 Scala 中使用_
_和 RDD 一样，在 Java 中需要通过 mapToPair() 创建出一个 JavaPairDStream 才能使用_
DStream无状态转化操作的例子（不完整列表）

|函数名称 |目的 |Scala示例|用来操作DStream[T]的用户自定义函数的函数签名|
|:--:|:--:|:--:|:--:|
|map() |对 DStream 中的每个元素应用给定函数，返回由各元素输出的元素组成的 DStream|ds.map(x => x + 1) |f: (T) -> U|
|flatMap() |对 DStream 中的每个元素应用给定函数，返回由各元素输出的迭代器组成的 DStream|ds.flatMap(x => x.split(" ")) |f: T -> Iterable[U]|
|filter() |返回由给定 DStream 中通过筛选的元素组成的 DStream|ds.filter(x => x != 1) |f: T -> Boolean|
|repartition() |改变 DStream 的分区数| ds.repartition(10) |N/A|
|reduceByKey() |将每个批次中键相同的记录归约| ds.reduceByKey((x, y) => x + y)|f: T, T -> T|
|groupByKey() |将每个批次中的记录根据键分组| ds.groupByKey() |N/A|

- 尽管这些函数看起来像作用在整个流上一样，但事实上每个 DStream 在内部是由许多 RDD（批次）组成，且无状态转化操作是分别应用到每个 RDD 上的
```java
// 在 Scala 中对 DStream 使用 map() 和 reduceByKey()
// 假设ApacheAccessingLog是用来从Apache日志中解析条目的工具类
val accessLogDStream = logData.map(line => ApacheAccessLog.parseFromLogLine(line))
val ipDStream = accessLogsDStream.map(entry => (entry.getIpAddress(), 1))
val ipCountsDStream = ipDStream.reduceByKey((x, y) => x + y)

// 在 Java 中对 DStream 使用 map() 和 reduceByKey()
// 假设ApacheAccessingLog是用来从Apache日志中解析条目的工具类
static final class IpTuple implements PairFunction<ApacheAccessLog, String, Long> {
	public Tuple2<String, Long> call(ApacheAccessLog log) {
		return new Tuple2<>(log.getIpAddress(), 1L);
	}
}
JavaDStream<ApacheAccessLog> accessLogsDStream =
	logData.map(new ParseFromLogLine());
JavaPairDStream<String, Long> ipDStream =
	accessLogsDStream.mapToPair(new IpTuple());
JavaPairDStream<String, Long> ipCountsDStream =
	ipDStream.reduceByKey(new LongSumReducer());
```
- 无状态转化操作也能在多个 DStream 间整合数据，不过也是在各个时间区间内
	- 可以在 DStream 上使用这些操作，这样就对每个批次分别执行了对应的 RDD 操作
```java
// 在 Scala 中连接两个 DStream
val ipBytesDStream =
	accessLogsDStream.map(entry => (entry.getIpAddress(), entry.getContentSize()))
val ipBytesSumDStream =
	ipBytesDStream.reduceByKey((x, y) => x + y)
val ipBytesRequestCountDStream =
	ipCountsDStream.join(ipBytesSumDStream)

// 在 Java 中连接两个 DStream
JavaPairDStream<String, Long> ipBytesDStream =
	accessLogsDStream.mapToPair(new IpContentTuple());
JavaPairDStream<String, Long> ipBytesSumDStream =
	ipBytesDStream.reduceByKey(new LongSumReducer());
JavaPairDStream<String, Tuple2<Long, Long>> ipBytesRequestCountDStream =
	ipCountsDStream.join(ipBytesSumDStream);
```
	- 我们还可以像在常规的 Spark 中一样使用 DStream 的 union() 操作将它和另一个 DStream的内容合并起来，也可以使用 StreamingContext.union() 来合并多个流
- 如果这些无状态转化操作不够用，DStream 还提供了一个叫作 transform() 的高级操作符，可以让你直接操作其内部的 RDD
	- transform() 操作允许你对 DStream 提供任意一个 RDD 到 RDD 的函数
	- 这个函数会在数据流中的每个批次中被调用，生成一个新的流
	- transform() 的一个常见应用就是重用你为 RDD 写的批处理代码
```java
// 在 Scala 中对 DStream 使用 transform()
val outlierDStream = accessLogsDStream.transform { rdd =>
 extractOutliers(rdd)
}

// 在 Java 中对 DStream 使用 transform()
JavaPairDStream<String, Long> ipRawDStream = accessLogsDStream.transform(
 new Function<JavaRDD<ApacheAccessLog>, JavaRDD<ApacheAccessLog>>() {
 public JavaPairRDD<ApacheAccessLog> call(JavaRDD<ApacheAccessLog> rdd) {
 return extractOutliers(rdd);
 }
});
```
- 也可以通过 StreamingContext.transform 或 DStream.transformWith(otherStream, func)来整合与转化多个 DStream
2. 有状态转化操作
- DStream 的有状态转化操作是跨时间区间跟踪数据的操作
	- 也就是说，一些先前批次的数据也被用来在新的批次中计算结果
	- 主要的两种类型: 滑动窗口 和 updateStateByKey()
		- 滑动窗口 以一个时间阶段为滑动窗口进行操作
		- updateStateByKey() 则用来跟踪每个键的状态变化（例如构建一个代表用户会话的对象）
- 有状态转化操作需要在你的 StreamingContext 中打开检查点机制来确保容错性
	- 可以通过传递一个目录作为参数给ssc.checkpoint() 来打开它
```
<!-- 设置检查点 -->
ssc.checkpoint("hdfs://...")
```
<!-- 进行本地开发时，你也可以使用本地路径（例如 /tmp）取代 HDFS -->
- **基于窗口的转化操作**
	- 基于窗口的操作会在一个比 StreamingContext 的批次间隔更长的时间范围内，通过整合多个批次的结果，计算出整个窗口的结果
	- 所有基于窗口的操作都需要两个参数: 
		- 窗口时长 以及 滑动步长
		- 两者都必须是StreamContext 的批次间隔的整数倍
		- 窗口时长控制每次计算最近的多少个批次的数据，其实就是最近的 windowDuration/batchInterval 个批次
			- 如果有一个以 10 秒为批次间隔的源DStream，要创建一个最近 30 秒的时间窗口（即最近 3 个批次），就应当把 windowDuration设为 30 秒
		- 滑动步长的默认值与批次间隔相等，用来控制对新的 DStream 进行计算的间隔
			- 如果源 DStream 批次间隔为 10 秒，并且我们只希望每两个批次计算一次窗口结果，就应该把滑动步长设置为 20 秒
	- 对 DStream 可以用的最简单窗口操作是 window()
		- 它返回一个新的 DStream 来表示所请求的窗口操作的结果数据
		- 换句话说，window() 生成的 DStream 中的每个 RDD 会包含多个批次中的数据，可以对这些数据进行 count()、transform() 等操作
[窗口市场 & 滑动步长](D:\MyDocuments\Typora\05-spark\Spark快速大数据分析\窗口市场&滑动步长)
```java
// 如何在 Scala 中使用 window() 对窗口进行计数
val accessLogsWindow = accessLogsDStream.window(Seconds(30), Seconds(10))
val windowCounts = accessLogsWindow.count()

// 如何在 Java 中使用 window() 对窗口进行计数
JavaDStream<ApacheAccessLog> accessLogsWindow = accessLogsDStream.window(
	Durations.seconds(30), Durations.seconds(10));
JavaDStream<Integer> windowCounts = accessLogsWindow.count()
```
- 尽管可以使用 window() 写出所有的窗口操作，Spark Streaming 还是提供了一些其他的窗口操作，让用户可以高效而方便地使用
	- 首先，reduceByWindow() 和 reduceByKeyAndWindow()让我们可以对每个窗口更高效地进行归约操作
		- 它们接收一个归约函数，在整个窗口上执行，比如 +
		- 除此以外，它们还有一种特殊形式，通过只考虑新进入窗口的数据和离开窗口的数据，让 Spark 增量计算归约结果
		- 这种特殊形式需要提供归约函数的一个逆函数，比如 + 对应的逆函数为 -
		- 对于较大的窗口，提供逆函数可以大大提高执行效率
[reduceByWindow()](D:\MyDocuments\Typora\05-spark\Spark快速大数据分析\reduceByWindow())
```java
// Scala 版本的每个 IP 地址的访问量计数
val ipDStream = accessLogsDStream.map(logEntry => (logEntry.getIpAddress(), 1))
val ipCountDStream = ipDStream.reduceByKeyAndWindow( 
	{(x, y) => x + y}, // 加上新进入窗口的批次中的元素
	{(x, y) => x - y}, // 移除离开窗口的老批次中的元素
	Seconds(30), // 窗口时长
	Seconds(10)) // 滑动步长

// Java 版本的每个 IP 地址的访问量计数
class ExtractIp extends PairFunction<ApacheAccessLog, String, Long> {
	public Tuple2<String, Long> call(ApacheAccessLog entry) {
		return new Tuple2(entry.getIpAddress(), 1L);
	}
}
class AddLongs extends Function2<Long, Long, Long>() {
	public Long call(Long v1, Long v2) { return v1 + v2; }
}
class SubtractLongs extends Function2<Long, Long, Long>() {
	public Long call(Long v1, Long v2) { return v1 - v2; }
}
JavaPairDStream<String, Long> ipAddressPairDStream = accessLogsDStream.mapToPair(
	new ExtractIp());

JavaPairDStream<String, Long> ipCountDStream = ipAddressPairDStream.
	reduceByKeyAndWindow(
	new AddLongs(), // 加上新进入窗口的批次中的元素
	new SubtractLongs() 
	// 移除离开窗口的老批次中的元素
	Durations.seconds(30), // 窗口时长
	Durations.seconds(10)); // 滑动步长
```
- DStream 还提供了 countByWindow() 和 countByValueAndWindow() 作为对数据进行
计数操作的简写
	- countByWindow() 返回一个表示每个窗口中元素个数的 DStream
	- countByValueAndWindow() 返回的 DStream 则包含窗口中每个值的个数
```java
// Scala 中的窗口计数操作
val ipDStream = accessLogsDStream.map{entry => entry.getIpAddress()}
val ipAddressRequestCount = ipDStream.countByValueAndWindow(Seconds(30), Seconds(10))
val requestCount = accessLogsDStream.countByWindow(Seconds(30), Seconds(10))

// Java 中的窗口计数操作
JavaDStream<String> ip = accessLogsDStream.map(
	new Function<ApacheAccessLog, String>() {
	public String call(ApacheAccessLog entry) {
	return entry.getIpAddress();
	}});
JavaDStream<Long> requestCount = accessLogsDStream.countByWindow(
	Dirations.seconds(30), Durations.seconds(10));
JavaPairDStream<String, Long> ipAddressRequestCount = ip.countByValueAndWindow(
	Dirations.seconds(30), Durations.seconds(10));
```
- **UpdateStateByKey转化操作**
	- 需要在 DStream 中跨批次维护状态（例如跟踪用户访问网站的会话）
	- updateStateByKey() 为我们提供了对一个状态变量的访问，用于键值对形式的DStream
		- 给定一个由（键，事件）对构成的 DStream，并传递一个指定如何根据新的事件更新每个键对应状态的函数，它可以构建出一个新的 DStream，其内部数据为（键，状态）对
	- 要使用 updateStateByKey()，提供了一个 update(events, oldState) 函数，接收与某键相关的事件以及该键之前对应的状态，返回这个键对应的新状态
		- 函数的签名:
			- events：是在当前批次中收到的事件的列表（可能为空）
			- oldState：是一个可选的状态对象，存放在 Option 内；如果一个键没有之前的状态，这个值可以空缺。
			- newState：由函数返回，也以 Option 形式存在；我们可以返回一个空的 Option 来表示想要删除该状态
	- updateStateByKey() 的结果会是一个新的 DStream，其内部的 RDD 序列是由每个时间区间对应的（键，状态）对组成的
```java
// 使用 updateStateByKey() 来跟踪日志消息中各 HTTP 响应代码的计数。这里的键是响应代码，状态是代表各响应代码计数的整数，事件则是页面访问

// 在 Scala 中使用 updateStateByKey() 运行响应代码的计数
def updateRunningSum(values: Seq[Long], state: Option[Long]) = {
	Some(state.getOrElse(0L) + values.size)
}

val responseCodeDStream = accessLogsDStream.map(log => (log.getResponseCode(), 1L))
val responseCodeCountDStream = responseCodeDStream.updateStateByKey(updateRunningSum _)

// 在 Java 中使用 updateStateByKey() 运行响应代码的计数
class UpdateRunningSum implements Function2<List<Long>,
	Optional<Long>, Optional<Long>> {
  public Optional<Long> call(List<Long> nums, Optional<Long> current) {
	long sum = current.or(0L);
		return Optional.of(sum + nums.size());
	}
};
JavaPairDStream<Integer, Long> responseCodeCountDStream = accessLogsDStream.mapToPair(
	new PairFunction<ApacheAccessLog, Integer, Long>() {
		public Tuple2<Integer, Long> call(ApacheAccessLog log) {
			return new Tuple2(log.getResponseCode(), 1L);
 }})
	.updateStateByKey(new UpdateRunningSum());
```

#### 输出操作
- 输出操作指定了对流数据经转化操作得到的数据所要执行的操作（例如把结果推入外部数
据库或输出到屏幕上）
<!-- 与RDD中的惰性求值类似，如果一个 DStream 及其派生出的 DStream 都没有被执行输出操作，那么这些 DStream 就都不会被求值。如果 StreamingContext 中没有设定输出操作，整个 context 就都不会启动 -->
- 常用的一种调试性输出操作是 print()，它会在每个批次中抓取 DStream 的前十个元素打印出来
- 一旦调试好了程序，就可以使用输出操作来保存结果了
	- Spark Streaming 对于 DStream 有与 Spark 类似的 save() 操作，它们接受一个目录作为参数来存储文件，还支持通过可选参数来设置文件的后缀名
		- 每个批次的结果被保存在给定目录的子目录中，且文件名中含有时间和后缀名
```java
// 在 Scala 中将 DStream 保存为文本文件
ipAddressRequestCount.saveAsTextFiles("outputDir", "txt")
```
	- 还有一个更为通用的 saveAsHadoopFiles() 函数，接收一个 Hadoop 输出格式作为参数
```java
// 在 Scala 中将 DStream 保存为 SequenceFile
val writableIpAddressRequestCount = ipAddressRequestCount.map {
	(ip, count) => (new Text(ip), new LongWritable(count)) }
writableIpAddressRequestCount.saveAsHadoopFiles[
	SequenceFileOutputFormat[Text, LongWritable]]("outputDir", "txt")

// 在 Java 中将 DStream 保存为 SequenceFile
JavaPairDStream<Text, LongWritable> writableDStream = ipDStream.mapToPair(
 new PairFunction<Tuple2<String, Long>, Text, LongWritable>() {
 public Tuple2<Text, LongWritable> call(Tuple2<String, Long> e) {
 return new Tuple2(new Text(e._1()), new LongWritable(e._2()));
 }});
class OutFormat extends SequenceFileOutputFormat<Text, LongWritable> {};
writableDStream.saveAsHadoopFiles(
 "outputDir", "txt", Text.class, LongWritable.class, OutFormat.class);
```
	- 还有一个通用的输出操作 foreachRDD()，它用来对 DStream 中的 RDD 运行任意计算
		- 和 transform() 有些类似，都可以让我们访问任意 RDD
		- 在 foreachRDD() 中，可以重用我们在 Spark 中实现的所有行动操作
			- 常见的用例之一是把数据写到诸如MySQL 的外部数据库中
			- 对于这种操作，Spark 没有提供对应的 saveAs() 函数，但可以使用 RDD 的 eachPartition() 方法来把它写出去
		- 为了方便，foreachRDD() 也可以提供给我们当前批次的时间，允许我们把不同时间的输出结果存到不同的位置
```java
// 在 Scala 中使用 foreachRDD() 将数据存储到外部系统中
ipAddressRequestCount.foreachRDD { rdd =>
	rdd.foreachPartition { partition =>
		// 打开到存储系统的连接（比如一个数据库的连接）
		partition.foreach { item =>
		// 使用连接把item存到系统中
		}
		// 关闭连接
	}
}
```

#### 输入源
- Spark Streaming 原生支持一些不同的数据源
	- 一些“核心”数据源已经被打包到 Spark Streaming 的 Maven 工件中
	- 其他的一些则可以通过 spark-streaming-kafka 等附加工件获取
如果你在设计一个新的应用，我们建议你从使用 HDFS 或 Kafka 这种简单的输入源开始
1. 核心数据源
所有用来从核心数据源创建 DStream 的方法都位于 StreamingContext 中
	- 我们已经在示例中用过其中一个： 套接字
	- 这里再讨论两个： 文件 和 Akka actor

	1. 文件流
	- Spark Streaming 支持从任意 Hadoop 兼容的文件系统目录中的文件创建数据流
		- 由于支持多种后端，这种方式广为使用，尤其是对于像日志这样始终要复制到 HDFS 上的数据
		- 要让 Spark Streaming 来处理数据
			- 我们需要为目录名字提供统一的日期格式
			- 文件也必须原子化创建（比如把文件移入 Spark 监控的目录）
				- 在文件系统中，文件重命名操作一般是原子化的
```java
// 例 10-29：用 Scala 读取目录中的文本文件流
val logData = ssc.textFileStream(logDirectory)

// 例 10-30：用 Java 读取目录中的文本文件流
JavaDStream<String> logData = jssc.textFileStream(logsDirectory);
```
<!-- 我们可以使用所提供的 ./bin/fakelogs_directory.sh 脚本来造出假日志。 -->
<!-- 如果有真实日志数据的话，也可以用 mv 命令将日志文件循环移入所监控的目录中。 -->
	- 除了文本数据，也可以读入任意 Hadoop 输入格式
		- 只需要将Key、Value 以及 InputFormat 类提供给 Spark Streaming 即可
```java
// 如果先前已经有了一个流处理作业来处理日志，并已经将得到的每个时间区间内传输的数据分别存储成了一个SequenceFile
// 用 Scala 读取目录中的 SequenceFile 流
ssc.fileStream[LongWritable, IntWritable,
	SequenceFileInputFormat[LongWritable, IntWritable]](inputDirectory).map {
	case (x, y) => (x.get(), y.get())
}
```
	2. Akka actor流
	- 另一个核心数据源接收器是 actorStream，它可以把 Akka actor（http://akka.io/ ）作为数据流的源
		- 要创建出一个 actor 流，需要创建一个 Akka actor，然后实现 org.apache.spark.streaming.receiver.ActorHelper 接口
		- 要把输入数据从 actor 复制到 Spark Streaming 中，需要在收到新数据时调用 actor 的 store() 函数
	- Akka actor 流不是很常见 
		- 流计算的文档（http://spark.apache.org/docs/latest/streaming-custom-receivers.html ）
		- Spark 中 的 ActorWordCount（https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/streaming/ActorWordCount.scala ）
2. 附加数据源
- 除核心数据源外，还可以用附加数据源接收器来从一些知名数据获取系统中接收的数据，这些接收器都作为 Spark Streaming 的组件进行独立打包了
- 需要在构建文件中添加额外的包才能使用它们
	- 现有的接收器包括 Twitter、Apache Kafka、Amazon Kinesis、Apache Flume，以及 ZeroMQ
	- 可以通过添加与 Spark 版本匹配的 Maven 工件 spark-streaming-[projectname]\_2.10 来引入这些附加接收器
	1. Apache Kafka
		- Apache Kafka（http://kafka.apache.org/ ）因其速度与弹性成为了一个流行的输入源
		- 使用Kafka 原生的支持，可以轻松处理许多主题的消息
		- 在工程中需要引入 Maven 工件 spark-streaming-kafka_2.10 来使用它
		- 包内提供的 KafkaUtils 对象可以在 StreamingContext 和 JavaStreamingContext 中以你的 Kafka 消息创建出 DStream
		- 由于 KafkaUtils 可以订阅多个主题，因此它创建出的 DStream 由成对的主题和消息组成
		- 要创建出一个流数据，需要使用 StreamingContext 实例、一个由逗号隔开的 ZooKeeper 主机列表字符串、消费者组的名字（唯一名字），以及一个从主题到针对这个主题的接收器线程数的映射表来调用createStream() 方法
```java
// 在 Scala 中用 Apache Kafka 订阅 Panda 主题
import org.apache.spark.streaming.kafka._
...
// 创建一个从主题到接收器线程数的映射表
val topics = List(("pandas", 1), ("logs", 1)).toMap
val topicLines = KafkaUtils.createStream(ssc, zkQuorum, group, topics)
StreamingLogInput.processLines(topicLines.map(_._2))

// 在 Java 中用 Apache Kafka 订阅 Panda 主题
import org.apache.spark.streaming.kafka.*;
...
// 创建一个从主题到接收器线程数的映射表
Map<String, Integer> topics = new HashMap<String, Integer>();
topics.put("pandas", 1);
topics.put("logs", 1);
JavaPairDStream<String, String> input =
 KafkaUtils.createStream(jssc, zkQuorum, group, topics);
input.print();
```
	2. Apache Flume
	- Spark 提供两个不同的接收器来使用 Apache Flume（http://flume.apache.org/ )
		- 推式接收器: 该接收器以 Avro 数据池的方式工作，由 Flume 向其中推数据
		- 拉式接收器:该接收器可以从自定义的中间数据池中拉数据，而其他进程可以使用 Flume 把数据推进该中间数据池
		- 两种方式都需要重新配置 Flume，并在某个节点配置的端口上运行接收器（不是已有的Spark 或者 Flume 使用的端口）
		- 要使用其中任何一种方法，都需要在工程中引入 Maven 工件 spark-streaming-flume_2.10
[Flume 接收器选项](D:\MyDocuments\Typora\05-spark\Spark快速大数据分析\Flume接收器选项)
	3. 推式接收器
	- 推式接收器的方法设置起来很容易，但是它不使用事务来接收数据
	- 在这种方式中，接收器以 Avro 数据池的方式工作，我们需要配置 Flume 来把数据发到 Avro 数据池
	- 我们提供的 FlumeUtils 对象会把接收器配置在一个特定的工作节点的主机名及端口号上
	- 这些设置必须和 Flume 配置相匹配
```java
// Flume 对 Avro 池的配置
a1.sinks = avroSink
a1.sinks.avroSink.type = avro
a1.sinks.avroSink.channel = memoryChannel
a1.sinks.avroSink.hostname = receiver-hostname
a1.sinks.avroSink.port = port-used-for-avro-sink-not-spark-port

// Scala 中的 FlumeUtils 代理
val events = FlumeUtils.createStream(ssc, receiverHostname, receiverPort)

// Java 中的 FlumeUtils 代理
JavaDStream<SparkFlumeEvent> events = FlumeUtils.createStream(ssc, receiverHostname, receiverPort)
```
	- 虽然这种方式很简洁，但缺点是没有事务支持
		- 这会增加运行接收器的工作节点发生错误时丢失少量数据的几率
		- 不仅如此，如果运行接收器的工作节点发生故障，系统会尝试从另一个位置启动接收器，这时需要重新配置 Flume 才能将数据发给新的工作节点。这样配置会比较麻烦
	4. 拉式接收器
	- 较新的方式是拉式接收器（在 Spark 1.1 中引入），它设置了一个专用的 Flume 数据池供 Spark Streaming 读取，并让接收器主动从数据池中拉取数据
		- 优点在于弹性较好，Spark Streaming 通过事务从数据池中读取并复制数据。在收到事务完成的通知前，这些数据还保留在数据池中
	- 需要先把自定义数据池配置为 Flume 的第三方插件
		- 安装插件的最新方法请参考Flume 文档的相关部分（https://flume.apache.org/FlumeUserGuide.html#installing-third-party-plugins ）
		- 由于插件是用 Scala 写的，因此需要把插件本身以及 Scala 库都添加到 Flume 插件中
```java
// Flume 数据池的 Maven 索引
groupId = org.apache.spark
artifactId = spark-streaming-flume-sink_2.10
version = 1.2.0
groupId = org.scala-lang
artifactId = scala-library
version = 2.10.4

// 当你把自定义 Flume 数据池添加到一个节点上之后，就需要配置 Flume 来把数据推送到这个数据池中
// Flume 对自定义数据池的配置
a1.sinks = spark
a1.sinks.spark.type = org.apache.spark.streaming.flume.sink.SparkSink
a1.sinks.spark.hostname = receiver-hostname
a1.sinks.spark.port = port-used-for-sync-not-spark-port
a1.sinks.spark.channel = memoryChannel

// 等到数据已经在数据池中缓存起来，就可以调用 FlumeUtils 来读取数据了
// 在 Scala 中使用 FlumeUtils 读取自定义数据池
val events = FlumeUtils.createPollingStream(ssc, receiverHostname, receiverPort)

// 在 Java 中使用 FlumeUtils 读取自定义数据池
JavaDStream<SparkFlumeEvent> events = FlumeUtils.createPollingStream(ssc,
	receiverHostname, receiverPort)
// 在 这 两 个 例 子 中，DStream 是 由 SparkFlumeEvent（https://spark.apache.org/docs/latest/api/java/org/apache/spark/streaming/flume/SparkFlumeEvent.html ）组成的
// 可以通过 event 访问下层的 AvroFlumeEvent

// 如果事件主体是 UTF-8 字符串
// Scala 中的 SparkFlumeEvent
// 假定flume事件是UTF-8编码的日志记录
val lines = events.map{e => new String(e.event.getBody().array(), "UTF-8")}
```
	5. 自定义输入源
	- 除了上述这些源，你也可以实现自己的接收器来支持别的输入源
		- 详细信息请参考 Spark 文档中的“自定义流计算接收器指南”（Streaming Custom Receivers guide，http://spark.apache.org/docs/latest/streaming-custom-receivers.html ）
3. 多数据源与集群规模
- 可以使用类似 union() 这样的操作将多个 DStream 合并
	- 有时，使用多个接收器对于提高聚合操作中的数据获取的吞吐量非常必要（如果只用一个接收器，可能会成为性能瓶颈）
	- 有时我们需要用不同的接收器来从不同的输入源中接收各种数据，然后使用 join 或 cogroup 进行整合
- 理解接收器是如何在 Spark 集群中运行的，对于我们使用多个接收器至关重要
	- 每个接收器都以 Spark 执行器程序中一个长期运行的任务的形式运行，因此会占据分配给应用的CPU 核心
	- 还需要有可用的 CPU 核心来处理数据。这意味着如果要运行多个接收器，就必须至少有和接收器数目相同的核心数，还要加上用来完成计算所需要的核心数
_不要在本地模式下把主节点配置为 "local" 或 "local[1]" 来运行 Spark 
Streaming 程序_
<!-- 这种配置只会分配一个 CPU 核心给任务，如果接收器运行在这样的配置里，就没有剩余的资源来处理收到的数据了。至少要使用"local[2]" 来利用更多的核心 -->

#### 24/7不间断运行
Spark Streaming 的一大优势在于它提供了强大的容错性保障
要不间断运行 Spark Streaming 应用，需要一些特别的配置。第一步是设置好诸如 HDFS 或Amazon S3 等可靠存储系统中的检查点机制。不仅如此，我们还需要考虑驱动器程序的容错性（需要特别的配置代码）以及对不可靠输入源的处理。
1. 检查点机制
- 检查点机制是我们在 Spark Streaming 中用来保障容错性的主要机制 —— 它可以使 Spark Streaming 阶段性地把应用数据存储到诸如 HDFS 或 Amazon S3 这样的可靠存储系统中，以供恢复时使用
- 具体来说，检查点机制主要为以下两个目的服务：
	- 控制发生失败时需要重算的状态数
		- Spark Streaming 可以通过转化图的谱系图来重算状态，检查点机制则可以控制需要在转化图中回溯多远
	- 提供驱动器程序容错
		- 如果流计算应用中的驱动器程序崩溃了，你可以重启驱动器程序并让驱动器程序从检查点恢复，这样 Spark Streaming 就可以读取之前运行的程序处理数据的进度，并从那里继续
```
# 配置检查点
ssc.checkpoint("hdfs://...")

# 即便是在本地模式下，如果你尝试运行一个有状态操作而没有打开检查点机制，Spark Streaming 也会给出提示。 此时，你需要使用一个本地文件系统中的路径来打开检查点。
# 不过，在所有的生产环境配置中，你都应当使用诸如 HDFS、S3 或者网络文件系统这样的带备份的系统
```
2. 驱动器程序容错
- 驱动器程序的容错要求我们以特殊的方式创建 StreamingContext
- 我们需要把检查点目录提供给StreamingContext。
	- 与直接调用 new StreamingContext 不同，应该使用 StreamingContext.getOrCreate() 函数。
```java
// 例 10-43：用 Scala 配置一个可以从错误中恢复的驱动器程序
def createStreamingContext() = {
 ...
 val sc = new SparkContext(conf)
 // 以1秒作为批次大小创建StreamingContext
 val ssc = new StreamingContext(sc, Seconds(1))
 ssc.checkpoint(checkpointDir)
}
...
val ssc = StreamingContext.getOrCreate(checkpointDir, createStreamingContext _)

// 例 10-44：用 Java 配置一个可以从错误中恢复的驱动器程序
JavaStreamingContextFactory fact = new JavaStreamingContextFactory() {
 public JavaStreamingContext call() {
 ...
 JavaSparkContext sc = new JavaSparkContext(conf);
 // 以1秒作为批次大小创建StreamingContext
 JavaStreamingContext jssc = new JavaStreamingContext(sc, Durations.seconds(1));
 jssc.checkpoint(checkpointDir);
 return jssc;
 }};
JavaStreamingContext jssc = JavaStreamingContext.getOrCreate(checkpointDir, fact);

// 当这段代码第一次运行时，假设检查点目录还不存在，那么 StreamingContext 会在你调用工厂函数（在 Scala 中为 createStreamingContext()，在 Java 中为 JavaStreamingContextFactory())时把目录创建出来
// 此处你需要设置检查点目录
// 在驱动器程序失败之后，如果你重启驱动器程序并再次执行代码，getOrCreate() 会重新从检查点目录中初始化出 StreamingContext，然后继续处理。
```
	- 除了用 getOrCreate() 来实现初始化代码以外，你还需要编写在驱动器程序崩溃时重启驱动器进程的代码。
		- 在大多数集群管理器中，Spark 不会在驱动器程序崩溃时自动重启驱动器进程，所以你需要使用诸如 monit 这样的工具来监视驱动器进程并进行重启
		- 最佳的实现方式往往取决于你的具体环境
		- Spark 在独立集群管理器中提供了更丰富的支持，可以在提交驱动器程序时使用 --supervise 标记来让 Spark 重启失败的驱动器程序
		- 你还要传递 --deploy-mode cluster 参数来确保驱动器程序在集群中运行，而不是在本地机器上运行
```
<!-- 使用监管模式启动驱动器程序 -->
./bin/spark-submit --deploy-mode cluster --supervise --master spark://... App.jar

<!-- 在使用这个选项时，如果你希望 Spark 独立模式集群的主节点也是容错的，就可以通过ZooKeeper 来配置主节点的容错性，详细信息请参考 Spark 的文档（https://spark.apache.org/docs/latest/spark-standalone.html#high-availability） -->
<!-- 这样配置之后，就不用再担心你的应用会出现单个节点失败的情况 -->
```
- 当驱动器程序崩溃时，Spark 的执行器进程也会重启
	- 这是 1.2 以及更早版本的 Spark 的预期的行为
		- 因为执行器程序不能在没有驱动器程序的情况下继续处理数据
		- 重启驱动器程序会启动新的执行器进程来继续之前的计算
3. 工作节点容错
- 为了应对工作节点失败的问题，Spark Streaming 使用与 Spark 的容错机制相同的方法
- 所有从外部数据源中收到的数据都在多个工作节点上备份
- 所有从备份数据转化操作的过程中创建出来的 RDD 都能容忍一个工作节点的失败，因为根据 RDD 谱系图，系统可以把丢失的数据从幸存的输入数据备份中重算出来
4. 接收器容错
- 运行接收器的工作节点的容错也是很重要的
	- 如果这样的节点发生错误，Spark Streaming会在集群中别的节点上重启失败的接收器
	- 这种情况会不会导致数据的丢失取决于数据源的行为（数据源是否会重发数据）以及接收器的实现（接收器是否会向数据源确认收到数据）
		- 在“接收器从数据池中拉取数据”的模型中，Spark 只会在数据已经在集群中备份时才会从数据池中移除元素
		- 而在“向接收器推数据”的模型中，如果接收器在数据备份之前失败，一些数据可能就会丢失
	- 总的来说，对于任意一个接收器，你必须同时考虑上游数据源的容错性（是否支持事务）来确保零数据丢失
- 接收器提供以下保证:
	- 所有从可靠文件系统中读取的数据（比如通过 StreamingContext.hadoopFiles 读取的）都是可靠的，因为底层的文件系统是有备份的。Spark Streaming 会记住哪些数据存放到了检查点中，并在应用崩溃后从检查点处继续执行
	- 对于像 Kafka、推式 Flume、Twitter 这样的不可靠数据源，Spark 会把输入数据复制到其他节点上，但是如果接收器任务崩溃，Spark 还是会丢失数据。在 Spark 1.2 中，收到的数据被记录到诸如 HDFS 这样的可靠的文件系统中，这样即使驱动器程序重启也不会导致数据丢失
	- 确保所有数据都被处理的最佳方式是使用可靠的数据源（例如 HDFS、拉式Flume 等）
5. 处理保证
- 由于 Spark Streaming 工作节点的容错保障，Spark Streaming 可以为所有的转化操作提供
“精确一次”执行的语义，即使一个工作节点在处理部分数据时发生失败，最终的转化结
果（即转化操作得到的 RDD）仍然与数据只被处理一次得到的结果一样
- 然而，当把转化操作得到的结果使用输出操作推入外部系统中时，写结果的任务可能因故
障而执行多次，一些数据可能也就被写了多次
	- 由于这引入了外部系统，因此我们需要专门针对各系统的代码来处理这样的情况
	- 我们可以使用事务操作来写入外部系统（即原子化地将一个 RDD 分区一次写入），或者设计幂等的更新操作（即多次运行同一个更新操作仍生成相同的结果）
		- 比如 Spark Streaming 的 saveAs...File 操作会在一个文件写完时自动将其原子化地移动到最终位置上，以此确保每个输出文件只存在一份

#### Streaming用户界面
Spark Streaming 提供了一个特殊的用户界面, 在常规的 Spark 用户界面（一般为 http://:4040）上的 Streaming 标签页里
- Streaming 用户界面展示了批处理和接收器的统计信息

#### 性能考量
1. 批次和窗口大小
- 最常见的问题是 Spark Streaming 可以使用的最小批次间隔是多少
	- 总的来说，500 毫秒已经被证实为对许多应用而言是比较好的最小批次大小
	- 寻找最小批次大小的最佳实践是从一个比较大的批次大小（10 秒左右）开始，不断使用更小的批次大小
		- 如果 Streaming 用户界面中显示的处理时间保持不变，你就可以进一步减小批次大小
		- 如果处理时间开始增加，你可能已经达到了应用的极限
- 对于窗口操作，计算结果的间隔（也就是滑动步长）对于性能也有巨大的影响
	- 当计算代价巨大并成为系统瓶颈时，就应该考虑提高滑动步长了
2. 并行度
- 减少批处理所消耗时间的常见方式还有提高并行度, 有以下三种方式：
	- 增加接收器数目
		- 有时如果记录太多导致单台机器来不及读入并分发的话，接收器会成为系统瓶颈
		- 这时你就需要通过创建多个输入 DStream（这样会创建多个接收器）来增加接收器数目，然后使用 union 来把数据合并为一个数据源
	- 将收到的数据显式地重新分区
		- 如果接收器数目无法再增加，你可以通过使用 DStream.repartition 来显式重新分区输入流（或者合并多个流得到的数据流）来重新分配收到的数据
	- 提高聚合计算的并行度
		- 对于像 reduceByKey() 这样的操作，你可以在第二个参数中指定并行度，我们在介绍RDD 时提到过类似的手段
3. 垃圾回收和内存使用
Java 的垃圾回收机制（简称 GC）也可能会引起问题
- 你可以通过打开 Java 的并发标志—清除收集器（Concurrent Mark-Sweep garbage collector）来减少 GC 引起的不可预测的长暂停
	- 并发标志—清除收集器总体上会消耗更多的资源，但是会减少暂停的发生
	- 可以通过在配置参数 spark.executor.extraJavaOptions 中添加 -XX:+UseConcMarkSweepGC来控制选择并发标志—清除收集器
```
<!-- 打开并发标志—清除收集器 -->
spark-submit --conf spark.executor.extraJavaOptions=-XX:+UseConcMarkSweepGC App.jar
```
- 除了使用较少引发暂停的垃圾回收器，你还可以通过减轻 GC 的压力来大幅度改善性能
	- 把 RDD 以序列化的格式缓存（而不使用原生的对象）也可以减轻 GC 的压力(默认情况下 Spark Streaming 生成的 RDD 都以序列化后的格式存储的原因)
	- 使用 Kryo 序列化工具可以进一步减少缓存在内存中的数据所需要的内存大小
- Spark 也允许我们控制缓存下来的 RDD 以怎样的策略从缓存中移除
	- 默认情况下，Spark使用 LRU 缓存
	- 如果你设置了 spark.cleaner.ttl，Spark 也会显式移除超出给定时间范围的老 RDD
	- 主动从缓存中移除不大可能再用到的 RDD，可以减轻 GC 的压力
