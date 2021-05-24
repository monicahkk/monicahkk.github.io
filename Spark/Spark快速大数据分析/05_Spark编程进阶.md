## Spark编程进阶

#### 共享变量
共享变量是一种可以在 Spark 任务中使用的特殊类型的变量
- 两种类型的共享变量：
	- 累加器（accumulator）:对信息进行聚合
	- 广播变量（broadcast variable）:来高效分发较大的对象

1. 累加器
	- 提供了将工作节点中的值聚合到驱动器程序中的简单语法
	- 一个常见用途是在调试时对作业执行过程中的事件进行计数
```python 3
# 在 Python 中累加空行
file = sc.textFile(inputFile)
# 创建Accumulator[Int]并初始化为0
blankLines = sc.accumulator(0)  # 加法操作 Spark 的一种累加器类型整型（Accumulator[Int]）

def extractCallSigns(line):
	global blankLines # 访问全局变量
	if (line == ""):
		blankLines += 1
	return line.split(" ")

callSigns = file.flatMap(extractCallSigns)
callSigns.saveAsTextFile(outputDir + "/callsigns")
print "Blank lines: %d" % blankLines.value

```
	- 用法
		- 通过在驱动器中调用 SparkContext.accumulator(initialValue) 方法，创建出存有初始值的累加器。返回值为 org.apache.spark.Accumulator[T] 对象，其中 T 是初始值initialValue 的类型
		- Spark 闭包里的执行器代码可以使用累加器的 += 方法（在 Java 中是 add）增加累加器的值
		- 驱动器程序可以调用累加器的 value 属性（在 Java 中使用 value() 或 setValue()）来访问累加器的值
	- 如有多个值需要跟踪时，或者当某个值需要在并行程序的多个地方增长时 会很方便
	- 为了防止产生含有过多错误的垃圾输出，可以使用累加器对有效记录和无效记录分别进行计数
	- 累加器的值只有在驱动器程序中可以访问，所以检查也应当在驱动器程序中完成
```python 3
# 在 Python 使用累加器进行错误计数
# 创建用来验证呼号的累加器
validSignCount = sc.accumulator(0)
invalidSignCount = sc.accumulator(0)

def validateSign(sign):
	global validSignCount, invalidSignCount
	if re.match(r"\A\d?[a-zA-Z]{1,2}\d{1,4}[a-zA-Z]{1,3}\Z", sign):
		validSignCount += 1
		return True
	else:
		invalidSignCount += 1
		return False

# 对与每个呼号的联系次数进行计数
validSigns = callSigns.filter(validateSign)
contactCount = validSigns.map(lambda sign: (sign, 1)).reduceByKey(lambda (x, y): x + y)

# 强制求值计算计数
contactCount.count()
if invalidSignCount.value < 0.1 * validSignCount.value:
	contactCount.saveAsTextFile(outputDir + "/contactCount")
else:
	print "Too many errors: %d in %d" % (invalidSignCount.value, validSignCount.value)
```
	1.1 累加器与容错性
		- 对于要在行动操作中使用的累加器，Spark只会把每个任务对各累加器的修改应用一次
			- 如果想要一个无论在失败还是重复计算时都绝对可靠的累加器，我们必须把它放在 foreach() 这样的行动操作中
		- 对于在 RDD 转化操作中使用的累加器，就不能保证有这种情况了
			- 转化操作中累加器可能会发生不止一次更新
			- 在转化操作中，累加器通常只用于调试目的
	1.2 自定义累加器
		- Spark 还直接支持 Double、Long 和 Float 型的累加器
		- 自定义累加器需要扩展 AccumulatorParam，这在 Spark API 文档（http://spark.apache.org/docs/latest/api/scala/index.html#package ）中有所介绍
2. 广播变量
	- Spark 会自动把闭包中所有引用到的变量发送到工作节点上 -> 很方便但低效
		- 首先，默认的任务发射机制是专门为小任务进行优化的；
		- 其次，事实上你可能会在多个并行操作中使用同一个变量，但是 Spark 会为每个操作分别发送
```python 3
# 在 Python 中查询国家
# 查询RDD contactCounts中的呼号的对应位置。将呼号前缀
# 读取为国家代码来进行查询
signPrefixes = loadCallSignTable()

def processSignCount(sign_count, signPrefixes):
	country = lookupCountry(sign_count[0], signPrefixes)
	count = sign_count[1]
	return (country, count)

countryContactCounts = (contactCounts.map(processSignCount).reduceByKey((lambda x, y: x+ y)))
# signPrefixes达到数 MB 大小，从主节点为每个任务发送一个这样的数组就会代价巨大
```
	- 把 signPrefixes 变为广播变量
		- 广播变量其实就是类型为 spark.broadcast.Broadcast[T] 的一个对象，其中存放着类型为 T 的值
		- 可以在任务中通过对Broadcast 对象调用 value 来获取该对象的值
		- 这个值只会被发送到各节点一次，使用的是一种高效的类似 BitTorrent 的通信机制
```python 3
# 在 Python 中使用广播变量查询国家
# 查询RDD contactCounts中的呼号的对应位置。将呼号前缀
# 读取为国家代码来进行查询
signPrefixes = sc.broadcast(loadCallSignTable())

def processSignCount(sign_count, signPrefixes):
	country = lookupCountry(sign_count[0], signPrefixes.value)
	count = sign_count[1]
	return (country, count)

countryContactCounts = (contactCounts.map(processSignCount).reduceByKey((lambda x, y: x+ y)))

countryContactCounts.saveAsTextFile(outputDir + "/countries.txt")
```
	- 使用广播变量的过程很简单。
		- 通过对一个类型 T 的对象调用 SparkContext.broadcast 创建出一个 Broadcast[T] 对象。任何可序列化的类型都可以这么实现。
		- 通过 value 属性访问该对象的值（在 Java 中为 value() 方法）。
		- 变量只会被发到各个节点一次，应作为只读值处理（修改这个值不会影响到别的节点）
	- 满足只读要求的最容易的使用方式是广播基本类型的值或者引用不可变
	- 有时传一个可变对象可能更为方便与高效, 就需要自己维护只读的条件
		- 必须确保从节点上运行的代码不会尝试去做诸如 val theArray = broadcastArray.value; theArray(0) = newValue 这样的事情
		- 当在工作节点上执行时，这一行将 newValue 赋给数组的第一个元素，但是只对该工作节点本地的这个数组的副本有效，而不会改变任何其他工作节点上通过 broadcastArray.value 所读取到的内容
	2.1 广播的优化
		- 如果序列化对象的时间很长或者传送花费的时间太久，这段时间很容易就成为性能瓶颈
		- 可以使用 spark.serializer 属性选择另一个序列化库来优化序列化过程（第 8 章中会讨论如何使用 Kryo 这种更快的序列化库）
		- 也可以为你的数据类型实现自己的序列化方式（对 Java 对象使用 java.io.Externalizable 接口实现序列化 或 使用 reduce() 方法为 Python 的 pickle 库定义自定义的序列化）
3. 基于分区进行操作
	- Spark 提供基于分区的 map 和 foreach，让你的部分代码只对 RDD 的每个分区运行一次，这样可以帮助降低这些操作的代价
```python 3
# 在 Python 中使用共享连接池
def processCallSigns(signs):
	"""使用连接池查询呼号"""
	# 创建一个连接池
	http = urllib3.PoolManager()
	# 与每条呼号记录相关联的URL
	urls = map(lambda x: "http://73s.com/qsos/%s.json" % x, signs)
	# 创建请求（非阻塞）
	requests = map(lambda x: (x, http.request('GET', x)), urls)
	# 获取结果
	result = map(lambda x: (x[0], json.loads(x[1].data)), requests)
	# 删除空的结果并返回
	return filter(lambda x: x[1] is not None, result)

def fetchCallSigns(input):
	"""获取呼号"""
	return input.mapPartitions(lambda callSigns : processCallSigns(callSigns))

contactsContactList = fetchCallSigns(validSigns)
```

| 函数名 | 调用所提供的 | 返回的 | 对于RDD[T]的函数签名 |
|:--:|:--:|:--:|
|mapPartitions()|该分区中元素的迭代器|返回的元素的迭代器 | f: (Iterator[T]) → Iterator[U]|
| mapPartitionsWithIndex() | 分区序号，以及每个分区中的元素的迭代器 | 返回的元素的迭代器|f: (Int, Iterator[T]) → Iterator[U]|
|foreachPartitions()|元素迭代器|无|f: (Iterator[T]) → Unit |

```python 3
# 在 Python 中不使用 mapPartitions() 求平均值
def combineCtrs(c1, c2):
     return (c1[0] + c2[0], c1[1] + c2[1])
     
def basicAvg(nums):
     """计算平均值"""
     nums.map(lambda num: (num, 1)).reduce(combineCtrs)
     
# 例 6-14：在 Python 中使用 mapPartitions() 求平均值
def partitionCtr(nums):
     """计算分区的sumCounter"""
     sumCount = [0, 0]
     for num in nums:
         sumCount[0] += num
         sumCount[1] += 1
     return [sumCount]
     
def fastAvg(nums):
     """计算平均值"""
     sumCount = nums.mapPartitions(partitionCtr).reduce(combineCtrs)
     return sumCount[0] / float(sumCount[1])
```
4. 与外部程序间的管道
	- Spark 在 RDD 上提供 pipe() 方法。Spark 的 pipe() 方法可以让我们使用任意一种语言实现 Spark 作业中的部分逻辑，只要它能读写 Unix 标准流就行
	- 通过 pipe()，你可以将 RDD 中的各元素从标准输入流中以字符串形式读出，并对这些元素执行任何你需要的操作，然后把结果以字符串的形式写入标准输出——这个过程就是 RDD 的转化操作过程
```R
# R 语言的距离程序
#!/usr/bin/env Rscript
library("Imap")
f <- file("stdin")
open(f)
while(length(line <- readLines(f,n=1)) > 0) {
    # 处理行
    contents <- Map(as.numeric, strsplit(line, ","))
    mydist <- gdist(contents[[1]][1], contents[[1]][2],
                    contents[[1]][3], contents[[1]][4],
                    units="m", a=6378137.0, b=6356752.3142, verbose = FALSE)
    write(mydist, stdout())
}
```
```python 3
# 例 6-16：在 Python 中使用 pipe() 调用 finddistance.R 的驱动器程序
# 使用一个R语言外部程序计算每次呼叫的距离
distScript = "./src/R/finddistance.R"
distScriptName = "finddistance.R"
sc.addFile(distScript)
def hasDistInfo(call):
    """验证一次呼叫是否有计算距离时必需的字段"""
    requiredFields = ["mylat", "mylong", "contactlat", "contactlong"]
    return all(map(lambda f: call[f], requiredFields))
def formatCall(call):
    """将呼叫按新的格式重新组织以使之可以被R程序解析"""
    return "{0},{1},{2},{3}".format(
        call["mylat"], call["mylong"],
        call["contactlat"], call["contactlong"])
        
pipeInputs = contactsContactList.values().flatMap(
     lambda calls: map(formatCall, filter(hasDistInfo, calls)))
distances = pipeInputs.pipe(SparkFiles.get(distScriptName))
print distances.collect()
```
	- 通过 SparkContext.addFile(path)，可以构建一个文件列表，让每个工作节点在 Spark 作业中下载列表中的文件，当作业中的行动操作被触发时，这些文件就会被各节点下载，然后我们就可以在工作节点上通过 SparkFiles.getRootDirectory 找到它们；也可以使用 SparkFiles.get(Filename)来定位单个文件
	- 所有通过 SparkContext.addFile(path) 添加的文件都存储在同一个目录中，所以有必要使用唯一的名字
	- 一旦脚本可以访问，RDD 的 pipe() 方法就可以让 RDD 中的元素很容易地通过脚本管道，假设有一个更好版本的 findDistance，可以以命令行参数的形式接收指定的 SEPARATOR
```SARSQL
rdd.pipe(Seq(SparkFiles.get("finddistance.R"), ",")) # 命令调用以可定位的参数序列的形式传递（命令本身在零偏移位置）
rdd.pipe(SparkFiles.get("finddistance.R") + " ,") # 将它作为一个命令字符串传递，然后 Spark 会将这个字符串拆解为可定位的参数序列
```
	- 可以通过 pipe() 指定命令行环境变量。只需要把环境变量到对应值的映射表作为 pipe() 的第二个参数传进去，S
	- park 就会设置好这些值
5. 数值RDD的操作
	- Spark 的数值操作是通过流式算法实现的，允许以每次一个元素的方式构建出模型。这些统计数据都会在调用 stats() 时通过一次遍历数据计算出来，并以 StatsCounter 对象返回
|方法|含义|
|:--:|:--:|
|count()|RDD 中的元素个数|
|mean()|元素的平均值|
|sum()|总和|
|max()|最大值|
|min()|最小值|
|variance()|元素的方差|
|sampleVariance()|从采样中计算出的方差|
|stdev()|标准差|
|sampleStdev()|采样的标准差|
```sql
# 用 Python 移除异常值
# 要把String类型RDD转为数字数据，这样才能
# 使用统计函数并移除异常值
distanceNumerics = distances.map(lambda string: float(string))
stats = distanceNumerics.stats()
stddev = std.stdev()
mean = stats.mean()
reasonableDistances = distanceNumerics.filter(
     lambda x: math.fabs(x - mean) < 3 * stddev)
print reasonableDistances.collect()
```
完整的源代码：
src/python/ChapterSixExample.py、src/main/scala/com/oreilly/learningsparkexamples/scala/ChapterSixExample.scala 
以及 src/main/java/com/oreilly/learningsparkexamples/java/ChapterSixExample.java