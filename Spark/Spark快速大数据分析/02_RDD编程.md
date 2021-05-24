## 弹性分布式数据集RDD

Resilient Distributed Dataset 

https://github.com/databricks/learning-spark

RDD 就是一个不可变的分布式对象集合

两种方法创建 RDD:

- 读取一个外部数据集
- 在驱动器程序里分发驱动器程序中的对象集合

```SPARQL
# 在 Python 中使用 textFile() 创建一个字符串的 RDD
lines = sc.textFile("README.md")
```

创建后,支持两种操作: 

- 转化操作(transformation) : 由一个 RDD生成一个新的RDD

```SPARQL
# 调用转化操作 filter()
pythonLines = lines.filter(lambda line: "Python" in line)
```

- 行动操作(action): 对 RDD 计算出一个结果，并把结果返回到驱动器程序中，或把结果存储到外部存储系统（如 HDFS）中

```SPARQL
# 调用 first() 行动操作
pythonLines.first()
u'## Interactive Python Shell'
```

- 区别: Spark计算RDD的方式不同; 惰性计算; 
  - 只有第一次在一个行动操作中用到时,才会真正计算; 会在每次对它们进行行动操作时重新计算
  - 让多个行动操作重用同一个RDD,可以使用 **RDD.persist()** 让Spark把RDD缓存下来
    - 可以把数据持久化到许多不同地方
    - 第一次对持久化的RDD计算后,Spark会把RDD内容保存到内存中(以分区的方式存储到集群中的各机器上)
    - 也可缓存到磁盘上

```SPARQL
# 把 RDD 持久化到内存中
>>> pythonLines.persist
>>> pythonLines.count()
2
>>> pythonLines.first()
u'## Interactive Python Shell'
```

*****

每个 Spark 程序或 shell 会话都按如下方式工作:
- 从外部数据创建出输入 RDD
- 使用诸如 filter() 这样的转化操作对 RDD 进行转化，以定义新的 RDD
- 告诉 Spark 对需要被重用的中间结果 RDD 执行 persist() 操作
- 使用行动操作（例如 count() 和 first() 等）来触发一次并行计算，Spark 会对计算进行优化后再执行

*****

#### 创建RDD
- 两种创建方式：读取外部数据集 / 在驱动器程序中对一个集合进行并行化
- 最简单的方式就是把程序中一个已有的集合传给 **SparkContext** 的 **parallelize()** 方法
```SPARQL
# Python 中的 parallelize() 方法
lines = sc.parallelize(["pandas", "i like pandas"])
# Scala 中的 parallelize() 方法
val lines = sc.parallelize(List("pandas", "i like pandas"))
# Java 中的 parallelize() 方法
JavaRDD<String> lines = sc.parallelize(Arrays.asList("pandas", "i like pandas"));
```
- 更常用的方式是从外部存储中读取数据来创建 RDD
```SPARQL
# Python 中的 textFile() 方法
lines = sc.textFile("/path/to/README.md")
# Scala 中的 textFile() 方法
val lines = sc.textFile("/path/to/README.md")
# Java 中的 textFile() 方法
JavaRDD<String> lines = sc.textFile("/path/to/README.md");
```

#### RDD操作
- 转化操作
  - 转化操作转化出来的 RDD 是惰性求值的，只有在行动操作中用到这些 RDD 时才会被计算
```SPARQL
# 用 Python 实现 filter() 转化操作
inputRDD = sc.textFile("log.txt")
errorsRDD = inputRDD.filter(lambda x: "error" in x)
# 用 Scala 实现 filter() 转化操作
val inputRDD = sc.textFile("log.txt")
val errorsRDD = inputRDD.filter(line => line.contains("error"))
# 用 Java 实现 filter() 转化操作
JavaRDD<String> inputRDD = sc.textFile("log.txt");
JavaRDD<String> errorsRDD = inputRDD.filter(
     new Function<String, Boolean>() {
         public Boolean call(String x) { return x.contains("error"); }
     }
});

# filter() 操作不会改变已有的 inputRDD 中的数据
```
```SPARQL
# 使用 union() 来打印出包含 error 或 warning 的行数; union() 与 filter() 的不同点在于它操作两个 RDD 而不是一个
#　用 Python 进行 union() 转化操作
errorsRDD = inputRDD.filter(lambda x: "error" in x)
warningsRDD = inputRDD.filter(lambda x: "warning" in x)
badLinesRDD = errorsRDD.union(warningsRDD)

# 更好的方法是直接筛选出要么包含 error 要么包含 warning 的行，这样只对 inputRDD 进行一次筛选即可
```
  - 派生出新的RDD后,Spark会使用谱系图(lineage graph)记录这些RDD之间的依赖关系
    - 用关系信息按需计算每个RDD
    - 依靠谱系图在持久化的RDD丢失部分数据时恢复所丢失的数据
      	![日志分析过程中创建出的RDD谱系图](D:\MyDocuments\Typora\spark\Spark快速大数据分析\分析日志创建的谱系图.bmp)
  
- 行动操作
  - 行动操作需要生成实际的输出，它们会强制执行那些求值必须用到的 RDD 的转化操作
  - collect() 函数，可以用来获取整个 RDD 中的数据 ---- 只有当你的整个数据集能在单台机器的内存中放得下时，才能使用 ---- collect() 不能用在大规模数据集上
  - 在大多数情况下，RDD 不能通过 collect() 收集到驱动器进程中(通常很大), 通常要把数据写到诸如 HDFS 或 Amazon S3 这样的分布式的存储系统中, 可以使用 saveAsTextFile()、saveAsSequenceFile()，或者任意的其他行动操作来把 RDD 的数据内容以各种自带的格式保存起来
  - 每当调用一个新的行动操作时，整个 RDD 都会从头开始计算。要避免这种低效的行为，可以将中间结果持久化
  
- 惰性求值: 对 RDD 调用转化操作时，操作不会立即执行，Spark 会在内部记录下所要求执行的操作的相关信息

  - RDD: 通过转化操作构建出来的、记录如何计算数据的指令列表
  - 把数据读取到 RDD 的操作也同样是惰性的
  - 虽然转化操作是惰性求值的，但还是可以随时通过运行一个行动操作来强制Spark 执行 RDD 的转化操作，比如使用 count()。这是 **一种对你所写的程序进行部分测试的简单方法**

- 向Spark传递函数

  - python —— 3种：

    - 较短的函数：lambda

```SPARQL
word = rdd.filter(lambda s: "error" in s)
```

    - 传递顶层函数

      - python会将函数所在的对象也序列化传出去

```SPARQL
def containsError(s):
 return "error" in s
word = rdd.filter(containsError)
```

      - 传递的对象是某个对象的成员 / 包含了某个对象中一个字段的引用（self.field），spark会把整个对象发到工作节点上

      - 如果传递的类里包含python不知如何序列化传输的对象，也会导致程序失败

```SPARQL
# 传递一个带字段引用的函数（别这么做！）
class SearchFunctions(object):
     def __init__(self, query):
         self.query = query
     def isMatch(self, s):
         return self.query in s
     def getMatchesFunctionReference(self, rdd):
         # 问题：在"self.isMatch"中引用了整个self
         return rdd.filter(self.isMatch)
     def getMatchesMemberReference(self, rdd):
         # 问题：在"self.query"中引用了整个self
         return rdd.filter(lambda x: self.query in x)
         
#替代方案　－　传递不带字段引用的 Python 函数
class WordFunctions(object):
     ...
     def getMatchesNoReference(self, rdd):
         # 安全：只把需要的字段提取到局部变量中
         query = self.query
         return rdd.filter(lambda x: query in x)
```

    - 定义的局部函数

- Ｓｃａｌａ

  - 把定义的内联函数、方法的引用或静态方法传递给 Spark

  - 传递的函数与其引用的数据需要可序列化

  - 传递一个对象的方法或者字段时，会包含对整个对象的引用; 可以把需要的字段放到一个局部变量中，来避免传递包含该字段的整个对象

```python 3
# cala 中的函数传递
class SearchFunctions(val query: String) {
   def isMatch(s: String): Boolean = {
       s.contains(query)
}
   def getMatchesFunctionReference(rdd: RDD[String]): RDD[String] = {
    // 问题："isMatch"表示"this.isMatch"，因此我们要传递整个"this"
    rdd.map(isMatch)
}
    def getMatchesFieldReference(rdd: RDD[String]): RDD[String] = {
    // 问题："query"表示"this.query"，因此我们要传递整个"this"
    rdd.map(x => x.split(query))
}
    def getMatchesNoReference(rdd: RDD[String]): RDD[String] = {
    // 安全：只把我们需要的字段拿出来放入局部变量中
val query_ = this.query
rdd.map(x => x.split(query_))
}
```

  - Scala 中出现了 NotSerializableException，通常问题就在于我们传递了一个不可序列化的类中的函数或字段

    - 传递局部可序列化变量或顶级对象中的函数始终是安全的

- Java

  - 基本的一些函数接口

    ​	![标准接口](D:\MyDocuments\Typora\spark\Spark快速大数据分析\标准java函数接口.PNG)

  - 可以把函数类内联定义为使用匿名内部类，也可以创建一个具名类

```java
#　在 Java 中使用匿名内部类进行函数传递
RDD<String> errors = lines.filter(new Function<String, Boolean>() {
      public Boolean call(String x) { return x.contains("error"); }
});

#　在 Java 中使用具名类进行函数传递
class ContainsError implements Function<String, Boolean>() {
      public Boolean call(String x) { return x.contains("error"); }
}
RDD<String> errors = lines.filter(new ContainsError()
```

  - 顶级具名类通常在组织大型程序时显得比较清晰，并且可以给它们的构造函数添加参数

```java
#带参数的 Java 函数类
class Contains implements Function<String, Boolean>() { 
     private String query; 
     public Contains(String query) { this.query = query; } 
     public Boolean call(String x) { return x.contains(query); } 
} 
RDD<String> errors = lines.filter(new Contains("error"));

# 在 Java 中使用 Java 8 地 lambda 表达式进行函数传递
RDD<String> errors = lines.filter(s -> s.contains("error"));
# [Oracle 的相关文档](http://docs.oracle.com/javase/tutorial/java/javaOO/lambdaexpressions.html)
# [Databricks 关于如何在 Spark 中使用 lambda 表达式的博客](http://databricks.com/blog/2014/04/14/spark-with-java-8.html)
```

  - 匿名内部类和 lambda 表达式都可以引用方法中封装的任意 final 变量，因此你可以像在 Python 和 Scala 中一样把这些变量传递给 Spark

- 常见的转化操作和行动操作

  1. 基本RDD

     1. 针对各个元素的转化操作

        - 两个最常用的转化操作是 map() 和 filter()

          - map() 接收一个函数, 把这个函数用于 RDD 中的每个元素, 将函数的返回结果作为结果RDD 中对应元素的值, 返回值类型不需要和输入类型一样

          - filter() 则接收一个函数，并将 RDD 中满足该函数的元素放入新的 RDD 中返回

            ![map & filter](D:\MyDocuments\Typora\spark\Spark快速大数据分析\map & filter.PNG)

```Python
# Python 版计算 RDD 中各值的平方
nums = sc.parallelize([1, 2, 3, 4])
squared = nums.map(lambda x: x * x).collect()
for num in squared:
     print "%i " % (num)
 
# Scala 版计算 RDD 中各值的平方
val input = sc.parallelize(List(1, 2, 3, 4))
val result = input.map(x => x * x)
println(result.collect().mkString(","))

# Java 版计算 RDD 中各值的平方
JavaRDD<Integer> rdd = sc.parallelize(Arrays.asList(1, 2, 3, 4));
JavaRDD<Integer> result = rdd.map(new Function<Integer, Integer>() {
     public Integer call(Integer x) { return x*x; }
});
System.out.println(StringUtils.join(result.collect(), ","))
```

          - flatMap(): 对每个输入元素生成多个输出元素, 返回的是一个返回值序列的迭代器, 得到一个包含各个迭代器可访问的所有元素的 RDD

```Python
# 一个简单用途是把输入的字符串切分为单词
# Python 中的 flatMap() 将行数据切分为单词
lines = sc.parallelize(["hello world", "hi"]) 
words = lines.flatMap(lambda line: line.split(" ")) 
words.first() # 返回"hello"

# Scala 中的 flatMap() 将行数据切分为单词
val lines = sc.parallelize(List("hello world", "hi")) 
val words = lines.flatMap(line => line.split(" ")) 
words.first() // 返回"hello"

# Java 中的 flatMap() 将行数据切分为单词
JavaRDD<String> lines = sc.parallelize(Arrays.asList("hello world", "hi")); 
JavaRDD<String> words = lines.flatMap(new FlatMapFunction<String, String>() { 
     public Iterable<String> call(String line) { 
         return Arrays.asList(line.split(" ")); 
     } 
}); 
words.first(); // 返回"hello"
```

            ![flatmap vs map](D:\MyDocuments\Typora\spark\Spark快速大数据分析\flatmap vs map.PNG)

     2. 伪集合操作

        - RDD 中最常缺失的集合属性是元素的唯一性

          - RDD.distinct(): 生成一个只包含不同元素的新RDD - distinct() 操作的开销很大，它需要将所有数据通过__网络进行混洗（shuffle）__，以确保每个元素都只有一份

            ![简单的聚合操作](D:\MyDocuments\Typora\spark\Spark快速大数据分析\简单的聚合操作.PNG)

          - union(other): 返回一个包含两个 RDD 中所有元素的 RDD

            - 如果输入的 RDD 中有重复数据，Spark 的 union() 操作也会__包含这些重复数据__

          - intersection(other): 只返回两个 RDD 中都有的元素

            - 在运行时也会去掉所有重复的元素
            - intersection() 的性能却要差很多, 它需要通过__网络混洗数据__来发现共有的元素

          - subtract(other): 数接收另一个 RDD 作为参数，返回一个由只存在于第一个 RDD 中而不存在于第二个 RDD 中的所有元素组成的 RDD

            - 也需要__数据混洗__

          - cartesian(other): 计算两个 RDD 的笛卡儿积

            - 大规模 RDD 的笛卡儿积开销巨大

        ![常见RDD操作](D:\MyDocuments\Typora\spark\Spark快速大数据分析\常见RDD操作.PNG)

     3. 行动操作

        - reduce(): 接收一个函数作为参数，这个函数要操作两个 RDD 的元素类型的数据并返回一个同样类型的新元素

```Python
# Python 中的 reduce()
sum = rdd.reduce(lambda x, y: x + y)

# Scala 中的 reduce()
val sum = rdd.reduce((x, y) => x + y)

# Java 中的 reduce()
Integer sum = rdd.reduce(new Function2<Integer, Integer, Integer>() { 
     public Integer call(Integer x, Integer y) { return x + y; } 
});
```

        - fold(): 接收一个与 reduce() 接收的函数签名相同的函数，再加上一个“初始值”来作为每个分区第一次调用时的结果;

          - 初始值应当是你提供的操作的单位元素, 使用你的函数对这个初始值进行多次计算不会改变结果
          - 可以通过原地修改并返回两个参数中的前一个的值来节约在 fold() 中创建对象的开销。但是你没有办法修改第二个参数
          - fold() 和 reduce() 都要求函数的返回值类型需要和我们所操作的 RDD 中的元素类型相同

        - aggregate(): 提供我们期待返回的类型的初始值。然后通过一个函数把 RDD 中的元素合并起来放入累加器, 需要提供第二个函数来将累加器两两合并(到每个节点是在本地进行累加的)

          - 以用 aggregate() 来计算 RDD 的平均值，来代替 map() 后面接 fold() 的方式

```Python
# Python 中的 aggregate()
sumCount = nums.aggregate((0, 0),
         (lambda acc, value: (acc[0] + value, acc[1] + 1),
         (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]))))
return sumCount[0] / float(sumCount[1])

# Scala 中的 aggregate()
val result = input.aggregate((0, 0))(
         (acc, value) => (acc._1 + value, acc._2 + 1),
         (acc1, acc2) => (acc1._1 + acc2._1, acc1._2 + acc2._2))
val avg = result._1 / result._2.toDouble
```
```java
# Java 中的 aggregate()
class AvgCount implements Serializable {
     public AvgCount(int total, int num) {
         this.total = total;
         this.num = num;
     }
     public int total;
     public int num;
	public double avg() {
		return total / (double) num;
      }
}
Function2<AvgCount, Integer, AvgCount> addAndCount =
      new Function2<AvgCount, Integer, AvgCount>() {
          public AvgCount call(AvgCount a, Integer x) {
          a.total += x;
          a.num += 1;
          return a;
      }
};
Function2<AvgCount, AvgCount, AvgCount> combine =
      new Function2<AvgCount, AvgCount, AvgCount>() {
      public AvgCount call(AvgCount a, AvgCount b) {
          a.total += b.total;
          a.num += b.num;
          return a;
      }
};
AvgCount initial = new AvgCount(0, 0);
AvgCount result = rdd.aggregate(initial, addAndCount, combine);
System.out.println(result.avg());
```

        - RDD 的一些行动操作会以普通集合或者值的形式将 RDD 的部分或全部数据返回驱动器程序中

          - 把数据返回驱动器程序中最简单、最常见的操作是 collect()，它会将整个 RDD 的内容返回, collect() 通常在单元测试中使用, 整个内容不会很大，可以放在内存中

          - take(n) 返回 RDD 中的 n 个元素，并且尝试只访问尽量少的分区，因此该操作会得到一个不均衡的集合

          - 以上对于单元测试和快速调试都很有用

          - top() : 从 RDD 中获取前几个元素, 会使用数据的默认顺序

          - takeSample(withReplacement, num, seed): 从数据中获取一个采样，并指定是否替换

          - foreach() : 对 RDD 中的每个元素进行操作，而不需要把 RDD 发回本地

            ![基本行动操作](D:\MyDocuments\Typora\spark\Spark快速大数据分析\基本行动操作.PNG)

        - 在不同RDD类型间转换

          - 有些函数只能用于特定类型的 RDD，比如 mean() 和 variance() 只能用在数值 RDD 上，而 join() 只能用在键值对 RDD 上
          - 在 Scala 中，将 RDD 转为有特定函数的 RDD（比如在 RDD[Double] 上进行数值操作）是由隐式转换来自动处理的, 加上 import org.apache.spark.SparkContext._ 来使用这些隐式转换, __可以隐式地将一个 RDD 转为各种封装类__
          - 在 Java 中，各种 RDD 的特殊类型间的转换更为明确。Java 中有两个专门的类 JavaDoubleRDD和 JavaPairRDD，来处理特殊类型的 RDD，这两个类还针对这些类型提供了额外的函数
          - 在 Python 中，所有的函数都实现在基本的RDD 类中，但如果操作对应的 RDD 数据类型不正确，就会导致运行时错误

     4. 持久化(缓存)

        - 不同的持久化级别: 默认值就是以序列化后的对象存储在 JVM 堆空间中

          - Scala和Java默认情况下persist()会把数据以序列化的形式缓存在JVM的堆空间中

          - python 始终序列化要持久化存储的数据

            ![持久化级别](D:\MyDocuments\Typora\spark\Spark快速大数据分析\持久化级别.PNG)

          - 如果要缓存的数据太多，内存中放不下，Spark 会自动利用最近最少使用（LRU）的缓存策略把最老的分区从内存中移除

          - unpersist()，调用该方法可以手动把持久化的 RDD 从缓存中移除

