### Spark SQL
Spark SQL 提供了以下三大功能：
1. Spark SQL 可以从各种结构化数据源（例如 JSON、Hive、Parquet 等）中读取数据
2. Spark SQL 不仅支持在 Spark 程序内使用 SQL 语句进行数据查询，也支持从类似商业智能软件 Tableau 这样的外部工具中通过标准数据库连接器（JDBC/ODBC）连接 Spark SQL 进行查询
3. 当在 Spark 程序内使用 Spark SQL 时，Spark SQL 支持 SQL 与常规的 Python/Java/Scala代码高度整合，包括连接 RDD 与 SQL 表、公开的自定义 SQL 函数接口等
Spark SQL 提供了一种特殊的 RDD，叫作 SchemaRDD ： 
	- 存放 Row 对象的 RDD，每个 Row 对象代表一行记录

<!-- 现成的推文： 使用 Databricks 参考应用（http://databricks.gitbooks.io/databricks-spark-reference-applications/content/twitter_classifier/README.html ）
可以直接使用本书 Git 仓库中的 files/testweet.json 文件 -->

#### 连接Spark SQL
要在应用中引入 Spark SQL 需要添加一些额外的依赖
- Spark SQL 编译时可以包含 Hive 支持，也可以不包含
	- 包含 Hive 支持的 Spark SQL 可以支持 Hive 表访问、UDF（用户自定义函数）、SerDe（序列化格式和反序列化格式），以及 Hive 查询语言（HiveQL/HQL）
	- 如果要在 Spark SQL 中包含 Hive 的库，并不需要事先安装 Hive
	- 下载的是二进制版本的 Spark，它应该已经在编译时添加了 Hive 支持
	- 如果你是从代码编译Spark，应该使用 build/sbt -Phive assembly 编译，以打开 Hive 支持
	- 如果你的应用与 Hive 之间发生了依赖冲突，并且无法通过依赖排除以及依赖封装解决问题，也可以使用没有 Hive 支持的 Spark SQL 进行编译和连接 —— 要连接的就是另一个 Maven 工件了
```java
// 带有 Hive 支持的 Spark SQL 的 Maven 索引
groupId = org.apache.spark
artifactId = spark-hive_2.10
version = 1.2.0

// 在 Python 中不需要对构建方式进行任何修改
```
	- 如果你不能引入 Hive 依赖，那就应该使用工件 spark-sql_2.10 来代替 spark-hive_2.10
- 当使用 Spark SQL 进行编程时，根据是否使用 Hive 支持，有两个不同的入口
	- 推荐使用的入口是 HiveContext，它可以提供 HiveQL 以及其他依赖于 Hive 的功能的支持
	- 更为基础的 SQLContext 则支持 Spark SQL 功能的一个子集，子集中去掉了需要依赖于 Hive 的功能
	- 这种分离主要是为那些可能会因为引入 Hive 的全部依赖而陷入依赖冲突的用户而设计的。使用 HiveContext 不需要事先部署好 Hive。
- 推荐使用 HiveQL 作为 Spark SQL 的查询语言
	- 关于 HiveQL 已经有许多资料面世
		- Programming Hive（http://shop.oreilly.com/product/0636920023555.do ）
		- 在线的Hive语言手册（https://cwiki.apache.org/confluence/display/Hive/LanguageManual ）
- 若要把 Spark SQL 连接到一个部署好的 Hive 上，你必须把 hive-site.xml 复制到Spark 的配置文件目录中（$SPARK_HOME/conf）。即使没有部署好 Hive，Spark SQL 也可以运行
	- 如果你没有部署好 Hive，Spark SQL 会在当前的工作目录中创建出自己的Hive 元数据仓库，叫作 metastore_db
	- 如果你尝试使用 HiveQL 中的 CREATE TABLE（并非 CREATE EXTERNAL TABLE）语句来创建表，这些表会被放在你默认的文件系统中的/user/hive/warehouse 目录中（如果你的 classpath 中有配好的 hdfs-site.xml，默认的文件系统就是 HDFS，否则就是本地文件系统）

#### 在应用中使用Spark SQL
- Spark SQL 最强大之处就是可以在 Spark 应用内使用。
	- 需要基于已有的 SparkContext 创建出一个 HiveContext（如果使用的是去除了 Hive 支持的 Spark 版本，则创建出 SQLContext）
	- 这个上下文环境提供了对Spark SQL 的数据进行查询和交互的额外函数
	- 使用 HiveContext 可以创建出表示结构化数据的 SchemaRDD，并且使用 SQL 或是类似 map() 的普通 RDD 操作来操作这些 SchemaRDD
1. 初始化Spark SQL
首先要在程序中添加一些 import 声明
```java
// Scala 中 SQL 的 import 声明
// 导入Spark SQL
import org.apache.spark.sql.hive.HiveContext
// 如果不能使用hive依赖的话
import org.apache.spark.sql.SQLContext

// 没有用类似在导入 SparkContext 时的方法那样导入HiveContext._ 来访问隐式转换
// 隐式转换被用来把带有类型信息的 RDD 转变为专门用于Spark SQL 查询的 RDD（也就是 SchemaRDD）

// Scala 中 SQL 需要导入的隐式转换支持
// 创建Spark SQL的HiveContext
val hiveCtx = ...
// 导入隐式转换支持
import hiveCtx._
```
```python 3
# Python 中 SQL 的 import 声明
# 导入Spark SQL
from pyspark.sql import HiveContext, Row
# 当不能引入hive依赖时
from pyspark.sql import SQLContext, Row

# 在 Python 中创建 SQL 上下文环境
# HiveContext 对象 & SQLContext 对象 都需要传入一个 SparkContext 对象作为运行的基础。
hiveCtx = HiveContext(sc)
```
可以准备读取数据并进行查询了
2. 基本查询示例
要在一张数据表上进行查询，需要调用 HiveContext 或 SQLContext 中的 sql() 方法
首先告诉 Spark SQL 要查询的数据是什么
	- 需要先从 JSON 文件中读取一些推特数据，把这些数据注册为一张临时表并赋予该表一个名字
	- 然后就可以用 SQL 来查询它
```python 3
# 在 Python 中读取并查询推文
input = hiveCtx.jsonFile(inputFile)
# 注册输入的SchemaRDD
input.registerTempTable("tweets")
# 依据retweetCount（转发计数）选出推文
topTweets = hiveCtx.sql("""SELECT text, retweetCount FROM
 tweets ORDER BY retweetCount LIMIT 10""")

# 如果你已经有安装好的 Hive，并且已经把你的 hive-site.xml 文件复制到了$SPARK_HOME/conf 目录下，那么你也可以直接运行 hiveCtx.sql 来查询已有的 Hive 表
```
3. SchemaRDD
读取数据和执行查询都会返回 SchemaRDD
从内部机理来看，SchemaRDD 是一个由 Row 对象组成的 RDD，附带包含每列数据类型的结构信息
在今后的 Spark 版本中（1.3 及以后），SchemaRDD 这个名字可能会被改为 DataFrame
- SchemaRDD 仍然是 RDD，所以你可以对其应用已有的 RDD 转化操作
- SchemaRDD 也提供了一些额外的功能支持
	- 最重要的是，你可以把任意 SchemaRDD 注册为临时表，这样就可以使用 HiveContext.sql 或 SQLContext.sql 来对它进行查询了
	- 可以通过 SchemaRDD 的 registerTempTable() 方法这么做
	- 临时表是当前使用的 HiveContext 或 SQLContext 中的临时变量，在你的应用退出时这些临时表就不再存在了
- SchemaRDD 可以存储一些基本数据类型，也可以存储由这些类型组成的结构体和数组。
	- SchemaRDD 使用 HiveQL 语法（https://cwiki.apache.org/confluence/display/Hive/LanguageManual+ DDL）定义的类型
	- 编译时除通过 -Phive 打开 Hive 支持外，还需打开 -Phive-thriftserver 选项

|Spark SQL/HiveQL类型 |Scala类型 |Java类型 |Python|
|:--:|:--:|:--:|:--:|
|TINYINT |Byte |Byte/byte |int/long ( 在 -128 到 127 之间 )|
|SMALLINT |Short |Short/short |int/long ( 在 -32768 到 32767之间 )|
|INT |Int |Int/int |int 或 long|
|BIGINT |Long |Long/long |long|
|FLOAT |Float |Float /float |float|
|DOUBLE |Double |Double/double |float|
|DECIMAL |Scala.math.BigDecimal |java.math.BigDecimal |decimal.Decimal|
|STRING |String |String |string|
|BINARY |Array[Byte] |byte[] |bytearray|
|BOOLEAN |Boolean |Boolean/boolean |bool|
|TIMESTAMP |java.sql.TimeStamp |java.sql.TimeStamp |datetime.datetime|
|ARRAY<DATA_TYPE> |Seq |List |list、tuple 或 array|
|MAP<KEY_TYPE, VAL_TYPE> |Map |Map |dict|
|STRUCT<COL1:
COL1_TYPE, ...> |Row |Row |Row|

- 使用Row对象
	- Row 对象表示 SchemaRDD 中的记录，其本质就是一个定长的字段数组
	- 在 Python 中，由于没有显式的类型系统，Row 对象变得稍有不同
		- 我们使用 row[i] 来访问第 i 个元素
		- 除此之外，Python 中的 Row 还支持以 row.column_name 的形式使用名字来访问其中的字段
```python 3
# 在 Python 中访问 topTweet 这个 SchemaRDD 中的 text 列
topTweetText = topTweets.map(lambda row: row.text)
```
4. 缓存
Spark SQL 的缓存机制与 Spark 中的稍有不同, Spark 可以更加高效地存储数据（知道每个列的类型信息）
- 为了确保使用更节约内存的表示方式进行缓存而不是储存整个对象，应当使用专门的 hiveCtx.cacheTable("tableName") 方法
	- 当缓存数据表时，Spark SQL 使用一种列式存储格式在内存中表示数据
	- 这些缓存下来的表只会在驱动器程序的生命周期里保留在内存中，所以如果驱动器进程退出，就需要重新缓存数据
_在 Spark 1.2 中，RDD 上原有的 cache() 方法也会引发一次对 cacheTable()方法的调用_
- 也可以使用 HiveQL/SQL 语句来缓存表
	- 只需要运行 CACHE TABLEtableName 或 UNCACHE TABLEtableName 来缓存表或者删除已有的缓存即可
	- 这种使用方式在 JDBC 服务器的命令行客户端中很常用

#### 读取和存储数据
- Spark SQL 支持很多种结构化数据源，可以让你跳过复杂的读取过程，轻松从各种数据源中读取到 Row 对象。这些数据源包括 Hive 表、JSON 和 Parquet 文件
- 当你使用SQL 查询这些数据源中的数据并且只用到了一部分字段时，Spark SQL 可以智能地只扫描这些用到的字段，而不是像 SparkContext.hadoopFile 中那样简单粗暴地扫描全部数据
- 也可以在程序中通过指定结构信息，将常规的 RDD 转化为SchemaRDD。这使得在 Python 或者 Java 对象上运行 SQL 查询更加简单
- 还可以自如地将这些 RDD 和来自其他 Spark SQL 数据源的SchemaRDD 进行连接操作
1. Apache Hive
	- 当从 Hive 中读取数据时，Spark SQL 支持任何 Hive 支持的存储格式（SerDe），包括文本文件、RCFiles、ORC、Parquet、Avro，以及 Protocol Buffer
	- 要把 Spark SQL 连接到已经部署好的 Hive 上，你需要提供一份 Hive 配置。你只需要把你的 hive-site.xml 文件复制到 Spark 的 ./conf/ 目录下即可
	- 如果你只是想探索一下 Spark SQL 而没有配置 hive-site.xml 文件，那么 Spark SQL 则会使用本地的 Hive 元数据仓，并且同样可以轻松地将数据读取到 Hive 表中进行查询
```python 3
# 使用 Python 从 Hive 读取
from pyspark.sql import HiveContext
hiveCtx = HiveContext(sc)
rows = hiveCtx.sql("SELECT key, value FROM mytable")
keys = rows.map(lambda row: row[0])
```
2. Parquet
	- Parquet（http://parquet.apache.org/ ）是一种流行的列式存储格式，可以高效地存储具有嵌套字段的记录
	- Parquet 格式经常在 Hadoop 生态圈中被使用，它也支持 Spark SQL 的全部数据类型
	- Spark SQL 提供了直接读取和存储 Parquet 格式文件的方法
```python 3
# Python 中的 Parquet 数据读取
# 从一个有name和favouriteAnimal字段的Parquet文件中读取数据
rows = hiveCtx.parquetFile(parquetFile)
names = rows.map(lambda row: row.name)
print "Everyone"
print names.collect()

# 也可以把 Parquet 文件注册为 Spark SQL 的临时表，并在这张表上运行查询语句
# Python 中的 Parquet 数据查询
# 寻找熊猫爱好者
tbl = rows.registerTempTable("people")
pandaFriends = hiveCtx.sql("SELECT name FROM people WHERE favouriteAnimal = \"panda\"")
print "Panda friends"
print pandaFriends.map(lambda row: row.name).collect()

# 可以使用 saveAsParquetFile() 把 SchemaRDD 的内容以 Parquet 格式保存
pandaFriends.saveAsParquetFile("hdfs://...")
```
3. JSON
	- 如果你有一个 JSON 文件，其中的记录遵循同样的结构信息，那么 Spark SQL 就可以通过扫描文件推测出结构信息，并且让你可以使用名字访问对应字段
	- 如果你在一个包含大量 JSON 文件的目录中进行尝试，你就会发现 Spark SQL 的结构信息推断可以让你非常高效地操作数据，而无需编写专门的代码来读取不同结构的文件
	- 要读取 JSON 数据，只要调用 hiveCtx 中的 jsonFile() 方法即可
	- 如果你想获得从数据中推断出来的结构信息，可以在生成的 SchemaRDD 上调用printSchema 方法
```python 3
# 输入记录
{"name": "Holden"}
{"name": "Sparky The Bear", "lovesPandas":true,"knows": {"friends":["holden"]}}

 # Python 中使用 Spark SQL 读取 JSON 数据
input = hiveCtx.jsonFile(inputFile)

# printSchema() 输出的结构信息
root
 |-- knows: struct (nullable = true)
 | |-- friends: array (nullable = true)
 | | |-- element: string (containsNull = false)
 |-- lovesPandas: boolean (nullable = true)
 |-- name: string (nullable = true)

# 推文的部分结构
root
 |-- contributorsIDs: array (nullable = true)
 | |-- element: string (containsNull = false)
 |-- createdAt: string (nullable = true)
 |-- currentUserRetweetId: integer (nullable = true)
 |-- hashtagEntities: array (nullable = true)
 | |-- element: struct (containsNull = false)
 | | |-- end: integer (nullable = true)
 | | |-- start: integer (nullable = true)
 | | |-- text: string (nullable = true)
 |-- id: long (nullable = true)
 |-- inReplyToScreenName: string (nullable = true)
 |-- inReplyToStatusId: long (nullable = true)
 |-- inReplyToUserId: long (nullable = true)
 |-- isFavorited: boolean (nullable = true)
 |-- isPossiblySensitive: boolean (nullable = true)
 |-- isTruncated: boolean (nullable = true)
 |-- mediaEntities: array (nullable = true)
 | |-- element: struct (containsNull = false)
 | | |-- displayURL: string (nullable = true)
 | | |-- end: integer (nullable = true)
 | | |-- expandedURL: string (nullable = true)
 | | |-- id: long (nullable = true)
 | | |-- mediaURL: string (nullable = true)
 | | |-- mediaURLHttps: string (nullable = true)
 | | |-- sizes: struct (nullable = true)
 | | | |-- 0: struct (nullable = true)
 | | | | |-- height: integer (nullable = true)
 | | | | |-- resize: integer (nullable = true)
 | | | | |-- width: integer (nullable = true)
 | | | |-- 1: struct (nullable = true)
 | | | | |-- height: integer (nullable = true)
 | | | | |-- resize: integer (nullable = true)
 | | | | |-- width: integer (nullable = true)
 | | | |-- 2: struct (nullable = true)
 | | | | |-- height: integer (nullable = true)
 | | | | |-- resize: integer (nullable = true)
 | | | | |-- width: integer (nullable = true)
 | | | |-- 3: struct (nullable = true)
 | | | | |-- height: integer (nullable = true)
 | | | | |-- resize: integer (nullable = true)
 | | | | |-- width: integer (nullable = true)
 | | |-- start: integer (nullable = true)
 | | |-- type: string (nullable = true)
 | | |-- url: string (nullable = true)
 |-- retweetCount: integer (nullable = true)
...
```
	- 如果你使用 Python，或已经把数据注册为了一张 SQL 表，你可以通过 . 来访问各个嵌套层级的嵌套元素（比如 toplevel.nextlevel）
	- 在 SQL 中可以通过用 [element] 指定下标来访问数组中的元素
```sql
-- 用 SQL 查询嵌套数据以及数组元素
select hashtagEntities[0].text from tweets LIMIT 1;
```
4. 基于RDD
除了读取数据，也可以基于 RDD 创建 SchemaRDD
	- 在 Scala 中，带有 case class 的 RDD可以隐式转换成 SchemaRDD
	- 在 Python 中，可以创建一个由 Row 对象组成的 RDD，然后调用 inferSchema()
```python 3
# 在 Python 中使用 Row 和具名元组创建 SchemaRDD
happyPeopleRDD = sc.parallelize([Row(name="holden", favouriteBeverage="coffee")])
happyPeopleSchemaRDD = hiveCtx.inferSchema(happyPeopleRDD)
happyPeopleSchemaRDD.registerTempTable("happy_people")
```
	 - 在 Java 中，可以调用 applySchema() 把 RDD 转为 SchemaRDD，只需要这个 RDD 中的数据类型带有公有的 getter 和 setter 方法，并且可以被序列化

#### JDBC/ODBC服务器
- JDBC 服务器作为一个独立的 Spark 驱动器程序运行，可以在多用户之间共享
	- 任意一个客户端都可以在内存中缓存数据表，对表进行查询
	- 集群的资源以及缓存数据都在所有用户之间共享
- Spark SQL 的 JDBC 服务器与 Hive 中的 HiveServer2 相一致
	- 由于使用了 Thrift 通信协议，它也被称为“Thrift server”
	- 注意，JDBC 服务器支持需要 Spark 在打开 Hive 支持的选项下编译
	- codegen 打开时，查询有可能会变慢，因为 Spark SQL 需要动态分析并编译代码，因此，短作业并不能真正体现 codegen 所带来的性能提升
- 服务器可以通过 Spark 目录中的 sbin/start-thriftserver.sh 启动
	 - 这个脚本接受的参数选项大多与 spark-submit 相同
	 - 默认情况下，服务器会 localhost:10000 上进行监听
	 	- 可以通过环境变量（HIVE_SERVER2_THRIFT_PORT 和 HIVE_SERVER2_THRIFT_BIND_HOST）修改这些设置
	 	- 也可以通过 Hive 配置选项（hive.server2.thrift.port 和 hive.server2.thrift.bind.host）来修改
	 	- 也可以通过命令行参数 --hiveconf property=value 来设置 Hive 选项
```
<!-- 启动 JDBC 服务器 -->
./sbin/start-thriftserver.sh --master sparkMaster
```
- Spark 也自带了 Beeline 客户端程序，我们可以使用它连接 JDBC 服务器
```
<!-- 使用 Beeline 连接 JDBC 服务器 -->
holden@hmbp2:~/repos/spark$ ./bin/beeline -u jdbc:hive2://localhost:10000
Spark assembly has been built with Hive, including Datanucleus jars on classpath
scan complete in 1ms
Connecting to jdbc:hive2://localhost:10000
Connected to: Spark SQL (version 1.2.0-SNAPSHOT)
Driver: spark-assembly (version 1.2.0-SNAPSHOT)
Transaction isolation: TRANSACTION_REPEATABLE_READ
Beeline version 1.2.0-SNAPSHOT by Apache Hive
0: jdbc:hive2://localhost:10000> show tables;
+---------+
| result |
+---------+
| pokes |
+---------+
1 row selected (1.182 seconds)
0: jdbc:hive2://localhost:10000>
```
_当启动 JDBC 服务器时，JDBC 服务器会在后台运行并且将所有输出重定向到一个日志文件中。如果你在使用 JDBC 服务器进行查询的过程中遇到了问题，可以查看日志寻找更为完整的报错信息_
- 许多外部工具也可以通过 ODBC 连接 Spark SQL
	 - Spark SQL 的 ODBC 驱动由 Simba（http://www.simba.com/ ）制作
	 - 可以从很多Spark供应商处下载到（比如DataBricksCloud、Datastax 以及 MapR）
	 - 由于 Spark SQL 使用了和Hive 相同的查询语言以及服务器，大多数可以连接到 Hive 的商务智能工具也可以通过已有的 Hive 连接器来连接到 Spark SQL 上
1. 使用Beeline
	- 在 Beeline 客户端中，你可以使用标准的 HiveQL 命令来创建、列举以及查询数据表
	- 可以从 Hive 语言手册（https://cwiki.apache.org/confluence/display/Hive/LanguageManual ）中找到关于 HiveQL 的所有语法细节
```sql
-- 首先，要从本地数据创建一张数据表，可以使用 CREATE TABLE 命令
-- 然后使用 LOAD DATA命令进行数据读取
-- 读取数据表 
> CREATE TABLE IF NOT EXISTS mytable (key INT, value STRING) 
 ROW FORMAT DELIMITED FIELDS TERMINATED BY ‘,’;
> LOAD DATA LOCAL INPATH ‘learning-spark-examples/files/int_string.csv’
 INTO TABLE mytable;

-- 要列举数据表，可以使用 SHOW TABLES 语句。你也可以通过 DESCRIBE tableName 查看每张表的结构信息

-- 列举数据表
> SHOW TABLES;
mytable
Time taken: 0.052 seconds

-- 想要缓存数据表，使用 CACHE TABLE tableName 语句
-- 缓存之后你可以使用 UNCACHE TABLE tableName 命令取消对表的缓存
-- 缓存的表会在这个 JDBC 服务器上的所有客户端之间共享

-- 最后，在 Beeline 中查看查询计划很简单，对查询语句运行 EXPLAIN 即可
-- Spark SQL shell 执行 EXPLAIN
spark-sql> EXPLAIN SELECT * FROM mytable where key = 1;
== Physical Plan ==
Filter (key#16 = 1)
 HiveTableScan [key#16,value#17], (MetastoreRelation default, mytable, None), None
Time taken: 0.551 seconds

-- 对于这个查询计划来说，Spark SQL 在一个 HiveTableScan 节点上使用了筛选操作
-- 你也可以直接写 SQL 语句对数据进行查询
-- Beeline shell 对于在多用户间共享的缓存数据表上进行快速的数据探索是非常有用的
```
2. 长生命周期的表与查询
- 使用 Spark SQL 的 JDBC 服务器的优点之一就是我们可以在多个不同程序之间共享缓存下来的数据表
	- JDBC Thrift 服务器是一个单驱动器程序，这就使得共享成为了可能
	- 如前一节中所述，你只需要注册该数据表并对其运行 CACHE 命令，就可以利用缓存了
- Spark SQL 独立 shell
	- 除了 JDBC 服务器，Spark SQL 也支持一个可以作为单独的进程使用的简易
	shell，可以通过 ./bin/spark-sql 启动
		- 这个 shell 会连接到你设置在 conf/hive-site.xml 中的 Hive 的元数据仓
		- 如果不存在这样的元数据仓，Spark SQL 也会在本地新建一个
	- 这个脚本主要对于本地开发比较有用
- 在共享的集群上，你应该使用 JDBC 服务器，让各用户通过 beeline 进行连接

#### 用户自定义函数
- 用户自定义函数，也叫 UDF，可以让我们使用 Python/Java/Scala 注册自定义函数，并在 SQL中调用
- 这种方法很常用，通常用来给机构内的 SQL 用户们提供高级功能支持，这样这些用户就可以直接调用注册的函数而无需自己去通过编程来实现了
- 在 Spark SQL 中，编写 UDF 尤为简单
- Spark SQL 不仅有自己的 UDF 接口，也支持已有的 Apache Hive UDF
1. Spark SQL UDF
	- 可以使用 Spark 支持的编程语言编写好函数，然后通过 Spark SQL 内建的方法传递进来，非常便捷地注册我们自己的 UDF
	- 在 Scala 和 Python 中，可以利用语言原生的函数和lambda 语法的支持，而在 Java 中，则需要扩展对应的 UDF 类
	- UDF 能够支持各种数据类型，返回类型也可以与调用时的参数类型完全不一样
	- 在 Python 和 Java 中，还需要用表 9-1 中列出的 SchemaRDD 对应的类型来指定返回值类型
	- Java 中的对应类型可以在 org.apache.spark.sql.api.java.DataType 中找到，而在Python 中则需要导入 DataType 支持
``` python 3
# Python 版本耳朵字符串长度 UDF
# 写一个求字符串长度的UDF
hiveCtx.registerFunction("strLenPython", lambda x: len(x), IntegerType())
lengthSchemaRDD = hiveCtx.sql("SELECT strLenPython('text') FROM tweets LIMIT 10")
```
2. Hive UDF
Spark SQL 也支持已有的 Hive UDF。标准的 Hive UDF 已经自动包含在了 Spark SQL 中。
	- 如果需要支持自定义的 Hive UDF，我们要确保该 UDF 所在的 JAR 包已经包含在了应用中
		- 需要注意的是，如果使用的是 JDBC 服务器，也可以使用 --jars 命令行标记来添加 JAR
	- 要使用 Hive UDF，应该使用 HiveContext，而不能使用常规的 SQLContext
		- 要注册一个Hive UDF，只需调用 hiveCtx.sql("CREATE TEMPORARY FUNCTION name AS class.function")

#### Spark SQL性能
<!-- Spark SQL 提供的高级查询语言及附加的类型信息可以使 Spark SQL 数据查询更加高效 -->
- Spark SQL 不仅是给熟悉 SQL 的用户使用的。
	- Spark SQL 使有条件的聚合操作变得非常容易，比如对多个列进行求值
	- 利用 Spark SQL 则不再需要创建一些特殊的对象来进行这种操作
```sql
-- Spark SQL 多列求和
SELECT SUM(user.favouritesCount), SUM(retweetCount), user.id 
FROM tweets
GROUP BY user.id
```
- Spark SQL 可以利用其对类型的了解来高效地表示数据
	- 当缓存数据时，Spark SQL 使用内存式的列式存储
		- 节约了缓存的空间
		- 尽可能地减少了后续查询中针对某几个字段查询时的数据读取
- 谓词下推可以让 Spark SQL 将查询中的一些部分工作“下移”到查询引擎上
	- 如果我们只需在 Spark 中读取某些特定的记录，标准的方法是读入整个数据集，然后在上面执行筛选条件
	- 在 Spark SQL 中，如果底层的数据存储支持只读取键值在一个范围内的记录，或是其他某些限制条件，Spark SQL 就可以把查询语句中的筛选限制条件推到数据存储层，从而大大减少需要读取的数据

**性能调优选项**

|选项 |默认值 |用途|
|:--:|:--:|:--:|
|spark.sql.codegen |false |设为 true 时，Spark SQL 会把每条查询语句在运行时编译为 Java 二进制代码。这可以提高大型查询的性能，但在进行小规模查询时会变慢|
|spark.sql.inMemoryColumnarStorage.compressed |false |自动对内存中的列式存储进行压缩|
|spark.sql.inMemoryColumnarStorage.batchSize |1000 |列式缓存时的每个批处理的大小。把这个值调大可能会导致内存不够的异常|
|spark.sql.parquet.compression.codec |snappy |使用哪种压缩编码器。可选的选项包括uncompressed/snappy/gzip/lzo|

- 使用 JDBC 连接接口和 Beeline shell 时，可以通过 set 令设置包括这些性能选项在内的各种选项
```beeline
<!-- 打开 codegen 选项的 Beeline 命令 -->
beeline> set spark.sql.codegen=true;
SET spark.sql.codegen=true
spark.sql.codegen=true
Time taken: 1.196 seconds
```
- 在一个传统的 Spark SQL 应用中，可以在 Spark 配置中设置这些 Spark 属性
```scala
<!-- 在 Scala 中打开 codegen 选项的代码 -->
conf.set("spark.sql.codegen", "true")
```
- 一些选项的配置需要给予特别的考量:
	- spark.sql.codegen
		- 这个选项可以让Spark SQL 把每条查询语句在运行前编译为 Java 二进制代码
		- 由于生成了专门运行指定查询的代码，codegen 可以让大型查询或者频繁重复的查询明显变快
		- 在运行特别快（1 ～ 2 秒）的即时查询语句时，codegen 有可能会增加额外开销，因为 codegen 需要让每条查询走一遍编译的过程
			- codegen 打开时最开始的几条查询会格外慢，因为 Spark SQL 需要初始化它的编译器。所以在测试 codegen 的额外开销之前你应该先运行 4 ～ 5 条查询语句
		- codegen 还是一个试验性的功能，但是我们推荐在所有大型的或者是重复运行的查询中使用 codegen
	- spark.sql.inMemoryColumnarStorage.batchSize
		- 缓存 SchemaRDD 时，Spark SQL 会按照这个选项制定的大小（默认值是 1000）把记录分组，然后分批压缩
			- 太小的批处理大小会导致压缩比过低，而批处理大小过大的话，比如当每个批次处理的数据超过内存所能容纳的大小时，也有可能会引发问题
			- 如果你表中的记录比较大（包含数百个字段或者包含像网页这样非常大的字符串字段），你就可能需要调低批处理大小来避免内存不够（OOM）的错误
			- 如果不是在这样的场景下，默认的批处理大小是比较合适的，因为压缩超过 1000 条记录时也基本无法获得更高的压缩比了

**操作 RDD 的方法同样适用于 Spark SQL中的 SchemaRDD**