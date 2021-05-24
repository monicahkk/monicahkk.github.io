## 在集群上运行Spark
### Spark运行时架构
- 分布式环境下，集群采用主/从结构：驱动器(Driver)节点[负责中央协调] & 执行器(executor)节点 -> 一个 Spark 应用(application) 
- Spark 应用通过一个叫作集群管理器（Cluster Manager）的外部服务在集群中的机器上启动。Spark 自带的集群管理器被称为独立集群管理器。Spark 也能运行在 Hadoop YARN 和 Apache Mesos 这两大开源集群管理器上

#### 驱动器节点
_Spark 驱动器是执行你的程序中的 main() 方法的进程 - 执行用户编写的用来创建
SparkContext、创建 RDD，以及进行 RDD 的转化操作和行动操作的代码_
- 把用户程序转为任务：Spark 驱动器程序负责把用户程序转为多个物理执行的单元，这些单元也被称为任务（task）
	所有的 Spark 程序都遵循同样的结构：
		- 程序从输入数据创建一系列 RDD
		- 再使用转化操作派生出新的 RDD
		- 最后使用行动操作收集或存储结果 RDD 中的数据
	隐式地创建出了一个由操作组成的逻辑上的有向无环图（Directed Acyclic Graph，简称 DAG）
	当驱动器程序运行时，它会把这个逻辑图转为物理执行计划
	Spark 会对逻辑执行计划作一些优化，把逻辑计划转为一系列步骤（stage），每个步骤又由多个任务组成，这些任务会被打包并送到集群中
任务是 Spark 中最小的工作单元
- 为执行器节点调度任务
	- 有了物理执行计划之后，Spark 驱动器程序必须在各执行器进程间协调任务的调度
	- 执行器进程启动后，会向驱动器进程注册自己，驱动器进程始终对应用中所有的执行器节点有完整的记录，每个执行器节点代表一个能够处理任务和存储 RDD 数据的进程
	- Spark 驱动器程序会根据当前的执行器节点集合，尝试把所有任务基于数据所在位置分配给合适的执行器进程。
		- 当任务执行时，执行器进程会把缓存数据存储起来，而驱动器进程同样会跟踪这些缓存数据的位置，并且利用这些位置信息来调度以后的任务，以尽量减少数据的网络传输。

#### 驱动器节点
_Spark 执行器节点是一种工作进程，负责在 Spark 作业中运行任务，任务间相互独立
Spark 应用启动时，执行器节点就被同时启动，并且始终伴随着整个 Spark 应用的生命周期而存在
如果有执行器节点发生了异常或崩溃，Spark 应用也可以继续执行_
两大作用：
1. 负责运行组成 Spark 应用的任务，并将结果返回给驱动器进程
2. 通过自身的块管理器（Block Manager）为用户程序中要求缓存的 RDD 提供内存式存储
RDD 是直接缓存在执行器进程内的，因此任务可以在运行时充分利用缓存数据加速运算
_在本地模式下，Spark 驱动器程序和各执行器程序在同一个 Java 进程中运行。这是一个特例；执行器程序通常都运行在专用的进程中_

#### 集群管理器
集群管理器是 Spark 中的可插拔式组件，除了 Spark 自带的独立集群管理器，Spark 也可以运行在其他外部集群管理器上，比如 YARN 和 Mesos
_Spark 文档中始终使用驱动器节点和执行器节点的概念来描述执行 Spark应用的进程。而主节点（master）和工作节点（worker）的概念则被用来
分别表述集群管理器中的中心化的部分和分布式的部分。这些概念很容易
混淆，所以要格外小心。例如，Hadoop YARN 会启动一个叫作资源管理器
（Resource Manager）的主节点守护进程，以及一系列叫作节点管理器（Node 
Manager）的工作节点守护进程。而在 YARN 的工作节点上，Spark 不仅可
以运行执行器进程，还可以运行驱动器进程_

#### 启动一个程序
可以使用 Spark 提供的统一脚本 spark-submit 将你的应用提交到那种集群管理器上
1. 用户通过 spark-submit 脚本提交应用。
2. spark-submit 脚本启动驱动器程序，调用用户定义的 main() 方法。
3. 驱动器程序与集群管理器通信，申请资源以启动执行器节点。
4. 集群管理器为驱动器程序启动执行器节点。
5. 驱动器进程执行用户应用中的操作。根据程序中所定义的对 RDD 的转化操作和行动操作，驱动器节点把工作以任务的形式发送到执行器进程。
6. 任务在执行器程序中进行计算并保存结果。
7. 如果驱动器程序的 main() 方法退出，或者调用了 SparkContext.stop()，驱动器程序会终止执行器进程，并且通过集群管理器释放资源

### 使用spark-submit部署应用
```sql
-- 提交 Python 应用
bin/spark-submit my_script.py
-- 如果在调用 spark-submit 时除了脚本或 JAR 包的名字之外没有别的参数，那么这个 Spark程序只会在本地执行

-- 提交应用时添加附加参数
bin/spark-submit --master spark://host:7077 --executor-memory 10g my_script.py
-- master 标记指定要连接的集群 URL
```
spark-submit的--master标记可以接收的值

|值 |描述|
|:--:|:--:|
|spark://host:port |连接到指定端口的 Spark 独立集群上。默认情况下 Spark 独立主节点使用 7077 端口|
|mesos://host:port |连接到指定端口的 Mesos 集群上。默认情况下 Mesos 主节点监听 5050 端口|
|yarn |连接到一个 YARN 集群。当在 YARN 上运行时，需要设置环境变量 HADOOP_CONF_DIR 指向 Hadoop 配置目录，以获取集群信息|
|local |运行本地模式，使用单核|
|local[N] |运行本地模式，使用 N 个核心|
|local[\*] |运行本地模式，使用尽可能多的核心|

spark-submit 提供的其他选项： 1. 调度信息； 2.是应用的运行时依赖
```sql
-- spark-submit 的一般格式
bin/spark-submit [options] <app jar | python file> [app options]

-- [options] 是要传给 spark-submit 的标记列表, spark-submit --help 列出所有可以接收的标记
-- <app jar | python File> 表示包含应用入口的 JAR 包或 Python 脚本
-- [app options] 是传给你的应用的选项
-- spark-submit 还允许通过 --conf prop=value 标记设置任意的 SparkConf 配置选项，也可以使用 --properties-File 指定一个包含键值对的属性文件
```
spark-submit的一些常见标记

|标记 |描述 |
|:--:|:--:|
|--master |表示要连接的集群管理器。这个标记可接收的选项见表 7-1|
|--deploy-mode |选择在本地（客户端“client”）启动驱动器程序，还是在集群中的一台工作节点机器（集群“cluster”）上启动。在客户端模式下，spark-submit 会将驱动器程序运行在 spark-submit 被调用的这台机器上。在集群模式下，驱动器程序会被传输并执行于集群的一个工作节点上。默认是本地模式|
|--class |运行 Java 或 Scala 程序时应用的主类|
|--name |应用的显示名，会显示在 Spark 的网页用户界面中|
|--jars |需要上传并放到应用的 CLASSPATH 中的 JAR 包的列表。如果应用依赖于少量第三方的 JAR 包，可以把它们放在这个参数里|
|--files |需要放到应用工作目录中的文件的列表。这个参数一般用来放需要分发到各节点的数据文件|
|--py-files |需要添加到 PYTHONPATH 中的文件的列表。其中可以包含 .py、.egg 以及 .zip 文件|
|--executor-memory |执行器进程使用的内存量，以字节为单位。可以使用后缀指定更大的单位，比如“512m”（512 MB）或“15g”（15 GB）|
|--driver-memory |驱动器进程使用的内存量，以字节为单位。可以使用后缀指定更大的单位，比如“512m”（512 MB）或“15g”（15 GB）|

```sql
-- 使用各种选项调用 spark-submit
 -- 使用独立集群模式提交Java应用
$	./bin/spark-submit \
	--master spark://hostname:7077 \
	--deploy-mode cluster \
	--class com.databricks.examples.SparkExample \
	--name "Example Program" \
	--jars dep1.jar,dep2.jar,dep3.jar \
	--total-executor-cores 300 \
	--executor-memory 10g \
	myApp.jar "options" "to your application" "go here"

-- 使用YARN客户端模式提交Python应用
$	export HADOP_CONF_DIR=/opt/hadoop/conf
$	./bin/spark-submit \
	--master yarn \
	--py-files somelib-1.2.egg,otherlib-4.4.zip,other-file.py \
	--deploy-mode client \
	--name "Example Program" \
	--queue exampleQueue \
	--num-executors 40 \
	--executor-memory 10g \
	my_script.py "options" "to your application" "go here"
```

### 打包代码与依赖
1. 对于 Python 用户而言，有多种安装第三方库的方法。
	- 由于 PySpark 使用工作节点机器上已有的 Python 环境，你可以通过标准的 Python 包管理器（比如 pip 和 easy_install）直接在集群中的所有机器上安装所依赖的库，或者把依赖手动安装到 Python 安装目录下的 site-packages/ 目录中。
	- 你也可以使用 spark-submit 的 --py-Files 参数提交独立的库，这样它们也会被添加到 Python 解释器的路径中。
	- 如果你没有在集群上安装包的权限，可以手动添加依赖库，这也很方便，但是要防范与已经安装在集群上的那些包发生冲突
_当你提交应用时，绝不要把 Spark 本身放在提交的依赖中。spark-submit 会
自动确保 Spark 在你的程序的运行路径中_
2. Java 和 Scala 用户也可以通过 spark-submit 的 --jars 标记提交独立的 JAR 包依赖
	- 当只有一两个库的简单依赖，并且这些库本身不依赖于其他库时，这种方法比较合适
	- 常规的做法是使用构建工具，生成单个大 JAR 包，包含应用的所有的传递依赖。这通常被称为超级（uber）JAR 或者组合（assembly）JAR
		- 使用最广泛的构建工具是 Maven 和 sbt。它们都可以用于这两种语言，不过Maven 通常用于 Java 工程，而 sbt 则一般用于 Scala 工程
	- 两个例子
```spark
<!-- 使用 Maven 构建的 Spark 应用的 pom.xml 文件 -->

<project>
	<modelVersion>4.0.0</modelVersion>

	<!-- 工程相关信息 -->
	<groupId>com.databricks</groupId>
	<artifactId>example-build</artifactId>
	<name>Simple Project</name>
	<packaging>jar</packaging>
	<version>1.0</version>

	<dependencies>
		<!-- Spark依赖 -->
		<dependency>
			<groupId>org.apache.spark</groupId>
			<artifactId>spark-core_2.10</artifactId>
			<version>1.2.0</version>
			<scope>provided</scope>
		</dependency>
		<!-- 第三方库 -->
		<dependency>
			<groupId>net.sf.jopt-simple</groupId>
			<artifactId>jopt-simple</artifactId>
			<version>4.3</version>
		</dependency>
		<!-- 第三方库 -->
		<dependency>
			<groupId>joda-time</groupId>
			<artifactId>joda-time</artifactId>
			<version>2.0</version>
		</dependency>
		</dependencies>

	<build>
		<plugins>
		<!-- 用来创建超级JAR包的Maven shade插件 -->
		<plugin>
			<groupId>org.apache.maven.plugins</groupId>
			<artifactId>maven-shade-plugin</artifactId>
			<version>2.3</version>
			<executions>
				<execution>
					<phase>package</phase>
					<goals>
						<goal>shade</goal>
						</goals>
					</execution>
				</executions>
			</plugin>
		</plugins>
	</build>
</project>

<!-- 声明了两个传递依赖：jopt-simple 和 joda-time，前者用来作选项解析，而后者是一个用来处理时间日期转换的工具库 -->
<!-- 这个工程也依赖于 Spark，不过把 Spark 标记为 provided 来确保 Spark 不与应用依赖的其他工件打包在一起 -->
<!-- 构建时，我们使用 maven-shade-plugin 插件来创建出包含所有依赖的超级 JAR 包。你可以让 Maven 在每次进行打包时执行插件的 shade 目标来使用此插件 -->

# 打包使用 Maven 构建的 Spark 应用
$ mvn package
# 在目标路径中，可以看到超级JAR包和原版打包方法生成的JAR包
$ ls target/
example-build-1.0.jar
original-example-build-1.0.jar
# 展开超级JAR包可以看到依赖库中的类
$ jar tf target/example-build-1.0.jar
...
joptsimple/HelpFormatter.class
...
org/joda/time/tz/UTCProvider.class
...
# 超级JAR可以直接传给spark-submit
$ /path/to/spark/bin/spark-submit --master local ... target/example-build-1.0.jar
```
```sbt
// 使用 sbt 0.13 的 Spark 应用的 build.sbt 文件
// sbt 构建文件是用配置语言写成的，在这个文件中我们把值赋给特定的键，用来定义工程的构建

import AssemblyKeys._

name := "Simple Project"

version := "1.0"

organization := "com.databricks"

scalaVersion := "2.10.3"

libraryDependencies ++= Seq(
	// Spark依赖
	"org.apache.spark" % "spark-core_2.10" % "1.2.0" % "provided",
	// 第三方库
	"net.sf.jopt-simple" % "jopt-simple" % "4.3",
	"joda-time" % "joda-time" % "2.0"
)

// 这条语句打开了assembly插件的功能
assemblySettings

// 配置assembly插件所使用的JAR
jarName in assembly := "my-project-assembly.jar"

// 一个用来把Scala本身排除在组合JAR包之外的特殊选项，因为Spark
// 已经包含了Scala
assemblyOption in assembly :=
	(assemblyOption in assembly).value.copy(includeScala = false)

// 这个构建文件的第一行从插件中引入了一些功能，这个插件用来支持创建项目的组合 JAR包。要使用这个插件，需要在 project/ 目录下加入一个小文件，来列出对插件的依赖。
// 我们只需要创建出 project/assembly.sbt 文件，并在其中加入 addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.11.2")。
// sbt-assembly 的实际版本可能会因使用的 sbt 版本不同而变化。例 7-8 适用于 sbt 0.13。

// 例 7-8：在 sbt 工程构建中添加 assembly 插件
# 显示project/assembly.sbt的内容
$ cat project/assembly.sbt
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.11.2")

// 定义好了构建文件之后，就可以创建出一个完全组合打包的 Spark 应用 JAR 包（例 7-9）。

// 例 7-9：打包使用 sbt 构建的 Spark 应用
$ sbt assembly
# 在目标路径中，可以看到一个组合JAR包
$ ls target/scala-2.10/
my-project-assembly.jar
# 展开组合JAR包可以看到依赖库中的类
$ jar tf target/scala-2.10/my-project-assembly.jar
...
joptsimple/HelpFormatter.class
...
org/joda/time/tz/UTCProvider.class
...
# 组合JAR可以直接传给spark-submit
$ /path/to/spark/bin/spark-submit --master local ... 
 target/scala-2.10/my-project-assembly.jar
```
3. 依赖冲突
依赖冲突表现为 Spark 作业执行过程中抛出 NoSuchMethodError、ClassNotFoundException，或其他与类加载相关的 JVM 异常
主要有两种解决方式：
	- 修改你的应用，使其使用的依赖库的版本与 Spark 所使用的相同
	- 使用通常被称为“shading”的方式打包你的应用。
		- Maven 构建工具通过对例 7-5 中使用的插件（事实上，shading 的功能也是这个插件取名为 maven-shade-plugin 的原因）进行高级配置来支持这种打包方式。
		- shading 可以让你以另一个命名空间保留冲突的包，并自动重写应用的代码使得它们使用重命名后的版本

### Spark应用内与应用间调度
在调度多用户集群时，Spark 主要依赖集群管理器来在 Spark 应用间共享资源
- Spark 应用有一种特殊情况，就是那些长期运行（long lived）的应用：
	- Spark SQL 中的 JDBC 服务器就是一个长期运行的 Spark 应用
	- 当 JDBC服务器启动后，它会从集群管理器获得一系列执行器节点，然后就成为用户提交 SQL 查询的永久入口
	- Spark 提供了一种用来配置应用内调度策略的机制
	- Spark 内部的公平调度器（Fair Scheduler）会让长期运行的应用定义调度任务的优先级队列 [官方文档：(http://spark.apache.org/docs/latest/job-scheduling.html )]

### 集群管理器
Spark 可以运行在各种集群管理器上，并通过集群管理器访问集群中的机器
- 在一堆机器上运行 Spark —— 自带的独立模式是部署该集群最简单的方法
- 需要与别的分布式应用共享的集群 —— Hadoop YARN 与 Apache Mesos
- 部署到 Amazon EC2

#### 独立集群管理器
这种集群管理器由一个主节点和几个工作节点组成，各自都分配有一定量的内存和 CPU 核心
1. 启动独立集群管理器：
	- 既可以通过手动启动一个主节点和多个工作节点来实现，也可以使用 Spark 的 sbin 目录中的启动脚本来实现
	- 需要预先设置机器间的 SSH 无密码访问
	- 步骤：
		1. 将编译好的 Spark 复制到所有机器的一个相同的目录下，比如 /home/yourname/spark
		2. 设置好从主节点机器到其他机器的 SSH 无密码登陆。这需要在所有机器上有相同的用户账号，并在主节点上通过 ssh-keygen 生成 SSH 私钥，然后将这个私钥放到所有工作节点的 .ssh/authorized_keys 文件中。
```unix
# 如果你之前没有设置过这种配置，你可以使用如下命令：
# 在主节点上：运行ssh-keygen并接受默认选项
$ ssh-keygen -t dsa
Enter file in which to save the key (/home/you/.ssh/id_dsa): [回车]
Enter passphrase (empty for no passphrase): [空]
Enter same passphrase again: [空]

# 在工作节点上：
# 把主节点的~/.ssh/id_dsa.pub文件复制到工作节点上，然后使用：
$ cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys
$ chmod 644 ~/.ssh/authorized_keys
```
		3. 编辑主节点的 conf/slaves 文件并填上所有工作节点的主机名。
		4. 在主节点上运行 sbin/start-all.sh（要在主节点上运行而不是在工作节点上）以启动集群。 如果全部启动成功，你不会得到需要密码的提示符，而且可以在 http://masternode:8080 看到集群管理器的网页用户界面，上面显示着所有的工作节点。
		5. 要停止集群，在主节点上运行 bin/stop-all.sh。
```unix
# 如果你使用的不是 UNIX 系统，或者想手动启动集群，你也可以使用 Spark 的 bin/ 目录下的 spark-class 脚本分别手动启动主节点和工作节点。在主节点上，输入：
	bin/spark-class org.apache.spark.deploy.master.Master
#然后在工作节点上输入：
	bin/spark-class org.apache.spark.deploy.worker.Worker spark://masternode:7077
#（其中 masternode 是你的主节点的主机名）。在 Windows 中，使用 \ 来代替 /
```
	- 默认情况下，集群管理器会选择合适的默认值自动为所有工作节点分配 CPU 核心与内存
	- 配置独立集群管理器的更多细节请参考 Spark 的官方文档( http://spark.apache.org/docs/latest/spark-standalone.html )
2. 提交应用
```sql
-- 要向独立集群管理器提交应用，需要把 spark://masternode:7077 作为主节点参数传给spark-submit。例如：
spark-submit --master spark://masternode:7077 yourapp

-- 这个集群的 URL 也显示在独立集群管理器的网页界面（位于 http://masternode:8080）上。
-- 注意，提交时使用的主机名和端口号必须精确匹配用户界面中的 URL

-- 你可以使用 --master 参数以同样的方式启动 spark-shell 或 pyspark，来连接到该集群上：
spark-shell --master spark://masternode:7077
pyspark --master spark://masternode:7077
```
	- 要检查你的应用或者 shell 是否正在运行，你需要查看集群管理器的网页用户界面 http://masternode:8080 并确保：
		1. 应用连接上了（即出现在了 Running Applications 中）
		2. 列出的所使用的核心和内存均大于 0
	- 阻碍应用运行的一个常见陷阱是 _为执行器进程申请的内存（spark-submit 的--executor-memory 标记传递的值）超过了集群所能提供的内存总量_。在这种情况下，独立集群管理器始终无法为应用分配执行器节点。请确保应用申请的值能够被集群接受。
*支持两种部署模式*:
	- 在客户端模式中（默认情况），驱动器程序会运行在你执行 spark-submit 的机器上，是 spark-submit 命令的一部分。
		- 这意味着你可以直接看到驱动器程序的输出，也可以直接输入数据进去（通过交互式 shell）
		- 但是这要求你提交应用的机器与工作节点间有很快的网络速度，并且在程序运行的过程中始终可用
	在集群模式下，驱动器程序会作为某个工作节点上一个独立的进程运行在独立集群管理器内部。它也会连接主节点来申请执行器节点。
		- 在这种模式下，spark-submit 是“一劳永逸”型，你可以在应用运行时关掉你的电脑
		- 你还可以通过集群管理器的网页用户界面访问应用的日志
	- 向 spark-submit 传递 --deploy-mode cluster 参数可以切换到集群模式。
3. 配置资源用量
<!-- Apache Mesos 支持应用运行时的更动态的资源共享，而 YARN 则有分级队列的概念 -->
独立集群管理器中，资源分配靠下面两个设置来控制:
- 执行器进程内存 - 控制执行器节点占用工作节点的多少内存 - 默认值是 1 GB
	- 你可以通过 spark-submit 的 --executor-memory 参数来配置此项
	- 每个应用在每个工作节点上最多拥有一个执行器进程（一台机器上可以运行多个从节点）
- 占用核心总数的最大值 - 默认值是无限
	- 可以通过 spark-submit 的 --total-executorcores 参数设置这个值，或者在你的 Spark 配置文件中设置 spark.cores.max 的值
	- 应用中所有执行器进程所占用的核心总数
- 验证这些设定：独立集群管理器的网页用户界面（http://masternode:8080）中查看当前的资源分配情况

独立集群管理器在默认情况下会为每个应用使用尽可能分散的执行器进程
- 可以通过设置配置属性 spark.deploy.spreadOut为 false 来要求 Spark 把执行器进程合并到尽量少的工作节点中
4. 高度可用性
- 独立模式能够很好地支持工作节点的故障
- 如果你想让集群的主节点也拥有高度可用性，Spark 还支持使用 Apache ZooKeeper（一个分布式协调系统）来维护多个备用的主节点，并在一个主节点失败时切换到新的主节点上
	- 官方文档（https://spark.apache.org/docs/latest/spark-standalone.html#high-availability ）

#### Hadoop YARN
- 可以让多种数据处理框架运行在一个共享的资源池上，并且通常安装在与 Hadoop 文件系统（简称 HDFS）相同的物理节点上
- 在 Spark 里使用 YARN 很简单：需要设置指向你的 Hadoop 配置目录的环境变量，然后使用 spark-submit 向一个特殊的主节点 URL 提交作业即可
1. 找到你的 Hadoop 的配置目录，并把它设为环境变量 HADOOP_CONF_DIR
```hadoop
<!-- 这个目录包含 yarn-site.xml 和其他配置文件；
如果你把 Hadoop 装到 HADOOP_HOME 中，那么这个目录通常位于 HADOOP_HOME/conf 中，否则可能位于系统目录 /etc/hadoop/conf 中。
然后用如下方式提交你的应用： -->
export HADOOP_CONF_DIR="..."
spark-submit --master yarn yourapp
```
	- 有两种将应用连接到集群的模式：客户端模式以及集群模式
		- 在客户端模式下应用的驱动器程序运行在提交应用的机器上（比如你的笔记本电脑），
		- 在集群模式下，驱动器程序也运行在一个 YARN 容器内部
		- 分别使用 yarn-client 和 yarn-cluster 两种参数
	- Spark 的交互式 shell 以及 pyspark 也都可以运行在 YARN 上。只要设置好 HADOOP_CONF_DIR 并对这些应用使用 --master yarn 参数即可
		- 由于这些应用需要从用户处获取输入，所以只能运行于客户端模式下
2. 配置资源用量
- 根据你在 spark-submit 或 spark-shell 等脚本的 --num-executors标记中设置的值，Spark 应用会使用固定数量的执行器节点
	- 默认值仅为 2
	- 可以设置通过 --executor-memory 设置每个执行器的内存用量，通过 --executor-cores 设置每个执行器进程从 YARN 中占用的核心数目
*Spark 通常在用量较大而总数较少的执行器组合（使用多核与更多内存）上
表现得更好，因为这样 Spark 可以优化各执行器进程间的通信*
<!-- 一些集群限制了每个执行器进程的最大内存（默认为 8 GB） -->
	- 出于资源管理的目的，某些 YARN 集群被设置为将应用调度到多个队列中
		- 使用 --queue 选项来选择你的队列的名字
	- 官方文档（ http://spark.apache.org/docs/latest/submitting-applications.html ）

#### Apache Mesos
- Apache Mesos 是一个通用集群管理器，既可以运行分析型工作负载又可以运行长期运行的服务（比如网页服务或者键值对存储）
- 要在 Mesos 上使用 Spark，需要把一个 mesos:// 的URI 传给 spark-submit
```
spark-submit --master mesos://masternode:5050 yourapp

<!-- 在运行多个主节点时，你可以使用 ZooKeeper 来为 Mesos 集群选出一个主节点点。在这种情况下，应该使用 mesos://zk:// 的 URI 来指向一个 ZooKeeper 节点列表。 -->

<!-- 比如，你有三个ZooKeeper 节点（node1、node2 和 node3），并且 ZooKeeper 分别运行在三台机器的 2181 端口上时，你应该使用如下 URI： -->
mesos://zk://node1:2181/mesos,node2:2181/mesos,node3:2181/mesos
```
1. Mesos调度模式
Mesos 提供了两种模式来在一个集群内的执行器进程间共享资源:
	- 在“细粒度”模式（默认）中，执行器进程占用的 CPU 核心数会在它们执行任务时动态变化，因此 _一台运行了多个执行器进程的机器可以动态共享 CPU 资源_
		- 当多用户共享的集群运行 shell 这样的交互式的工作负载时，由于应用会在它们不工作时降低它们所占用的核心数，以允许别的用户程序使用集群，这种情况下细粒度模式显得非常合适
		- 在细粒度模式下调度任务会带来更多的延迟（这样的话，一些像 Spark Streaming 这样需要低延迟的应用就会表现很差），应用需要在用户输入新的命令时，为重新分配 CPU 核心等待一段时间
	- 在“粗粒度”模式中，Spark 提前为每个执行器进程分配固定数量的 CPU 数目，并且在应用结束前绝不释放这些资源，哪怕执行器进程当前不在运行任务
	- 以通过向 spark-submit 传递 --conf spark.mesos.coarse=true 来打开粗粒度模式
		- 可以在一个 Mesos 集群中使用混合的调度模式（比如将一部分 Spark 应用的 spark.mesos.coarse 设置为 true，而另一部分不这么设置）
2. 客户端和集群模式
	- Spark 1.2 而言，在 Mesos 上 Spark 只支持以客户端的部署模式运行应用
	- 如果你还是希望在 Mesos 集群中运行你的驱动器节点，那么 Aurora（http://aurora.apache.org/ ）或 Chronos（http://airbnb.io/chronos ）这样的框架可以让你将任意脚本提交到 Mesos 上执行，并监控它们的运行
3. 配置资源用量
	- 以通过 spark-submit 的两个参数 --executor-memory 和 --total-executor-cores 来控制运行在 Mesos 上的资源用量
		- 前者用来设置每个执行器进程的内存
		- 后者则用来设置应用占用的核心数（所有执行器节点占用的总数）的最大值
	- 默认情况下，Spark 会使用尽可能多的核心启动各个执行器节点，来将应用合并到尽量少的执行器实例中，并为应用分配所需要的核心数
	- 如果你不设置 --total-executor-cores 参数，Mesos 会尝试使用集群中所有可用的核心

#### Amazon EC2
- Spark 自带一个可以在 Amazon EC2 上启动集群的脚本。这个脚本会启动一些节点，并且在它们上面安装独立集群管理器
	- EC2 脚本还会安装好其他相关的服务，比如HDFS、Tachyon 还有用来监控集群的 Ganglia
	- Spark 的 EC2 脚本叫作 spark-ec2，位于 Spark 安装目录下的 ec2 文件夹中。它需要 Python 2.6 或更高版本的运行环境
		- 可以在下载 Spark 后直接运行 EC2 脚本而无需预先编译Spark
- EC2 脚本可以管理多个已命名的集群（cluster），并且使用 EC2 安全组来区分它们。
	- 对于每个集群，脚本都会为主节点创建一个叫作 clustername-master 的安全组，而为工作节点创建一个叫作 clustername-slaves 的安全组
1. 启动集群
	- 要启动一个集群，你应该先创建一个 Amazon 网络服务（AWS）账号，并且获取访问键ID 和访问键密码，然后把它们设在环境变量中：
```
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```
	- 然后再创建出 EC2 的 SSH 密钥对，然后下载私钥文件（通常叫作 keypair.pem），这样你就可以 SSH 到你的机器上
	- 接下来，运行 spark-ec2 脚本的 launch 命令，提供你的密钥对的名字、私钥文件和集群的名字。
```spark
# 默认情况下，这条命令会使用 m1.xlarge 类型的 EC2 实例，启动一个有一个主节点和一个工作节点的集群：
cd /path/to/spark/ec2
./spark-ec2 -k mykeypair -i mykeypair.pem launch mycluste

# 你也可以使用 spark-ec2 的参数选项配置实例的类型、工作节点个数、EC2 地区，以及其他一些要素。例如：
# 启动包含5个m3.xlarge类型的工作节点的集群
./spark-ec2 -k mykeypair -i mykeypair.pem -s 5 -t m3.xlarge launch mycluster

# 要获得参数选项的完整列表，运行 spark-ec2 --help
```
spark-ec2的常见选项

|选项 |含义|
|:--:|:--:|
|-k KEYPAIR |要使用的 keypair 的名字|
|-i IDENTITY_FiLE |私钥文件（以 .pem 结尾）|
|-s NUM_SLAVES |工作节点数量|
|-t INSTANCE_TYPE |使用的实例类型|
|-r REGION |使用 Amazon 实例所在的区域（例如 us-west-1）|
|-z ZONE |使用的地带（例如 us-west-1b）|
|--spot-price=PRICE |在给定的出价使用 spot 实例（单位为美元）|

<!-- 从启动脚本开始，通常需要五分钟左右来完成启动机器、登录到机器上并配置 Spark 的全部过程 -->
2. 登录集群
可以使用存有私钥的 .pem 文件通过 SSH 登录到集群的主节点上
```
# spark-ec2提供了登录命令
./spark-ec2 -k mykeypair -i mykeypair.pem login mycluster

#可以通过运行下面的命令获得主节点的主机名
./spark-ec2 get-master mycluster

#然后自行使用 命令 SSH 到主节点上
ssh -i keypair.pem root@masternode 

# 进入集群以后，就可以使用 /root/spark 中的 Spark 环境来运行程序了
# 这是一个独立模式的集群环境，主节点 URL 是 spark://masternode:7077
# 当你使用 spark-submit 启动应用时，Spark 会自动配置为将应用提交到这个独立集群上
# 你可以从 http://masternode:8080 访问集群的网页用户界面

# 只有从集群中的机器上启动的程序可以使用 spark-submit 把作业提交上去；出于安全目的，防火墙规则会阻止外部主机提交作业
# 要在集群上运行一个预先打包的应用，需要先把程序包通过 SCP 复制到集群上：
scp -i mykeypair.pem app.jar root@masternode:~
```
3. 销毁集群
```
./spark-ec2 destroy mycluster
# 这条命令会终止集群中的所有的实例（包括 mycluster-master 和 mycluster-slaves 两个安全组中的所有实例）
```
4. 暂停和重启集群
spark-ec2 还可以让你中止运行集群的 Amazon 实例，并且可以让你稍后重启这些实例
	- 停止这些实例会将它们关机，并丢失“临时”盘上的所有数据
	- 中止的实例会保留 root 目录下的所有数据（例如你上传到那里的所有文件），这样你就可以快速恢复自己的工作
```
# 要中止一个集群，运行：
./spark-ec2 stop mycluster

# 然后，过一会儿，再次启动集群：
./spark-ec2 -k mykeypair -i mykeypair.pem start mycluster

# Spark EC2 的脚本并没有提供调整集群大小的命令，但你可以通过增减mycluster-slaves 安全组中的机器来实现对集群大小的控制

# 要增加机器，首先应当中止集群，然后使用 AWS 管理控制台，右击一台工作节点并选择“Launch more like this（启动更多像这个实例一样的实例）”。
# 然后使用 spark-ec2 start 启动集群。

# 要移除机器，只需在 AWS 控制台上终止这一实例即可（不过要小心，这也会破坏集群中 HDFS 上的数据）
```
5. 集群存储
Spark EC2 集群已经配置好了两套 Hadoop 文件系统以供存储临时数据
	- “临时”HDFS，使用节点上的临时盘
		- 大多数类型的 Amazon 实例都在“临时”盘上带有大容量的本地空间，这些空间会在实例关机时消失
		- 这种文件系统安装在节点的 /root/ephemeral-hdfs 目录中，你可以使用 bin/hdfs 命令来访问并列出文件
		- 你也可以访问这种 HDFS 的网页用户界面，其 URL 地址位于 http://masternode:50070
	- “永久”HDFS，使用节点的 root 分卷
		- 这种 HDFS 在集群重启时仍然保留数据，不过一般空间较小，访问起来也比临时的那种慢
		- 这种 HDFS 适合存放你不想多次下载的中等大小的数据集中
		- 它安装于 /root/persistent-hdfs 目录，网页用户界面地址是 http://masternode:60070
	- 除了这些以外，你最有可能访问的就是 Amazon S3 中的数据了。你可以在 Spark 中使用s3n:// 的 URI 结构来访问其中的数据

### 选择合适的集群管理器
- 如果是从零开始，可以先选择独立集群管理器。
	- 独立模式安装起来最简单，而且如果你只是使用 Spark 的话，独立集群管理器提供与其他集群管理器完全一样的全部功能
- 如果你要在使用 Spark 的同时使用其他应用，或者是要用到更丰富的资源调度功能
（例如队列），那么 YARN 和 Mesos 都能满足你的需求
	- 在这两者中，对于大多数Hadoop 发行版来说，一般 YARN 已经预装好了。
- Mesos 相对于 YARN 和独立模式的一大优点在于其细粒度共享的选项，该选项可以将类似 Spark shell 这样的交互式应用中的不同命令分配到不同的 CPU 上
	- 对于多用户同时运行交互式 shell 的用例更有用处
- 在任何时候，最好把 Spark 运行在运行 HDFS 的节点上，这样能快速访问存储
	- 可以自行在同样的节点上安装 Mesos 或独立集群管理器
	- 如果使用 YARN 的话，大多数发行版已经把 YARN 和 HDFS 安装在了一起
- 依然应当查阅所使用Spark 版本的官方文档（http://spark.apache.org/docs/latest/ ）来了解最新的选项