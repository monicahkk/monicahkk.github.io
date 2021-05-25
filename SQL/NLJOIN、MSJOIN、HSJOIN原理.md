# 三种JOIN

<!-- The optimizer can choose one of three basic join strategies when queries require tables to be joined: nested-loop join, merge join, or hash join. -->
优化器在表联接的时候选择以下三种关联策略之一：NLJOIN, MSJOIN, HSJOIN

## Nested-loop join 
<!-- A nested-loop join is performed in one of the following two ways: -->
NLJOIN有如下两个执行方案：

- <!-- Scanning the inner table for each accessed row of the outer table -->
  依次取外部表的行,扫描内部表（全表）并找出所有满足连接条件的行

  <!-- For example, column A in table T1 and column A in table T2 have the following values: -->
  例如，外部表T1的列A和内部表T2的列A：

  | Outer table T1: Column A | Inner table T2: Column A |
  | :----------------------: | :----------------------: |
  |            2             |            3             |
  |            3             |            2             |
  |            3             |            2             |
  |                          |            3             |
  |                          |            1             |

  <!-- To complete a nested-loop join between tables T1 and T2, the database manager performs the following steps: -->
  数据库会遵循以下步骤对T1和T2进行NLJOIN：

  <!--
  1. Read the first row in T1. The value for A is 2.
	2. Scan T2 until a match (2) is found, and then join the two rows.
	3. Repeat Step 2 until the end of the table is reached.
	4. Go back to T1 and read the next row (3).
	5. Scan T2 (starting at the first row) until a match (3) is found, and then join the two rows.
	6. Repeat Step 5 until the end of the table is reached.
	7. Go back to T1 and read the next row (3).
	8. Scan T2 as before, joining all rows that match (3). 
-->

	1. 读取T1的第一行，此时值是2
	2. 扫描T2直到找到一个匹配的值(2)，然后联接这两行
	3. 重复步骤2直到扫描完T2表
	4. 返回T1表读取下一行(3)
	5. 扫描T2表(从第一行开始)直到找到一个匹配的值(3)，然后联接这两行
	6. 重复步骤5直到扫描完T2表
	7. 返回T1表读取下一行(3)
	8. 如前所述，扫描T2表，联接所有值为3的行
	
- <!-- Performing an index lookup on the inner table for each accessed row of the outer table -->
  依次取外部表的行,扫描内部表的索引找出所有满足连接条件的行

<!--   This method can be used if there is a predicate of the form:  -->
  可被用于如下形式的连接条件：
  ```
     expr(outer_table.column) relop inner_table.column
  ```
  <!-- where `relop` is a relative operator (for example =, >, >=, <, or <=) and `expr` is a valid expression on the outer table. For example: -->
  `relop`代表运算符(例如 =, >, >=, <, or <=)，`expr`代表外部表的一个合法表达式。例如：

  ```
     outer.c1 + outer.c2 <= inner.c1
     outer.c4 < inner.c3
  ```

  <!-- This method might significantly reduce the number of rows that are accessed in the inner table for each access of the outer table; the degree of benefit depends on a number of factors, including the selectivity of the join predicate. -->
  这个方法可以显著的减少表关联内表需要被扫描的行数；减少的程度取决于许多因素，包括连接类型的选择。

<!-- When it evaluates a nested-loop join, the optimizer also decides whether to sort the outer table before performing the join. If it sorts the outer table, based on the join columns, the number of read operations against the inner table to access pages on disk might be reduced, because they are more likely to be in the buffer pool already. If the join uses a highly clustered index to access the inner table, and the outer table has been sorted, the number of accessed index pages might be minimized. -->
在评估NLJOIN的时候，优化器也会决定是否在运行join前先对外部表排序。    
如果先基于join的列将外部表排序，访问磁盘上相关页进行读操作的次数可能会减少，因为他们有很大可能已经存在于缓冲池中。    
如果连接用了一个高聚簇的索引访问内部表，并且外部表已经被分割，访问磁盘上相关索引页进行读操作的次数将会是最小的。

<!-- If the optimizer expects that the join will make a later sort more expensive, it might also choose to perform the sort before the join. A later sort might be required to support a GROUP BY, DISTINCT, ORDER BY, or merge join operation. -->
如果优化器预计连接之后在排序会更消耗性能，优化器可能会选择先排序在连接。一个后执行的排序可能会被要求支持GROUP BY, DISTINCT, ORDER BY或者MSJOIN

## Merge join

<!-- A merge join, sometimes known as a merge scan join or a sort merge join, requires a predicate of the form `table1.column = table2.column`. This is called an equality join predicate. A merge join requires ordered input on the joining columns, either through index access or by sorting. A merge join cannot be used if the join column is a LONG field column or a large object (LOB) column.
 -->
MSJOIN有时被看作是一个合并扫描连接或者一个排序合并连接，MSJOIN要求连接条件形如`<表1>.<列> = <表2>.<列>`（等值连接）。    
MSJOIN要求输入的连接列有序（要么是通过索引要么是已经排序）。    
MSJOIN不能被用于连接LONG类型的列或者LOB列。    

<!-- In a merge join, the joined tables are scanned at the same time. The outer table of the merge join is scanned only once. The inner table is also scanned once, unless repeated values occur in the outer table. If repeated values occur, a group of rows in the inner table might be scanned again. -->
MSJOIN中，被连接的表被同时扫描。    
外部表只扫描一次。    
内部表也只扫描一次（除非外部表有重复值），如果有重复值，内部表的一组行可能会被再次扫描。    

<!-- For example, column A in table T1 and column A in table T2 have the following values: -->
例如：T1表的列A和T2表的列A如下所示：

| Outer table T1: Column A | Inner table T2: Column A |
| :----------------------: | :----------------------: |
|            2             |            1             |
|            3             |            2             |
|            3             |            2             |
|                          |            3             |
|                          |            3             |

<!-- To complete a merge join between tables T1 and T2, the database manager performs the following steps: -->
为了完成T1和T2的MSLJOIN，数据库管理器遵循如下步骤执行：

<!-- 
1. Read the first row in T1. The value for A is 2.
2. Scan T2 until a match (2) is found, and then join the two rows.
3. Keep scanning T2 while the columns match, joining rows.
4. When the 3 in T2 is read, go back to T1 and read the next row.
5. The next value in T1 is 3, which matches T2, so join the rows.
6. Keep scanning T2 while the columns match, joining rows.
7. When the end of T2 is reached, go back to T1 to get the next row. Note that the next value in T1 is the same as the previous value from T1, so T2 is scanned again, starting at the first 3 in T2. The database manager remembers this position. 
-->
1. 读取T1的第一行。此时A的值为2。
2. 扫描T2直到值为2的行（第2行），连接对应行（T1第1行与T2第2行）。
3. 继续扫描T2，如果值=2相等，则连接对应行（T1第1行与T2第3行）。
4. 当读到T2表中的3（第4行），返回T1读下一行（第2行）。
5. T1下一行的值=3（第2行），与T2表相等，连接两行（T1第2行与T2第4行）
6. 持续扫描T2表，如果值相等，则连接两行
7. 当到达T2表的末尾，回到T1表读取下一行。注意，如果T1表的下一行值与T1当前或之前行的值相等（第3行），T2表会被重新扫描（从T2表第3行开始），数据库管理器会记录这个位置（T2表第3行）

## Hash join

<!-- A hash join requires one or more predicates of the form `table1.columnX = table2.columnY` None of the columns can be either a LONG field column or a LOB column. -->
HSJOIN要求一个或多个形如`表1.列X = 表2.列Y`的连接等式。所有的列都不能是LONG类型及LOB类型。

<!-- A hash join is performed as follows: First, the designated inner table is scanned and rows are copied into memory buffers that are drawn from the sort heap specified by the **sortheap** database configuration parameter. The memory buffers are divided into sections, based on a hash value that is computed on the columns of the join predicates. If the size of the inner table exceeds the available sort heap space, buffers from selected sections are written to temporary tables. -->
HSJOIN运行步骤如下：
首先，扫描指定的内部表，将行复制到内存缓冲区（基于用数据库配置的“sortheap”指定的排序内存）。通过连接列的hash值，内存缓冲区被分成不同的区域。  如果内部表的大小超出了排序内存空间，对应区域的缓存会被写入临时表。

<!-- When the inner table has been processed, the second (or outer) table is scanned and its rows are matched with rows from the inner table by first comparing the hash value that was computed for the columns of the join predicates. If the hash value for the outer row column matches the hash value for the inner row column, the actual join predicate column values are compared. -->
在处理内部表的时候，外部表被扫描，并且内外表的行首先通过连接列的hash值进行匹配。如果hash值相等，再对比真实值。

<!-- Outer table rows that correspond to portions of the table that are not written to a temporary table are matched immediately with inner table rows in memory. If the corresponding portion of the inner table was written to a temporary table, the outer row is also written to a temporary table. Finally, matching pairs of table portions from temporary tables are read, the hash values of their rows are matched, and the join predicates are checked. -->
对应于没有被写入临时表的区域，外部表的相应行（hash值相等）会在内存中与内部表进行匹配。如果内部表的对应区域已经写入临时表，外部行的对应区域也会被写入一个临时表。最终，临时表中对应的区域被读取，此时行的hash值是匹配的，只需检查连捷列的真实值。

<!-- Hash join processing can also use filters to improve performance. If the optimizer chooses to use filters, and sufficient memory is available, filters can often reduce the number of outer table rows which need to be processed by the join. -->
HSJOIN也使用筛选器来提升性能。如果优化器选择使用筛选器，并且有效内存足够，筛选器通常可以减少外部表需要参与join的行数。

<!-- For the full performance benefit of hash joins, you might need to change the value of the **sortheap** database configuration parameter and the **sheapthres** database manager configuration parameter. -->
为了发挥出HSJOIN的全部优势，需要修改数据库配置参数“sortheap”及“sheapthres”的值

<!-- Hash join performance is best if you can avoid hash loops and overflow to disk. To tune hash join performance, estimate the maximum amount of memory that is available for **sheapthres**, and then tune the **sortheap** parameter. Increase its setting until you avoid as many hash loops and disk overflows as possible, but do not reach the limit that is specified by the **sheapthres** parameter. -->
当可以避免hash循环和磁盘溢出时，HSJOIN的效果最好。通过评估可用于“sheapthres”的最大内存，调整“sortheap”的参数，来测试HSJOIN的效果。提升HSJOIN的设置，尽可能的避免许多的hash循环和磁盘溢出，但是不要达到为“sheapthres”参数设定的极限值。

<!-- Increasing the **sortheap** value should also improve the performance of queries that have multiple sorts. -->
增加“sortheap”的值可以同样提升复杂排序的查询性能。

-------------------------------------------
### 说明：
#### 排序
排序分为共享排序（在数据库共享内存中排序）和私有排序（在agent的私有内存中排序）
DB2中有关排序内存的三个参数：SORTHEAP, SHEAPTHRES_SHR, SHEAPTHRES
- SORTHEAP：
  - 此参数定义需要排序堆内存的操作将使用的最大专用或共享内存页数。 
  - 如果排序为专用排序，那么此参数将影响代理程序专用内存。如果排序为共享排序，那么此参数将影响数据库共享内存。每个排序都有一个独立的排序堆，该排序堆由数据库管理器根据需要分配。此排序堆是将数据排序的区域。如果由优化器定向，那么将使用优化器提供的信息分配一个 比此参数指定的排序堆小的排序堆。
  <!-- 转自：https://www.ibm.com/docs/zh/db2/10.1.0?topic=parameters-sortheap-sort-heap-size -->
- SHEAPTHRES_SHR：
  - 表示对排序内存使用者每次可使用的数据库共享内存总量的软限制。
  <!-- 转自：https://www.ibm.com/docs/zh/db2/10.1.0?topic=SSEPGG_10.1.0/com.ibm.db2.luw.admin.config.doc/doc/r0006014.html?cp=SSEPGG_10.1.0 -->
- SHEAPTHRES： 
  - 此参数是对专用排序在任何给定时间可以使用的总内存量的实例范围软限制。当某个实例使用的专用排序内存总量达到此限制时，为其他传入专用排序请求分配的内存将显著减少。
  <!-- 转自：https://www.ibm.com/docs/zh/db2/10.1.0?topic=SSEPGG_10.1.0/com.ibm.db2.luw.admin.config.doc/doc/r0000260.html?cp=SSEPGG_10.1.0 -->

#### LOBs
- CLOB: 存储大型SBCS数据或混合数据，例如冗长的文档
- DBCLOB: 存储大型DBCS数据
- BLOB: 存储非传统数据，例如图片，语音和混合媒体。还可以存储结构化数据，以供不同类型和用户定义的函数使用。BLOB是二进制字符串。
<!-- 转自：https://www.ibm.com/docs/zh/db2-for-zos/11?topic=types-large-objects-lobs -->