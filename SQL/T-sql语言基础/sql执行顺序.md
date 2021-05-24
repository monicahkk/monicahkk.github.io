**逻辑处理顺序**：

- from
- where
- group by
- having
- select
  - over
  - distinct
  - top
- order by

**运算符优先级**：
1. （）(parentheses)
2. * (Multiply)，/(Division), %(Modulo)
3. +(Positive)，- (Negative)，+(Add)，-(Subtract)
4. =, >, <, >=, <=, <>, !=, !>, !<
5. NOT
6. AND
7. BETWEEN, IN, LIKE, OR
8. = (Assignment)
# !=, !>, !< 非标准运算符，建议少用

CASE:简单表达式和搜索表达式

逻辑表达式涉及缺少的值，计算结果为UNKNOWN
对查询过滤条件处理的定义：接受TRUE
对CHECK约束处理的定义：拒绝FALSE
UNKNOWN取反(negate)，结果仍为UNKNOWN
NULL=NULL 结果为UNKNOWN，IS NULL 和 IS NOT NULL
分组和排序时，认为两个NULL是相等的

同时操作（ALL-At-Once Operation）

避免除0
```
SELECT col1, col2
FROM dbo.T1
WHERE cpl1 <> 0 and col2 > 2*col1;
```

**数据类型**
普通字符：一个字节(byte)保存每个字符 —— CHAR, VARCHAR —— 限制只能使用英文 —— 'Hello'
Unicode字符：两个字节 —— NCHAR, NVARCHAR —— 所有语言都可用相同的Unicode码表示 —— N'Hello'
CHAR,NCHAR —— 读取数据代价较高
VARCHAR,NVARCHAR —— 更新的效率较低 —— 可使用MAX说明符

列的有效排序规则(collation)
查询系统目前支持的所有排序规则和描述
```
SELECT name, description
FROM sys.fn_helpcollations();
```

串联运算
NULL串联还是NULL

|| 函数名 || 说明 ||
| COALESCE(a，b) | 接受一列输入值，返回其中第一个不为NULL的值 |
| SUBSTRING(string, start, length) | 从字符串中提取子串 |
| LEFT(string, n) 和 RIGHT(string, n) | 返回输入字符串中从左/右开始指定个数的字符 |
| LEN(string) | 字符数 |
| datalength(string) | 字节数 |
| charindex(substeing, string[,start_pos]) | 返回某个子串第一次出现的起始位置 |
| PATINDEX(pattern, string) | 某个模式第一次出现的起始位置 |
| REPLACE(string, substring1, substring2) | 替换 |
| REPLICATE(string, n) | 指定次数复制字符串值 |
| STUFF(string, pos, delete_length, insertstring) | 删除字符串中一个子串，再插入一个新的子字符串作为替换 |
| UPPER(string) 和 LOWER(string) |  |
| RTRIM(string) 和 LTRIM(string) | 删除输入字符串中的尾随空格或前导空格 |

```
SELECT supplierid,
  RIGHT(REPLICATE('0',9) + cast(supplierid AS VARCHAR(10), 10) AS strsupplierid)
FROM Production.Suppliers;
```

LIKE谓词
|| 通配符 || 说明 ||
| % | 任意长度的字符串，包括空字符串 |
| _ | 任意单个字符 |
| [<字符列>] | 匹配指定字符中的一个字符 |
| [<字符>-<字符>] | 指定范围内的一个字符 |
| [^<>] | 不属于.... |
| ESCAPE | 转义字符 |

日期和时间数据
LANGUAGE / DATEFORMAT 输入值的解释方式
优先使用语言无关的格式，类似'YYYYMMDD'
CONVERT 修改语言样式

sql server 联机丛书



!=, !>, !< 非标准运算符，建议少用