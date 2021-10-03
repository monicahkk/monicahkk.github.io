# csv

## 函数说明

### reader类
```python
csv.reader(csvfile, dialect='excel', **fmtparams)
```
- 返回一个 reader 对象，该对象将逐行遍历 csvfile
- csvfile 可以是任何对象，只要这个对象支持 [iterator](https://docs.python.org/zh-cn/3/glossary.html#term-iterator) 协议并在每次调用 __next__() 方法时都返回字符串，文件对象 和列表对象均适用
- 可选参数 dialect 是用于不同的 CSV 变种的特定参数组。
- 可选关键字参数 fmtparams 可以覆写当前变种格式中的单个格式设置

| Reader对象 | 解释 |
| :-: | :-: |
| __next__() | 返回 reader 的可迭代对象的下一行，它可以是一个列表（如果对象是由 reader() 返回）或字典（如果是一个 DictReader 实例），根据当前 Dialect 来解析。 通常你应当以 next(reader) 的形式来调用它。 |
| line_num | 迭代器已经读取了的行数。它与返回的记录数不同，因为记录可能跨越多行 |

| DictReader对象 | 解释 |
| :-: | :-: |
| fieldnames | 字段名称。如果在创建对象时未传入字段名称，则首次访问时或从文件中读取第一条记录时会初始化此属性。 |

```python
import csv
with open('some.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# 非csv
import csv
with open('passwd', newline='') as f:
    reader = csv.reader(f, delimiter=':', quoting=csv.QUOTE_NONE)
    for row in reader:
        print(row)
```

### writer类
- 对于 Writer 对象，行 必须是（一组可迭代的）字符串或数字。对于 DictWriter 对象，行 必须是一个字典，这个字典将字段名映射为字符串或数字（数字要先经过 str() 转换类型）

| Writer对象 | 解释 |
| :-: | :-: |
| writerow | 将 row 形参写入到 writer 的文件对象，根据当前 Dialect 进行格式化。 返回对下层文件对象的 write 方法的调用的返回值 |
| writerows | 将 `rows*`（即能迭代出多个上述 `*row` 对象的迭代器）中的所有元素写入 writer 的文件对象，并根据当前设置的变种进行格式化 |
| dialect | 变种描述，只读，供 writer 使用 |

| DictWriter对象 | 解释 |
| :-: | :-: |
| writeheader | 在 writer 的文件对象中，写入一行字段名称（字段名称在构造函数中指定），并根据当前设置的变种进行格式化 |

```python
import csv
with open('some.csv', 'w', newline='') as f:
    writer = csv.writer(f)    
    writer.writerows(someiterable)    # 写入多行
    
with open('filename.txt', 'w') as fw:
	for i in range(len(dataArray)):
	    fw.write(str(dataArray[i]))   # 写入一个矩阵
	    fw.write('\n')
	fw.close()

with open('filename.csv', 'w') as csvFile:
    csvWriter = csv.writer(csvFile)
    for datain datas:
        csvWriter.writerow(data)      # 每次写入一行
```

#### 变种格式

| 变种格式 | 说明 |
| :-: | :-: |
| delimiter | 分隔符，默认',' |
| doublequote | 控制出现在字段中的 `引号字符` 本身应如何被引出。<br>当该属性为 `True` 时，双写引号字符。<br> 如果该属性为 `False`，则在 `引号字符` 的前面放置 `转义符`。<br>默认值为 `True`。<br>在输出时，如果 `doublequote` 是 `False`，且 `转义符` 未指定，且在字段中发现 `引号字符` 时，会抛出 `Error` 异常。 |
| escapechar | 一个用于 writer 的单字符，用来在 quoting 设置为 QUOTE_NONE 的情况下转义 定界符，在 doublequote 设置为 False 的情况下转义 引号字符。在读取时，escapechar 去除了其后所跟字符的任何特殊含义。该属性默认为 None，表示禁用转义 |
| lineterminator | 放在 writer 产生的行的结尾，默认为 '\r\n'。 |
| quotechar | 一个单字符，用于包住含有特殊字符的字段，特殊字符如 定界符 或 引号字符 或换行符。默认为 '"' |
| quoting | 控制 writer 何时生成引号，以及 reader 何时识别引号。 |
| skipinitialspace | 如果为 True，则忽略 定界符 之后的空格 |
| strict | 如果为 True，则在输入错误的 CSV 时抛出 Error 异常 |

### 示例
#### 不需要单写close
```python
indata = open(from_file).read()
```

#### 修改编码
```python
import csv
with open('some.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
```

#### 注册一个新的变种
```python
import csv
csv.register_dialect('unixpwd', delimiter=':', quoting=csv.QUOTE_NONE)
with open('passwd', newline='') as f:
    reader = csv.reader(f, 'unixpwd')
```

#### 获取每一行
```python
import csv

with open('data.csv','r') as f:
    reader = csv.reader(f) 
    print(type(reader)) # 返回_csv.reader对象

    for row in reader: # 输出
        print(row)

# 不需要单写close
indata = open(from_file).read()
```
#### 获取某一行
```python
with open('data.csv','r') as f:
    reader = csv.reader(f)
    datalist = list(reader) 
    print(datalist[1])
```
#### 获取某一列
```python
with open('data.csv','r') as f:
    reader = csv.reader(f)
    for i in reader:
        print(i[0])
```
#### 写入取消空行
```python
# 取消空行
open('data.csv', 'w', newline='')
```

## 读 / 写的文件模式

| 模式 | 解释 | 
| :-: | :-: |
| 'r' | open for reading (default) |
| 'w' | open for writing, truncating the file first |
| 'x' | create a new file and open it for writing |
| 'a' | open for writing, appending to the end of the file if it exists |
| 'b' | binary mode |
| 't' | text mode (default) |
| '+' | open a disk file for updating (reading and writing) |
| 'U' | universal newline mode (deprecated) |

参考文献:
https://docs.python.org/zh-cn/3/library/csv.html

## 指针seek()

python中可以使用seek()移动文件指针到指定位置，然后读/写。

通常配合 r+ 、w+、a+ 模式，在此三种模式下，seek指针移动只能从头开始移动，即seek(x,0) 。

| 模式 | 默认 | 写方式 | 与seek()配合---写 | 与seek()配合---读 |
| -- | -- | -- | -- | -- |
| r+ | 文件指针在文件头部，即seek(0) | 覆盖 | f = open('test.txt','r+',encoding='utf-8')<br>f.seek(3,0)<br>f.write('aaa')<br># 移动文件指针到指定位置，再写 | f = open('test.txt','r+',encoding='utf-8')<br>f.seek(3,0)<br>f.read()<br># 移动文件指针到指定位置，读取后面的内容 |
| w+ | 文件指针在文件头部，即seek(0) | 清除 | f = open('test.txt','w+',encoding='utf-8')<br>f.seek(3,0)<br>f.write('aaa')<br># 清除文件内容，移动文件指针到指定位置，再写 | f = open('test.txt','w+',encoding='utf-8')<br>f.write('aaa')<br>f.seek(3,0)<br>f.read()<br># 清除文件内容写入，移动文件指针到指定位置，读取后面内容` |
| a+ | 文件指针在文件尾部，即seek(0,2) | 追加 | f = open('test.txt','a+',encoding='utf-8')<br>f.seek(3,0)<br>f.write('aaa')<br># 直接在文件末尾写入，seek移动指针不起作用 | 同 r+ |

seek(offset[,whence])：
- offset--偏移量，可以是负值，代表从后向前移动；
- whence--偏移相对位置，分别有：os.SEEK_SET（相对文件起始位置，也可用“0”表示）；os.SEEK_CUR（相对文件当前位置，也可用“1”表示）；os.SEEK_END（相对文件结尾位置，也可用“2”表示）。
```python
seek(x,0)    # 表示指针从开头位置移动到x位置 
seek(x,1)    # 表示指针从当前位置向后移动x个位置 
seek(-x,2)   # 表示指针从文件结尾向前移动x个位置
# 例：file.seek(-1,2)，文件指针从文件末尾向前移动一个字符，配合read相关方法/函数可读取该字符。
```
-----------------------

# xlwt

```python 3
# 通过xlwt 、openpyxl这个库
import xlwt
import openpyxl
'''
 data: 要保存的数据
 fields: 表头
 sheetname: 工作簿名称
 wbname: 文件名
'''
def savetoexcel(data, fields, sheetname, wbname):
    wb = openpyxl.load_workbook(filename=wbname)

    sheet = wb.active
    sheet.title = sheetname

    field = 1
    for field in range(1, len(fields) + 1):  # 写入表头
        _ = sheet.cell(row=1, column=field, value=str(fields[field - 1]))

    row1 = 1
    col1 = 0
    for row1 in range(2, len(data) + 2):  # 写入数据
        for col1 in range(1, len(data[row1 - 2]) + 1):
            _ = sheet.cell(row=row1, column=col1, value=str(data[row1 - 2][col1 - 1]))

    wb.save(filename=wbname)
    print(wbname + "保存成功")
    
if __name__ == '__main__':
	wb1 = openpyxl.Workbook()
	datas = [
	    ("Mike", "male", 24),
	    ("Lee", "male", 26),
	    ("Joy", "female", 22)
	]
	headers = ['name', 'gender', 'age']
	wb1.save('filename.xlsx')
	savetoexcel(datas, headers, '工作表1', 'filename.xlsx')

# 针对DataFrame格式，可通过to_excel进行保存
import pandas as pd
datas = [
    ("Mike", "male", 24),
    ("Lee", "male", 26),
    ("Joy", "female", 22)
]
datas = pd.DataFrame(datas)
headers = ['name', 'gender', 'age']
datas.to_excel('filename.xlsx',header=headers)
```