
```shell
[Environment]: :SetEnvironmentVariable("Path","$env:Path;C:\Python27",	"User""  配置环境变量
mkdir test  创建文件夹
type nul>test.go 创建文件
echo 'hello world' >test.go 写入文件
```

```PYTHON
python list functions

# Pydoc https://www.jianshu.com/p/bf2df7e433ec
# pydoc是python自带的一个文档生成工具，使用pydoc可以很方便的查看类和方法结构，pydoc模块可以从python代码中获取docstring，然后生成帮助信息
python -m pydoc input # 直接查看某个py文件的内容
python3 -m pydoc -p 1234 # 启动本地服务，在web上查看文档
python3 -m pydoc -w testpydoc # 生成html说明文档
python3 -m pydoc -k 关键词 # -k查找模块

pydoc open
pydoc file
pydoc os
pydoc sys
```

 [python字符串格式化](https://www.cnblogs.com/songdanlee/p/11105807.html)

- github.com
- launchpad.net
- gitorious.org
- sourceforge.net

### assert（断言）
```python
# 用于判断一个表达式，在表达式条件为 false 的时候触发异常
assert expression [, arguments]
# 等价于：
if not expression:
    raise AssertionError(arguments)
    
#例子
import sys
assert ('linux' in sys.platform), "该代码只能在 Linux 下执行"

# 接下来要执行的代码
```

### hash() 函数
```python
hash(object) 
# object -- 对象；可以应用于数字、字符串和对象，不能直接应用于 list、set、dictionary
# 所得的结果不仅和对象的内容有关，还和对象的 id()，也就是内存地址有关
# hash() 函数的对象字符不管有多长，返回的 hash 值都是固定长度的，也用于校验程序在传输过程中是否被第三方（木马）修改

# 例子
>>>hash('test')            # 字符串
2314058222102390712
>>> hash(1)                 # 数字
1
>>> hash(str([1,2,3]))      # 集合
1335416675971793195
>>> hash(str(sorted({'1':1}))) # 字典
7666464346782421378
>>>
```

### python中OrderedDict的使用
```python
# collections.OrderedDict - 使用OrderedDict会根据放入元素的先后顺序进行排序
# 如果其顺序不同那么Python也会把他们当做是两个不同的对象
d1 = collections.OrderedDict()
d1['a'] = 'A'
d1['b'] = 'B'
d1['c'] = 'C'
d1['1'] = '1'
d1['2'] = '2'
for k,v in d1.items():
    print k,v
```