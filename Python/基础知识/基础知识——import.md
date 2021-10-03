# `_all_`

`__all__`，用于模块导入时限制，如：`from module import *`
此时被导入模块若定义了`__all__`属性，则只有`__all__`内指定的属性、方法、类可被导入；若没定义，则导入模块内的所有公有属性，方法和类。

## 没有定义`__all__`
```python
# bb.py
class A():
    def __init__(self,name,age):
        self.name=name
        self.age=age
class B():
    def __init__(self,name,id):
        self.name=name
        self.id=id
def fun():
    print "func() is run!"
def fun1():
    print "func1() is run!"

# test_bb.py
from bb import *
a=A('zhansan','18')
print a.name,a.age
b=B("lisi",1001)
print b.name,b.id
fun()
fun1()

# 运行结果：
zhansan 18
lisi 1001
func() is run!
func1() is run!

# 由于bb.py中没有定义__all__属性，所以导入了bb.py中所有的公有属性
```

## 定义`__all__`
```python
# bb.py
__all__=('A','func')
class A():
    def __init__(self,name,age):
        self.name=name
        self.age=age
class B():
    def __init__(self,name,id):
        self.name=name
        self.id=id
def func():
    print "func() is run!"
def func1():
    print "func1() is run!"
    
# test_bb.py
from bb import *
a = A('zhansan','18')
print a.name,a.age
func()

# b = B("lisi",1001)
# NameError: name 'B' is not defined

# func1()
# NameError: name 'func1' is not defined 　

# 运行结果：
zhansan 18
func() is run!

# 由于bb.py中使用了__all__=('A','func')，所以在别的模块导入该模块时，只能导入__all__中的变量、方法、类
```

## 不同作用域
`def [变量名]`: public方法
`def _[变量名]`: protected方法
`def __[变量名]`: private方法

```python
# bb.py
def func(): # 模块中的public方法
    print 'func() is run!'
def _func(): # 模块中的protected方法
    print '_func() is run!'
def __func(): # 模块中的private方法 
    print '__func() is run!'

# test_bb.py
from bb import *  
func()
# _func()
# __func()

# 运行结果：
func() is run!

# from bb import * 只能导入公有的属性、方法、类, 无法导入以单下划线开头（protected）或以双下划线开头(private)的属性、方法、类
# _func() #NameError: name '_func' is not defined
# __func() #NameError: name '__func' is not defined
```

```python
#bb.py
__all__=('func','__func','_A') # 放入__all__中所有属性均可导入，即使是以下划线开头
class _A():
    def __init__(self,name):
        self.name=name
def func():
    print "func() is run!"
def func1():
    print "func1() is run!"
def _func():
    print "_func() is run!"
def __func():
    print "__func() is run!"


#test_bb.py
from bb import *
func()
# func1() # func1不在__all__中，无法导入 NameError: name 'func1' is not defined 
# _func() # _func不在__all__中，无法导入 NameError: name '_func' is not defined
__func() # __func在__all__中，可以导入
a = _A('zhangsan') # _A在__all__中，可以导入
print a.name

# 运行结果：
func() is run!
__func() is run!
zhangsan
```

## 2种不同的import方法

```python
# bb.py
def func():
    print 'func() is run!'
def _func():
    print '_func() is run!'
def __func():
    print '__func() is run!'
#test_bb.py
from bb import func,_func,__func #可以通过这种方式导入public,protected,private
func()
_func()
__func()

# 运行结果：
func() is run!
_func() is run!
__func() is run!
```

```python
# bb.py
def func():
    print 'func() is run!'
def _func():
    print '_func() is run!'
def __func():
    print '__func() is run!'
 
#test_bb.py
import bb # 可以通过这种方式导入 public,protected,private
bb.func()
bb._func()
bb.__func()

# 运行结果：
func() is run!
_func() is run!
__func() is run!
```

参考文献：
https://www.cnblogs.com/wxlog/p/10566628.html