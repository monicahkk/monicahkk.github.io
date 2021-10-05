目录：
1 <a href="#1">Series</a>   
    1.1 <a href="#1.1">数据类型</a>   
    1.2 <a href="#1.2">与多维数组类似</a>   
    1.3 <a href="#1.3">与字典类似</a>  
    1.4 <a href="#1.4">矢量操作与对齐 Series 标签</a>  
    1.5 <a href="#1.5">名称属性</a>  
2 <a href="#2">DataFrame</a>    
    构造函数   
    属性和数据   
    2.1 生成 DataFrame    
        2.1.1 <a href="#2.1.1">用 Series 字典或字典生成 DataFrame</a>   
        2.1.2 <a href="#2.1.2">用多维数组字典、列表字典生成 DataFrame</a>   
        2.1.3 <a href="#2.1.3">用结构多维数组或记录多维数组生成 DataFrame</a>   
        2.1.4 <a href="#2.1.4">用列表字典生成 DataFrame</a>   
        2.1.5 <a href="#2.1.5">用元组字典生成 DataFrame</a>   
        2.1.6 <a href="#2.1.6">用 Series 创建 DataFrame</a>   
    2.2 <a href="#2.2">缺失数据</a>      
    2.3 <a href="#2.3">备选构建器</a>   
    2.3.1 from_dict    
    2.3.2 <a href="#2.3.2">用方法链分配新列</a>   
    2.4 <a href="#2.4">索引 / 选择</a>   
    2.4.1 从新索引&选取&标签操作    
    2.5 <a href="#2.5">数据对齐和运算</a>   
    2.6 <a href="#2.6">转置</a>   
    2.7 <a href="#2.7">DataFrame 应用 NumPy 函数</a>   
    2.8 <a href="#2.8">控制台显示</a>   


```python
import pandas as pd 
pd.__version__
```




    '1.1.0'




```python
import numpy as np 
np.__version__
```




    '1.19.1'



# 数据结构
数据对齐是内在的: 除非显式指定，Pandas 不会断开标签和数据之间的连接

## <a name="1">Series</a>
- Series 是带标签的一维数组，可存储整数、浮点数、字符串、Python 对象等类型的数据
- 调用 pd.Series 函数即可创建 Series
- 轴标签统称为索引，Pandas 的索引值可以重复,不支持重复索引值的操作会触发异常


```python
data = ([0,1,2,3,4,5],1,3.14,'hello world!')
index = ['a','b','c','c']

s = pd.Series(data, index=index)
s
```




    a    [0, 1, 2, 3, 4, 5]
    b                     1
    c                  3.14
    c          hello world!
    dtype: object



### <a name="1.1">数据类型</a>:
- `data` 支持以下数据类型:
    - Python 字典
    - 多维数组
    - 标量值
    
- `index` 是轴标签列表。不同数据可分为以下几种情况
    - 多维数组: 
        - `index` 长度必须与 `data` 长度一致;
        - 没有指定 `index` 参数时，创建数值型索引，即 `[0, ..., len(data) - 1]`
    - 字典:
        - 未设置 `index` 参数时，如果 `Python 版本 >= 3.6` 且 `Pandas 版本 >= 0.23`，`Series` 按字典的插入顺序排序索引;
        - 如果设置了 `index` 参数，则按索引标签提取 `data` 里对应的值
    - 标量值:
        - `data` 是标量值时，必须提供索引。
        - `Series` 按索引长度重复该标量值


```python
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s.index
```




    Index(['a', 'b', 'c', 'd', 'e'], dtype='object')




```python
# 没有指定 `index` 参数
pd.Series(np.random.randn(5))
```




    0   -0.419614
    1    0.050753
    2   -0.618089
    3    0.885981
    4    1.380941
    dtype: float64




```python
# 未设置 index 参数时
d = {'b': 1, 'a': 0, 'c': 2}
pd.Series(d)
```




    b    1
    a    0
    c    2
    dtype: int64




```python
# 未设置 index 参数时
d = {'a': 0., 'b': 1., 'c': 2.}
pd.Series(d)
```




    a    0.0
    b    1.0
    c    2.0
    dtype: float64




```python
# 设置了 index 参数，则按索引标签提取 data 里对应的值
pd.Series(d, index=['b', 'c', 'd', 'a'])
```




    b    1.0
    c    2.0
    d    NaN
    a    0.0
    dtype: float64




```python
# data 是标量值时，必须提供索引
pd.Series(5., index=['a', 'b', 'c', 'd', 'e'])
```




    a    5.0
    b    5.0
    c    5.0
    d    5.0
    e    5.0
    dtype: float64




```python
# Series 按索引长度重复该标量值
pd.Series(5.)
```




    0    5.0
    dtype: float64



## <a name="1.2">与多维数组类似</a>
- 支持大多数 NumPy 函数，还支持索引切片  
- 支持 dtype  

Series 的数据类型一般是 NumPy 数据类型。不过，Pandas 和第三方库在一些方面扩展了 NumPy 类型系统，即**扩展数据类型**   

Series.array 用于提取 Series 数组,Series.array 一般是扩展数组。   

简单说，扩展数组是把 N 个 numpy.ndarray 包在一起的打包器; Series 只是类似于多维数组，提取真正的多维数组，要用 Series.to_numpy()


```python
s[0]
```




    0.983211233244982




```python
# 索引切片
s[:3]
```




    a    0.983211
    b    1.734580
    c   -0.832023
    dtype: float64




```python
s[s > s.median()]
```




    b    1.734580
    d    1.362331
    dtype: float64




```python
s[[4, 3, 1]]
```




    e   -0.297005
    d    1.362331
    b    1.734580
    dtype: float64




```python
np.exp(s)
```




    a    2.673026
    b    5.666547
    c    0.435168
    d    3.905285
    e    0.743040
    dtype: float64




```python
# 扩展数组
s.array
```




    <PandasArray>
    [   0.983211233244982,   1.7345799618271345,  -0.8320234693804991,
       1.3623306906358081, -0.29700497995670966]
    Length: 5, dtype: float64




```python
# 真正的多维数组
s.to_numpy()
```




    array([ 0.98321123,  1.73457996, -0.83202347,  1.36233069, -0.29700498])



## <a name="1.3">与字典类似</a>
- Series 类似固定大小的字典，可以用索引标签提取值或设置值
- 引用 Series 里没有的标签会触发异常
- get 方法可以提取 Series 里没有的标签，返回 None 或指定默认值


```python
# 用索引标签提取值
s['a']
```




    0.983211233244982




```python
# 用索引标签设置值
s['e'] = 12
print('e' in s)
print('f' in s)

# 引用 Series 里没有的标签会触发异常
# s['f']
```

    True
    False
    


```python
# 提取 Series 里没有的标签，返回 None 或指定默认值
print(s.get('f'))
print(s.get('f', np.nan))
```

    None
    nan
    

## <a name="1.4">矢量操作与对齐 Series 标签</a>
- Series 和多维数组的主要区别在于， Series 之间的操作会自动基于标签对齐数据。因此，不用顾及执行计算操作的 Series 是否有相同的标签
- 操作未对齐索引的 Series， 其计算结果是所有涉及索引的并集。如果在 Series 里找不到标签，运算结果标记为 NaN，即缺失值。编写无需显式对齐数据的代码，给交互数据分析和研究提供了巨大的自由度和灵活性。Pandas 数据结构集成的数据对齐功能，是 Pandas 区别于大多数标签型数据处理工具的重要特性
- 也可以用**dropna** 函数清除含有缺失值的标签


```python
s[1:] + s[:-1]
```




    a         NaN
    b    3.469160
    c   -1.664047
    d    2.724661
    e         NaN
    dtype: float64



## <a name="1.5">名称属性</a>
- Series 支持 name 属性：一般情况下，Series 自动分配 name，特别是提取一维 DataFrame 切片时
- pandas.Series.rename() 方法用于重命名 Series


```python
s = pd.Series(np.random.randn(5), name='something')
s
```




    0    0.255920
    1    0.661741
    2   -0.267671
    3   -0.108744
    4   -1.009123
    Name: something, dtype: float64




```python
s.name
```




    'something'



# <a name="2">DataFrame</a>
DataFrame 是由多种类型的列构成的二维标签数据结构，类似于 Excel 、SQL 表，或 Series 对象构成的字典

- DataFrame 是最常用的 Pandas 对象，与 Series 一样，DataFrame 支持多种类型的输入数据：
    - 一维 ndarray、列表、字典、Series 字典
    - 二维 numpy.ndarray
    - 结构多维数组或记录多维数组
    - Series
    - DataFrame
 
- 除了数据，还可以有选择地传递 index（行标签）和 columns（列标签）参数。传递了索引或列，就可以确保生成的 DataFrame 里包含索引或列。Series 字典加上指定索引时，会丢弃与传递的索引不匹配的所有数据。

- 没有传递轴标签时，按常规依据输入数据进行构建

### 构造函数

| 方法                                           | 描述       |
| :--------------------------------------------- | :--------- |
| DataFrame([data, index, columns, dtype, copy]) | 构造数据框 |

### 属性和数据

| 方法                                        | 描述                                                         |
| :------------------------------------------ | :----------------------------------------------------------- |
| Axes                                        | index: row labels；columns: column labels                    |
| DataFrame.as_matrix([columns])              | 转换为矩阵                                                   |
| DataFrame.dtypes                            | 返回数据的类型                                               |
| DataFrame.ftypes                            | Return the ftypes (indication of sparse/dense and dtype) in this object. |
| DataFrame.get_dtype_counts()                | 返回数据框数据类型的个数                                     |
| DataFrame.get_ftype_counts()                | Return the counts of ftypes in this object.                  |
| DataFrame.select_dtypes([include, exclude]) | 根据数据类型选取子数据框                                     |
| DataFrame.values                            | Numpy的展示方式                                              |
| DataFrame.axes                              | 返回横纵坐标的标签名                                         |
| DataFrame.ndim                              | 返回数据框的纬度                                             |
| DataFrame.size                              | 返回数据框元素的个数                                         |
| DataFrame.shape                             | 返回数据框的形状                                             |
| DataFrame.memory_usage([index, deep])       | Memory usage of DataFrame columns.                           |

## <a name="2.1.1">用 Series 字典或字典生成 DataFrame</a>
- 生成的索引是每个 Series 索引的并集。先把嵌套字典转换为 Series
- 如果没有指定列，DataFrame 的列就是字典键的有序列表


```python
d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(d, index=['d', 'b', 'a'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d</th>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## <a name="2.1.2">用多维数组字典、列表字典生成 DataFrame</a>
- 多维数组的长度必须相同。如果传递了索引参数，index 的长度必须与数组一致
- 如果没有传递索引参数，生成的结果是 range(n)，n 为数组长度


```python
d = {'one': [1., 2., 3., 4.], 'two': [4., 3., 2., 1.]}
pd.DataFrame(d)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(d, index=['a', 'b', 'c', 'd'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>4.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## <a name="2.1.3">用结构多维数组或记录多维数组生成 DataFrame</a>
- 本例与数组字典的操作方式相同
- DataFrame 的运作方式与 NumPy 二维数组不同


```python
data = np.zeros((2, ), dtype=[('A', 'i4'), ('B', 'f4'), ('C', 'a10')])
# data[:] = [(1, 2., 'Hello'), (2, 3., "World")]
# pd.DataFrame(data)
data
```




    array([(0, 0., b''), (0, 0., b'')],
          dtype=[('A', '<i4'), ('B', '<f4'), ('C', 'S10')])




```python
pd.DataFrame(data, index=['first', 'second'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>first</th>
      <td>1</td>
      <td>2.0</td>
      <td>b'Hello'</td>
    </tr>
    <tr>
      <th>second</th>
      <td>2</td>
      <td>3.0</td>
      <td>b'World'</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(data, columns=['C', 'A', 'B'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b'Hello'</td>
      <td>1</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b'World'</td>
      <td>2</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



## <a name="2.1.4">用列表字典生成 DataFrame</a>


```python
data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
pd.DataFrame(data2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(data2, index=['first', 'second'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>first</th>
      <td>1</td>
      <td>2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>second</th>
      <td>5</td>
      <td>10</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(data2, columns=['a', 'b'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



## <a name="2.1.5">用元组字典生成 DataFrame</a>
- 元组字典可以自动创建多层索引 DataFrame


```python
pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
               ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
               ('a', 'c'): {('A', 'B'): 5, ('A', 'C'): 6},
               ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
               ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">a</th>
      <th colspan="2" halign="left">b</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>b</th>
      <th>a</th>
      <th>c</th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">A</th>
      <th>B</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>D</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



## <a name="2.1.6">用 Series 创建 DataFrame</a>
- 生成的 DataFrame 继承了输入的 Series 的索引，如果没有指定列名，默认列名是输入 Series 的名称

## <a name="2.2">缺失数据</a>
- DataFrame 里的缺失值用 np.nan 表示。DataFrame 构建器以 numpy.MaskedArray 为参数时 ，被屏蔽的条目为缺失数据

| 方法                                       | 描述                                                         |
| :----------------------------------------- | :----------------------------------------------------------- |
| DataFrame.dropna([axis, how, thresh, …])   | Return object with labels on given axis omitted where alternately any |
| DataFrame.fillna([value, method, axis, …]) | 填充空值                                                     |
| DataFrame.replace([to_replace, value, …])  | Replace values given in ‘to_replace’ with ‘value’.           |


```python

```

## <a name="2.3">备选构建器</a>
### 转换为其他格式

| 方法                                              | 描述                                                         |
| :------------------------------------------------ | :----------------------------------------------------------- |
| DataFrame.from_csv(path[, header, sep, …])        | Read CSV file (DEPRECATED, please use pandas.read_csv() instead). |
| DataFrame.from_dict(data[, orient, dtype])        | Construct DataFrame from dict of array-like or dicts         |
| DataFrame.from_items(items[, columns, orient])    | Convert (key, value) pairs to DataFrame.                     |
| DataFrame.from_records(data[, index, …])          | Convert structured or record ndarray to DataFrame            |
| DataFrame.info([verbose, buf, max_cols, …])       | Concise summary of a DataFrame.                              |
| DataFrame.to_pickle(path[, compression, …])       | Pickle (serialize) object to input file path.                |
| DataFrame.to_csv([path_or_buf, sep, na_rep, …])   | Write DataFrame to a comma-separated values (csv) file       |
| DataFrame.to_hdf(path_or_buf, key, **kwargs)      | Write the contained data to an HDF5 file using HDFStore.     |
| DataFrame.to_sql(name, con[, flavor, …])          | Write records stored in a DataFrame to a SQL database.       |
| DataFrame.to_dict([orient, into])                 | Convert DataFrame to dictionary.                             |
| DataFrame.to_excel(excel_writer[, …])             | Write DataFrame to an excel sheet                            |
| DataFrame.to_json([path_or_buf, orient, …])       | Convert the object to a JSON string.                         |
| DataFrame.to_html([buf, columns, col_space, …])   | Render a DataFrame as an HTML table.                         |
| DataFrame.to_feather(fname)                       | write out the binary feather-format for DataFrames           |
| DataFrame.to_latex([buf, columns, …])             | Render an object to a tabular environment table.             |
| DataFrame.to_stata(fname[, convert_dates, …])     | A class for writing Stata binary dta files from array-like objects |
| DataFrame.to_msgpack([path_or_buf, encoding])     | msgpack (serialize) object to input file path                |
| DataFrame.to_gbq(destination_table, project_id)   | Write a DataFrame to a Google BigQuery table.                |
| DataFrame.to_records([index, convert_datetime64]) | Convert DataFrame to record array.                           |
| DataFrame.to_sparse([fill_value, kind])           | Convert to SparseDataFrame                                   |
| DataFrame.to_dense()                              | Return dense representation of NDFrame (as opposed to sparse) |
| DataFrame.to_string([buf, columns, …])            | Render a DataFrame to a console-friendly tabular output.     |
| DataFrame.to_clipboard([excel, sep])              | Attempt to write text representation of object to the system clipboard This can be pasted into Excel, for example. |

### DataFrame.from_dict
- DataFrame.from_dict 接收字典组成的字典或数组序列字典，并生成 DataFrame
- 除了 orient 参数默认为 columns，本构建器的操作与 DataFrame 构建器类似
- 把 orient 参数设置为 'index'， 即可把字典的键作为行标签
- orient='index' 时，键是行标签。本例还传递了列名


```python
pd.DataFrame.from_dict(dict([('A', [1, 2, 3]), ('B', [4, 5, 6])]))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame.from_dict(dict([('A', [1, 2, 3]), ('B', [4, 5, 6])]),
                        orient='index', columns=['one', 'two', 'three'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>B</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



### DataFrame.from_records
- DataFrame.from_records 构建器支持元组列表或结构数据类型（dtype）的多维数组
- 本构建器与 DataFrame 构建器类似，只不过生成的 DataFrame 索引是结构数据类型指定的字段

## <a name="2.3.2">用方法链分配新列</a>
- 受 dplyr 的 mutate 启发，DataFrame 提供了 assign() 方法，可以利用现有的列创建新列
- 上例中，插入了一个预计算的值。还可以传递带参数的函数，在 assign 的 DataFrame 上求值
- assign 返回的都是数据副本，原 DataFrame 不变
- 未引用 DataFrame 时，传递可调用的，不是实际要插入的值。这种方式常见于在操作链中调用 assign 的操作
- assign 函数签名就是 `**kwargs`。键是新字段的列名，值为是插入值（例如，Series 或 NumPy 数组），或把 DataFrame 当做调用参数的函数。返回结果是插入新值的 DataFrame 副本


```python
# pd.read_csv('data/iris.data')
# (iris.assign(sepal_ratio=iris['SepalWidth'] / iris['SepalLength']).head())
# iris.assign(sepal_ratio=lambda x: (x['SepalWidth'] / x['SepalLength'])).head()
```


```python
dependent = pd.DataFrame({"A": [1, 1, 1]})
dependent.assign(A=lambda x: x["A"] + 1, B=lambda x: x["A"] + 2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



## 提取、添加、删除列
- DataFrame 就像带索引的 Series 字典，提取、设置、删除列的操作与字典类似
- 删除（del、pop）列的方式也与字典类似
- 标量值以广播的方式填充列
- 插入与 DataFrame 索引不同的 Series 时，以 DataFrame 的索引为准
- 可以插入原生多维数组，但长度必须与 DataFrame 索引长度一致
- 默认在 DataFrame 尾部插入列。insert 函数可以指定插入列的位置


```python
df['three'] = df['one'] * df['two']
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
del df['two']
```


```python
three = df.pop('three')
```


```python
df['foo'] = 'bar'
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>foo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>bar</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>bar</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['one_trunc'] = df['one'][:2]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>foo</th>
      <th>one_trunc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>bar</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>bar</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>bar</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>bar</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.insert(1, 'bar', df['one'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>bar</th>
      <th>foo</th>
      <th>one_trunc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>bar</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>bar</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>bar</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>bar</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## <a name="2.4">索引 / 选择</a>
索引基础用法如下：

|操作	|句法	|结果|
| :----: | :----: | :----: |
|选择列	|df[col]	|Series|
|用标签选择行	|df.loc[label]	|Series|
|用整数位置选择行	|df.iloc[loc]	|Series|
|行切片	|df[5:10]	|DataFrame|
|用布尔向量选择行	|df[bool_vec]	|DataFrame|

### 索引和迭代

| 方法                                            | 描述                                                         |
| :---------------------------------------------- | :----------------------------------------------------------- |
| DataFrame.head([n])                             | 返回前n行数据                                                |
| DataFrame.at                                    | 快速标签常量访问器                                           |
| DataFrame.iat                                   | 快速整型常量访问器                                           |
| DataFrame.loc                                   | 标签定位                                                     |
| DataFrame.iloc                                  | 整型定位                                                     |
| DataFrame.insert(loc, column, value[, …])       | 在特殊地点插入行                                             |
| DataFrame.iter()                                | Iterate over infor axis                                      |
| DataFrame.iteritems()                           | 返回列名和序列的迭代器                                       |
| DataFrame.iterrows()                            | 返回索引和序列的迭代器                                       |
| DataFrame.itertuples([index, name])             | Iterate over DataFrame rows as namedtuples, with index value as first element of the tuple. |
| DataFrame.lookup(row_labels, col_labels)        | Label-based “fancy indexing” function for DataFrame.         |
| DataFrame.pop(item)                             | 返回删除的项目                                               |
| DataFrame.tail([n])                             | 返回最后n行                                                  |
| DataFrame.xs(key[, axis, level, drop_level])    | Returns a cross-section (row(s) or column(s)) from the Series/DataFrame. |
| DataFrame.isin(values)                          | 是否包含数据框中的元素                                       |
| DataFrame.where(cond[, other, inplace, …])      | 条件筛选                                                     |
| DataFrame.mask(cond[, other, inplace, axis, …]) | Return an object of same shape as self and whose corresponding entries are from self where cond is False and otherwise are from other. |
| DataFrame.query(expr[, inplace])                | Query the columns of a frame with a boolean expression.      |


```python
df.loc['b']
```




    one            2
    bar            2
    foo          bar
    one_trunc      2
    Name: b, dtype: object




```python
df.iloc[2]
```




    one      0.223227
    two      0.456285
    three   -1.827533
    Name: c, dtype: float64




```python
# 单条件筛选
# df[df['columnName'] > 'the value']
# 筛选one列的取值大于0的记录,但是只显示满足条件的one，three列的值
df[['one','three']][df['one']>0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.796113</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.790722</td>
      <td>-2.093928</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.223227</td>
      <td>-1.827533</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 使用isin函数根据特定值筛选记录：
# 筛选a值等于3或者5的记录
df[df.one.isin([3, 5])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# 多条件筛选
# 使用&筛选a列的取值大于0，b列的取值小于0的记录
df[(df['one'] > 0) & (df['two'] < 0)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.796113</td>
      <td>-0.338432</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 使用numpy的logical_and函数完成同样的功能
df[np.logical_and(df['one']> 0,df['two']<0)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.796113</td>
      <td>-0.338432</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 排除特定行
# 选出one列的值不等于1.7961132136721008的记录
ex_list = list(df['one'])
ex_list.remove(1.7961132136721008)
df[df.one.isin(ex_list)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>b</th>
      <td>0.790722</td>
      <td>0.337198</td>
      <td>-2.093928</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.223227</td>
      <td>0.456285</td>
      <td>-1.827533</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>-0.789518</td>
      <td>-0.034103</td>
    </tr>
  </tbody>
</table>
</div>



### 索引筛选   
loc：选取行/列标签索引数据   
iloc：选取行/列位置编号索引数据   
ix：通过行/列标签索引数据，也可以通过行/列位置索引数据   

### 从新索引&选取&标签操作

| 方法                                              | 描述                                                         |
| :------------------------------------------------ | :----------------------------------------------------------- |
| DataFrame.add_prefix(prefix)                      | 添加前缀                                                     |
| DataFrame.add_suffix(suffix)                      | 添加后缀                                                     |
| DataFrame.align(other[, join, axis, level, …])    | Align two object on their axes with the                      |
| DataFrame.drop(labels[, axis, level, …])          | 返回删除的列                                                 |
| DataFrame.drop_duplicates([subset, keep, …])      | Return DataFrame with duplicate rows removed, optionally only |
| DataFrame.duplicated([subset, keep])              | Return boolean Series denoting duplicate rows, optionally only |
| DataFrame.equals(other)                           | 两个数据框是否相同                                           |
| DataFrame.filter([items, like, regex, axis])      | 过滤特定的子数据框                                           |
| DataFrame.first(offset)                           | Convenience method for subsetting initial periods of time series data based on a date offset. |
| DataFrame.head([n])                               | 返回前n行                                                    |
| DataFrame.idxmax([axis, skipna])                  | Return index of first occurrence of maximum over requested axis. |
| DataFrame.idxmin([axis, skipna])                  | Return index of first occurrence of minimum over requested axis. |
| DataFrame.last(offset)                            | Convenience method for subsetting final periods of time series data based on a date offset. |
| DataFrame.reindex([index, columns])               | Conform DataFrame to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index. |
| DataFrame.reindex_axis(labels[, axis, …])         | Conform input object to new index with optional filling logic, placing NA/NaN in locations having no value in the previous index. |
| DataFrame.reindex_like(other[, method, …])        | Return an object with matching indices to myself.            |
| DataFrame.rename([index, columns])                | Alter axes input function or functions.                      |
| DataFrame.rename_axis(mapper[, axis, copy, …])    | Alter index and / or columns using input function or functions. |
| DataFrame.reset_index([level, drop, …])           | For DataFrame with multi-level index, return new DataFrame with labeling information in the columns under the index names, defaulting to ‘level_0’, ‘level_1’, etc. |
| DataFrame.sample([n, frac, replace, …])           | 返回随机抽样                                                 |
| DataFrame.select(crit[, axis])                    | Return data corresponding to axis labels matching criteria   |
| DataFrame.set_index(keys[, drop, append, …])      | Set the DataFrame index (row labels) using one or more existing columns. |
| DataFrame.tail([n])                               | 返回最后几行                                                 |
| DataFrame.take(indices[, axis, convert, is_copy]) | Analogous to ndarray.take                                    |
| DataFrame.truncate([before, after, axis, copy])   | Truncates a sorted NDFrame before and/or after some particular index value. |


```python
# 选取标签为one、three的列，选完还是DataFrame
df = df.loc[:, ['one','three']]
df = df.iloc[:, [0,2]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.796113</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.790722</td>
      <td>-2.093928</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.223227</td>
      <td>-1.827533</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>-0.034103</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 选取标签为one、three的列且只取前2行，选完还是DataFrame
df = df.loc[0:2, ['A','C']]
df = df.iloc[0:2, ['0','2']]
```


```python
# 选取行，选完还是DataFrame
df = df.loc[0:2, :]
df = df.iloc[0:2, :]
```


```python
# 在使用的时候需要统一，在行/列选择时同时出现索引和名称
df.ix[1:3,['a','b']]
```


```python
# at 根据指定行index及列label，快速定位DataFrame的元素，选择列时仅支持列名
df.at[3,'a']
```


```python
# iat 与at的功能相同，只使用索引参数
df.iat[3,0]
```




    nan




```python
# 切片
df[1:4] # 取行
df[['a','c']] # 取特定列
```


```python
# 取某一列里特定值对应的所有行
dt.loc[dt['columnName'] == 'the value']
```


```python
# 行名称
dfname._stat_axis.values.tolist() 
```


```python
# 列名称
dfname.columns.values.tolist() 
```

### 类型转换

| 方法                                    | 描述                   |
| :-------------------------------------- | :--------------------- |
| DataFrame.astype(dtype[, copy, errors]) | 转换数据类型           |
| DataFrame.copy([deep])                  | 复制数据框             |
| DataFrame.isnull()                      | 以布尔的方式返回空值   |
| DataFrame.notnull()                     | 以布尔的方式返回非空值 |


```python

```


```python

```

## <a name="2.5">数据对齐和运算</a>
- DataFrame 对象可以自动对齐**列与索引（行标签）**的数据
- 与上文一样，生成的结果是列和行标签的并集
- DataFrame 和 Series 之间执行操作时，默认操作是在 DataFrame 的列上对齐 Series 的索引，按行执行广播操作
- 时间序列是特例，DataFrame 索引包含日期时，按列广播
- 标量操作与其它数据结构一样
- 支持布尔运算符

### 二元运算

| 方法                                             | 描述                                                         |
| :----------------------------------------------- | :----------------------------------------------------------- |
| DataFrame.add(other[, axis, level, fill_value])  | 加法，元素指向                                               |
| DataFrame.sub(other[, axis, level, fill_value])  | 减法，元素指向                                               |
| DataFrame.mul(other[, axis, level, fill_value])  | 乘法，元素指向                                               |
| DataFrame.div(other[, axis, level, fill_value])  | 小数除法，元素指向                                           |
| DataFrame.truediv(other[, axis, level, …])       | 真除法，元素指向                                             |
| DataFrame.floordiv(other[, axis, level, …])      | 向下取整除法，元素指向                                       |
| DataFrame.mod(other[, axis, level, fill_value])  | 模运算，元素指向                                             |
| DataFrame.pow(other[, axis, level, fill_value])  | 幂运算，元素指向                                             |
| DataFrame.radd(other[, axis, level, fill_value]) | 右侧加法，元素指向                                           |
| DataFrame.rsub(other[, axis, level, fill_value]) | 右侧减法，元素指向                                           |
| DataFrame.rmul(other[, axis, level, fill_value]) | 右侧乘法，元素指向                                           |
| DataFrame.rdiv(other[, axis, level, fill_value]) | 右侧小数除法，元素指向                                       |
| DataFrame.rtruediv(other[, axis, level, …])      | 右侧真除法，元素指向                                         |
| DataFrame.rfloordiv(other[, axis, level, …])     | 右侧向下取整除法，元素指向                                   |
| DataFrame.rmod(other[, axis, level, fill_value]) | 右侧模运算，元素指向                                         |
| DataFrame.rpow(other[, axis, level, fill_value]) | 右侧幂运算，元素指向                                         |
| DataFrame.lt(other[, axis, level])               | 类似Array.lt                                                 |
| DataFrame.gt(other[, axis, level])               | 类似Array.gt                                                 |
| DataFrame.le(other[, axis, level])               | 类似Array.le                                                 |
| DataFrame.ge(other[, axis, level])               | 类似Array.ge                                                 |
| DataFrame.ne(other[, axis, level])               | 类似Array.ne                                                 |
| DataFrame.eq(other[, axis, level])               | 类似Array.eq                                                 |
| DataFrame.combine(other, func[, fill_value, …])  | Add two DataFrame objects and do not propagate NaN values, so if for a |
| DataFrame.combine_first(other)                   | Combine two DataFrame objects and default to non-null values in frame calling the method. |


```python
df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
df + df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.195046</td>
      <td>0.252772</td>
      <td>-0.052685</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.307507</td>
      <td>-1.452633</td>
      <td>0.500917</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.443433</td>
      <td>-0.625828</td>
      <td>2.789085</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.712633</td>
      <td>2.934665</td>
      <td>-0.525107</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.169796</td>
      <td>0.103454</td>
      <td>1.541764</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.585010</td>
      <td>3.322248</td>
      <td>0.633127</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-2.885445</td>
      <td>1.705544</td>
      <td>-2.447082</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df - df.iloc[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.519847</td>
      <td>-0.574759</td>
      <td>1.561635</td>
      <td>1.227317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.023369</td>
      <td>-1.732123</td>
      <td>2.564768</td>
      <td>0.732526</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.367272</td>
      <td>2.313049</td>
      <td>0.182911</td>
      <td>-0.250232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.728182</td>
      <td>-0.494344</td>
      <td>1.211211</td>
      <td>3.805475</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.212306</td>
      <td>0.099876</td>
      <td>-0.372364</td>
      <td>2.043350</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-3.062969</td>
      <td>0.164911</td>
      <td>0.015078</td>
      <td>1.358969</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.129824</td>
      <td>-3.372593</td>
      <td>1.524106</td>
      <td>2.182533</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.572312</td>
      <td>-0.224511</td>
      <td>1.233250</td>
      <td>2.033795</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.420670</td>
      <td>-0.159382</td>
      <td>0.975323</td>
      <td>3.306193</td>
    </tr>
  </tbody>
</table>
</div>




```python
index = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>-1.202377</td>
      <td>1.110094</td>
      <td>-0.674429</td>
    </tr>
    <tr>
      <th>2000-01-02</th>
      <td>1.107486</td>
      <td>0.036567</td>
      <td>0.785175</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>0.579559</td>
      <td>0.924574</td>
      <td>0.173773</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>0.266074</td>
      <td>1.134603</td>
      <td>0.104369</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>-2.013397</td>
      <td>1.073245</td>
      <td>1.209766</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>-0.338076</td>
      <td>0.972843</td>
      <td>0.372932</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>0.478091</td>
      <td>-0.197655</td>
      <td>-1.130669</td>
    </tr>
    <tr>
      <th>2000-01-08</th>
      <td>0.069406</td>
      <td>1.347692</td>
      <td>-0.275206</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(df['A'])
```




    pandas.core.series.Series




```python
df.sub(df['A'], axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>0.0</td>
      <td>2.312472</td>
      <td>0.527949</td>
    </tr>
    <tr>
      <th>2000-01-02</th>
      <td>0.0</td>
      <td>-1.070918</td>
      <td>-0.322311</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>0.0</td>
      <td>0.345014</td>
      <td>-0.405787</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>0.0</td>
      <td>0.868530</td>
      <td>-0.161705</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>0.0</td>
      <td>3.086642</td>
      <td>3.223163</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>0.0</td>
      <td>1.310920</td>
      <td>0.711008</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>0.0</td>
      <td>-0.675746</td>
      <td>-1.608760</td>
    </tr>
    <tr>
      <th>2000-01-08</th>
      <td>0.0</td>
      <td>1.278286</td>
      <td>-0.344612</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 1]}, dtype=bool)
df2 = pd.DataFrame({'a': [0, 1, 1], 'b': [1, 1, 0]}, dtype=bool)
df1 & df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 | df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 ^ df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## <a name="2.6">转置</a>
- 类似于多维数组，T 属性（即 transpose 函数）可以转置 DataFrame


```python
df[:5].T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2000-01-01</th>
      <th>2000-01-02</th>
      <th>2000-01-03</th>
      <th>2000-01-04</th>
      <th>2000-01-05</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>-1.202377</td>
      <td>1.107486</td>
      <td>0.579559</td>
      <td>0.266074</td>
      <td>-2.013397</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1.110094</td>
      <td>0.036567</td>
      <td>0.924574</td>
      <td>1.134603</td>
      <td>1.073245</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.674429</td>
      <td>0.785175</td>
      <td>0.173773</td>
      <td>0.104369</td>
      <td>1.209766</td>
    </tr>
  </tbody>
</table>
</div>



## <a name="2.7">DataFrame 应用 NumPy 函数</a>
- Series 与 DataFrame 可使用 log、exp、sqrt 等多种元素级 NumPy 通用函数（ufunc） ，假设 DataFrame 的数据都是数字


```python
np.exp(df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>0.300479</td>
      <td>3.034644</td>
      <td>0.509447</td>
    </tr>
    <tr>
      <th>2000-01-02</th>
      <td>3.026738</td>
      <td>1.037244</td>
      <td>2.192790</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>1.785252</td>
      <td>2.520793</td>
      <td>1.189785</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>1.304831</td>
      <td>3.109940</td>
      <td>1.110010</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>0.133534</td>
      <td>2.924856</td>
      <td>3.352700</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>0.713141</td>
      <td>2.645456</td>
      <td>1.451985</td>
    </tr>
    <tr>
      <th>2000-01-07</th>
      <td>1.612993</td>
      <td>0.820653</td>
      <td>0.322817</td>
    </tr>
    <tr>
      <th>2000-01-08</th>
      <td>1.071871</td>
      <td>3.848531</td>
      <td>0.759416</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.asarray(df)
```




    array([[-1.2023775 ,  1.11009412, -0.67442868],
           [ 1.10748558,  0.03656737,  0.78517471],
           [ 0.57955942,  0.92457357,  0.17377256],
           [ 0.2660736 ,  1.13460347,  0.1043686 ],
           [-2.01339686,  1.07324521,  1.20976603],
           [-0.33807608,  0.97284349,  0.37293157],
           [ 0.47809137, -0.19765452, -1.13066851],
           [ 0.06940581,  1.34769158, -0.27520613]])



- DataFrame 不是多维数组的替代品，它的索引语义和数据模型与多维数组都不同
- Series 应用 __array_ufunc__，支持 NumPy 通用函数
- 通用函数应用于 Series 的底层数组, 多个 Series 传递给 ufunc 时，会先进行对齐
- Pandas 可以自动对齐 ufunc 里的多个带标签输入数据


```python
ser = pd.Series([1, 2, 3, 4])
np.exp(ser)
```




    0     2.718282
    1     7.389056
    2    20.085537
    3    54.598150
    dtype: float64




```python
ser1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
ser2 = pd.Series([1, 3, 5], index=['b', 'a', 'c'])
ser1
```




    a    1
    b    2
    c    3
    dtype: int64




```python
ser2
```




    b    1
    a    3
    c    5
    dtype: int64




```python
np.remainder(ser1, ser2)
```




    a    1
    b    0
    c    3
    dtype: int64



- 一般来说，Pandas 提取两个索引的并集，不重叠的值用缺失值填充
- 对 Series 和 Index 应用二进制 ufunc 时，优先执行 Series，并返回的结果也是 Series
- NumPy 通用函数可以安全地应用于非多维数组支持的 Series，例如，SparseArray（参见稀疏计算）。如有可能，应用 ufunc 而不把基础数据转换为多维数组


```python
ser3 = pd.Series([2, 4, 6], index=['b', 'c', 'd'])
np.remainder(ser1, ser3)
```




    a    NaN
    b    0.0
    c    3.0
    d    NaN
    dtype: float64




```python
ser = pd.Series([1, 2, 3])
idx = pd.Index([4, 5, 6])
np.maximum(ser, idx)
```




    0    4
    1    5
    2    6
    dtype: int64



## <a name="2.8">控制台显示</a>
- 控制台显示大型 DataFrame 时，会根据空间调整显示大小。info()函数可以查看 DataFrame 的信息摘要
- 尽管 to_string 有时不匹配控制台的宽度，但还是可以用 to_string 以表格形式返回 DataFrame 的字符串表示形式
- 默认情况下，过宽的 DataFrame 会跨多行输出
- display.width 选项可以更改单行输出的宽度
- 可以用 display.max_colwidth 调整最大列宽
- expand_frame_repr 选项可以禁用此功能，在一个区块里输出整个表格


```python
np.random.randn(3, 12)
```




    array([[ 0.61240027,  0.07599991,  0.92227016,  0.85256123,  0.40214876,
            -1.20678027, -0.27541872,  1.06458855,  0.21832131,  0.98155055,
            -2.71009908, -1.81023123],
           [-0.56160208, -1.34438518,  1.21684708,  0.2710761 , -1.03446883,
            -0.83135799, -0.9282529 ,  0.30678431, -0.47309092,  0.06352417,
             1.04565417, -1.5714487 ],
           [-0.07299268, -0.25835849, -1.24838477, -1.03283659,  0.07080225,
            -0.22465254, -0.14007428, -1.37242264,  0.30803208,  0.23975675,
            -0.62819408, -0.56046381]])




```python
pd.set_option('display.width', 30)  # 默认值为 80
pd.DataFrame(np.random.randn(3, 12))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.668773</td>
      <td>-0.161357</td>
      <td>-1.329318</td>
      <td>2.805218</td>
      <td>-0.189457</td>
      <td>-0.218103</td>
      <td>-0.696182</td>
      <td>-1.489721</td>
      <td>0.062632</td>
      <td>-1.374856</td>
      <td>0.154923</td>
      <td>-0.481592</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.666330</td>
      <td>1.177886</td>
      <td>0.645667</td>
      <td>-0.204775</td>
      <td>-0.388108</td>
      <td>0.171606</td>
      <td>0.875074</td>
      <td>0.696638</td>
      <td>-0.627951</td>
      <td>-1.252995</td>
      <td>0.367286</td>
      <td>0.787507</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.212327</td>
      <td>-1.511281</td>
      <td>-0.113940</td>
      <td>1.187196</td>
      <td>-1.228607</td>
      <td>1.931129</td>
      <td>0.905883</td>
      <td>-1.764774</td>
      <td>1.061841</td>
      <td>-1.851094</td>
      <td>-0.850108</td>
      <td>0.390638</td>
    </tr>
  </tbody>
</table>
</div>



## DataFrame 列属性访问和 IPython 代码补全
- DataFrame 列标签是有效的 Python 变量名时，可以像属性一样访问该列


```python

```


```python
index = pd.date_range('1/1/2000', periods=8)
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
df = pd.DataFrame(np.random.randn(8, 3), index=index,columns=['A', 'B', 'C'])
```

## Head 与 Tail
- head() 与 tail() 用于快速预览 Series 与 DataFrame，默认显示 5 条数据，也可以指定显示数据的数量


```python
long_series = pd.Series(np.random.randn(1000))
long_series.head()
```




    0    0.516055
    1    1.127936
    2    0.793643
    3    1.266561
    4    0.264052
    dtype: float64




```python
long_series.tail(3)
```




    997   -0.901060
    998    1.780454
    999    1.279255
    dtype: float64



## 属性与底层数据
Pandas 可以通过多个属性访问元数据：
- shape:
    - 输出对象的轴维度，与 ndarray 一致
- 轴标签
    - Series: Index (仅有此轴)
    - DataFrame: Index (行) 与列
    
注意： 为属性赋值是安全的！

- .array 属性用于提取 Index 或 Series 里的数据
- 提取 NumPy 数组，用 to_numpy() 或 numpy.asarray()
    - to_numpy() 可以控制 numpy.ndarray 生成的数据类型
    - NumPy 未提供时区信息的 datetime 数据类型，Pandas 则提供了两种表现形式
        1. 一种是带 Timestamp 的 numpy.ndarray，提供了正确的 tz 信息
        2. 一种是 datetime64[ns]，这也是一种 numpy.ndarray，值被转换为 UTC，但去掉了时区信息  
        3. 时区信息可以用 dtype=object 保存，或用 dtype='datetime64[ns]' 去除


```python
df[:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-01</th>
      <td>-1.161885</td>
      <td>-0.635347</td>
      <td>0.834364</td>
    </tr>
    <tr>
      <th>2000-01-02</th>
      <td>-0.372137</td>
      <td>-1.018353</td>
      <td>2.103763</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = [x.lower() for x in df.columns]
```


```python
s.array
```




    <PandasArray>
    [  2.1116101023790135,
     -0.09210630424433626,
      0.37010250417134893,
      0.36285682543462205,
       0.8313819630208876]
    Length: 5, dtype: float64




```python
s.index.array
```




    <PandasArray>
    ['a', 'b', 'c', 'd', 'e']
    Length: 5, dtype: object




```python
s.to_numpy()
```




    array([ 2.1116101 , -0.0921063 ,  0.3701025 ,  0.36285683,  0.83138196])




```python
np.asarray(s)
```




    array([ 2.1116101 , -0.0921063 ,  0.3701025 ,  0.36285683,  0.83138196])




```python
ser = pd.Series(pd.date_range('2000', periods=2, tz="CET"))
ser.to_numpy(dtype=object)
```




    array([Timestamp('2000-01-01 00:00:00+0100', tz='CET', freq='D'),
           Timestamp('2000-01-02 00:00:00+0100', tz='CET', freq='D')],
          dtype=object)




```python
ser.to_numpy(dtype="datetime64[ns]")
```




    array(['1999-12-31T23:00:00.000000000', '2000-01-01T23:00:00.000000000'],
          dtype='datetime64[ns]')



## 加速操作
- 借助 numexpr 与 bottleneck 支持库，Pandas 可以加速特定类型的二进制数值与布尔操作
- 处理大型数据集时，这两个支持库特别有用，加速效果也非常明显。 
    - numexpr 使用智能分块、缓存与多核技术。
    - bottleneck 是一组专属 cython 例程，处理含 nans 值的数组时，特别快


```python
pd.set_option('compute.use_bottleneck', False)
pd.set_option('compute.use_numexpr', False)
```

## 二进制操作
Pandas 数据结构之间执行二进制操作，要注意下列两个关键点：

- 多维（DataFrame）与低维（Series）对象之间的广播机制；
- 计算中的缺失值处理。
  

这两个问题可以同时处理，但下面先介绍怎么分开处理。

### 匹配/广播机制
DataFrame 支持 add()、sub()、mul()、div() 及 radd()、rsub() 等方法执行二进制操作。  
广播机制重点关注输入的 Series。通过 axis 关键字，匹配 index 或 columns 即可调用这些函数


```python
df = pd.DataFrame({
    'one': pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
    'two': pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']),
    'three': pd.Series(np.random.randn(3), index=['b', 'c', 'd'])})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.796113</td>
      <td>-0.338432</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.790722</td>
      <td>0.337198</td>
      <td>-2.093928</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.223227</td>
      <td>0.456285</td>
      <td>-1.827533</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>-0.789518</td>
      <td>-0.034103</td>
    </tr>
  </tbody>
</table>
</div>




```python
row = df.iloc[1]
column = df['two']
```


```python
df.sub(row, axis='columns')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.005391</td>
      <td>-0.675630</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-0.567496</td>
      <td>0.119087</td>
      <td>0.266395</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>-1.126717</td>
      <td>2.059825</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sub(row, axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.005391</td>
      <td>-0.675630</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-0.567496</td>
      <td>0.119087</td>
      <td>0.266395</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>-1.126717</td>
      <td>2.059825</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sub(column, axis='index')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.134545</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.453524</td>
      <td>0.0</td>
      <td>-2.431126</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-0.233059</td>
      <td>0.0</td>
      <td>-2.283818</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.755416</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sub(column, axis=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.134545</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.453524</td>
      <td>0.0</td>
      <td>-2.431126</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-0.233059</td>
      <td>0.0</td>
      <td>-2.283818</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.755416</td>
    </tr>
  </tbody>
</table>
</div>



还可以用 Series 对齐多层索引 DataFrame 的某一层级


```python
dfmi = df.copy()
dfmi.index = pd.MultiIndex.from_tuples([(1, 'a'), (1, 'b'),(1, 'c'), (2, 'a')],names=['first', 'second'])
dfmi.sub(column, axis=0, level='second')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
    <tr>
      <th>first</th>
      <th>second</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>a</th>
      <td>2.134545</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.453524</td>
      <td>0.000000</td>
      <td>-2.431126</td>
    </tr>
    <tr>
      <th>c</th>
      <td>-0.233059</td>
      <td>0.000000</td>
      <td>-2.283818</td>
    </tr>
    <tr>
      <th>2</th>
      <th>a</th>
      <td>NaN</td>
      <td>-0.451086</td>
      <td>0.304329</td>
    </tr>
  </tbody>
</table>
</div>



Series 与 Index 还支持 divmod() 内置函数，该函数同时执行向下取整除与模运算，返回两个与左侧类型相同的元组


```python
s = pd.Series(np.arange(10))
s
```




    0    0
    1    1
    2    2
    3    3
    4    4
    5    5
    6    6
    7    7
    8    8
    9    9
    dtype: int32




```python
div, rem = divmod(s, 3)
div
```




    0    0
    1    0
    2    0
    3    1
    4    1
    5    1
    6    2
    7    2
    8    2
    9    3
    dtype: int32




```python
rem
```




    0    0
    1    1
    2    2
    3    0
    4    1
    5    2
    6    0
    7    1
    8    2
    9    0
    dtype: int32




```python
idx = pd.Index(np.arange(10))
idx
```




    Int64Index([0, 1, 2, 3, 4, 5,
                6, 7, 8, 9],
               dtype='int64')




```python
div, rem = divmod(idx, 3)
div
```




    Int64Index([0, 0, 0, 1, 1, 1,
                2, 2, 2, 3],
               dtype='int64')




```python
rem
```




    Int64Index([0, 1, 2, 0, 1, 2,
                0, 1, 2, 0],
               dtype='int64')



divmod() 还支持元素级运算


```python
div, rem = divmod(s, [2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
div
```




    0    0
    1    0
    2    0
    3    1
    4    1
    5    1
    6    1
    7    1
    8    1
    9    1
    dtype: int32




```python
rem
```




    0    0
    1    1
    2    2
    3    0
    4    0
    5    1
    6    1
    7    2
    8    2
    9    3
    dtype: int32



## 缺失值与填充缺失值操作
Series 与 DataFrame 的算数函数支持 fill_value 选项，即用指定值替换某个位置的缺失值。比如，两个 DataFrame 相加，除非两个 DataFrame 里同一个位置都有缺失值，其相加的和仍为 NaN，如果只有一个 DataFrame 里存在缺失值，则可以用 fill_value 指定一个值来替代 NaN，当然，也可以用 fillna 把 NaN 替换为想要的值


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.796113</td>
      <td>-0.338432</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.790722</td>
      <td>0.337198</td>
      <td>-2.093928</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.223227</td>
      <td>0.456285</td>
      <td>-1.827533</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>-0.789518</td>
      <td>-0.034103</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.DataFrame({ 'one': pd.Series(np.random.randn(3), index=['a', 'b', 'c']), 'two': pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd']), 'three': pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd'])})
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.444057</td>
      <td>-0.576390</td>
      <td>0.705153</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-1.356421</td>
      <td>-0.411172</td>
      <td>-0.731835</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.382577</td>
      <td>-0.982723</td>
      <td>-0.873004</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>0.905023</td>
      <td>0.860152</td>
    </tr>
  </tbody>
</table>
</div>




```python
df + df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.240170</td>
      <td>-0.914822</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.565699</td>
      <td>-0.073974</td>
      <td>-2.825762</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.605803</td>
      <td>-0.526438</td>
      <td>-2.700537</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>0.115504</td>
      <td>0.826049</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.add(df2, fill_value=0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>2.240170</td>
      <td>-0.914822</td>
      <td>0.705153</td>
    </tr>
    <tr>
      <th>b</th>
      <td>-0.565699</td>
      <td>-0.073974</td>
      <td>-2.825762</td>
    </tr>
    <tr>
      <th>c</th>
      <td>0.605803</td>
      <td>-0.526438</td>
      <td>-2.700537</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>0.115504</td>
      <td>0.826049</td>
    </tr>
  </tbody>
</table>
</div>



## 比较操作
与上一小节的算数运算类似，Series 与 DataFrame 还支持 eq、ne、lt、gt、le、ge 等二进制比较操作的方法：

|序号	|缩写	|英文	|中文|
| :-: | :-: | :-: | :-: |
|1	|eq	|equal to	|等于|
|2	|ne	|not equal to	|不等于|
|3	|lt	|less than	|小于|
|4	|gt	|greater than	|大于|
|5	|le	|less than or equal to	|小于等于|
|6	|ge	|greater than or equal to|大于等于|

这些操作生成一个与左侧输入对象类型相同的 Pandas 对象，即，dtype 为 bool。boolean 对象可用于索引操作


```python
df.gt(df2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>b</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>c</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>d</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.ne(df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>b</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>c</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>d</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## 布尔简化
empty、any()、all()、bool() 可以把数据汇总简化至单个布尔值


```python
(df > 0).all()
```




    one      False
    two      False
    three    False
    dtype: bool




```python
(df > 0).any()
```




    one       True
    two       True
    three    False
    dtype: bool



进一步把上面的结果简化为单个布尔值


```python
(df > 0).any().any()
```




    True



通过 empty 属性，可以验证 Pandas 对象是否为空


```python
df.empty
```




    False




```python
pd.DataFrame(columns=list('ABC')).empty
```




    True



用 bool() 方法验证单元素 pandas 对象的布尔值。


```python
pd.Series([True]).bool()
```




    True




```python
pd.DataFrame([[False]]).bool()
```




    False



## 比较对象是否等效
一般情况下，多种方式都能得出相同的结果。以 df + df 与 df * 2 为例。应用上一小节学到的知识，测试这两种计算方式的结果是否一致，一般人都会用 (df + df == df * 2).all()，不过，这个表达式的结果是 False


```python
df + df == df * 2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>b</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>c</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>d</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
(df + df == df * 2).all()
```




    one      False
    two       True
    three    False
    dtype: bool



注意：布尔型 DataFrame df + df == df * 2 中有 False 值！这是因为两个 NaN 值的比较结果为不等



```python
np.nan == np.nan
```




    False



为了验证数据是否等效，Series 与 DataFrame 等 N 维框架提供了 equals() 方法，用这个方法验证 NaN 值的结果为相等



```python
(df + df).equals(df * 2)
```




    True



注意：Series 与 DataFrame 索引的顺序必须一致，验证结果才能为 True


```python
df1 = pd.DataFrame({'col': ['foo', 0, np.nan]})
df2 = pd.DataFrame({'col': [np.nan, 0, 'foo']}, index=[2, 1, 0])

df1.equals(df2)
```




    False




```python
df1.equals(df2.sort_index())
```




    True




```python
df2.sort_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>





### 描述统计学

| 方法                                            | 描述                                                         |
| :---------------------------------------------- | :----------------------------------------------------------- |
| DataFrame.abs()                                 | 返回绝对值                                                   |
| DataFrame.all([axis, bool_only, skipna, level]) | Return whether all elements are True over requested axis     |
| DataFrame.any([axis, bool_only, skipna, level]) | Return whether any element is True over requested axis       |
| DataFrame.clip([lower, upper, axis])            | Trim values at input threshold(s).                           |
| DataFrame.clip_lower(threshold[, axis])         | Return copy of the input with values below given value(s) truncated. |
| DataFrame.clip_upper(threshold[, axis])         | Return copy of input with values above given value(s) truncated. |
| DataFrame.corr([method, min_periods])           | 返回本数据框成对列的相关性系数                               |
| DataFrame.corrwith(other[, axis, drop])         | 返回不同数据框的相关性                                       |
| DataFrame.count([axis, level, numeric_only])    | 返回非空元素的个数                                           |
| DataFrame.cov([min_periods])                    | 计算协方差                                                   |
| DataFrame.cummax([axis, skipna])                | Return cumulative max over requested axis.                   |
| DataFrame.cummin([axis, skipna])                | Return cumulative minimum over requested axis.               |
| DataFrame.cumprod([axis, skipna])               | 返回累积                                                     |
| DataFrame.cumsum([axis, skipna])                | 返回累和                                                     |
| DataFrame.describe([percentiles, include, …])   | 整体描述数据框                                               |
| DataFrame.diff([periods, axis])                 | 1st discrete difference of object                            |
| DataFrame.eval(expr[, inplace])                 | Evaluate an expression in the context of the calling DataFrame instance. |
| DataFrame.kurt([axis, skipna, level, …])        | 返回无偏峰度Fisher’s  (kurtosis of normal == 0.0).           |
| DataFrame.mad([axis, skipna, level])            | 返回偏差                                                     |
| DataFrame.max([axis, skipna, level, …])         | 返回最大值                                                   |
| DataFrame.mean([axis, skipna, level, …])        | 返回均值                                                     |
| DataFrame.median([axis, skipna, level, …])      | 返回中位数                                                   |
| DataFrame.min([axis, skipna, level, …])         | 返回最小值                                                   |
| DataFrame.mode([axis, numeric_only])            | 返回众数                                                     |
| DataFrame.pct_change([periods, fill_method, …]) | 返回百分比变化                                               |
| DataFrame.prod([axis, skipna, level, …])        | 返回连乘积                                                   |
| DataFrame.quantile([q, axis, numeric_only, …])  | 返回分位数                                                   |
| DataFrame.rank([axis, method, numeric_only, …]) | 返回数字的排序                                               |
| DataFrame.round([decimals])                     | Round a DataFrame to a variable number of decimal places.    |
| DataFrame.sem([axis, skipna, level, ddof, …])   | 返回无偏标准误                                               |
| DataFrame.skew([axis, skipna, level, …])        | 返回无偏偏度                                                 |
| DataFrame.sum([axis, skipna, level, …])         | 求和                                                         |
| DataFrame.std([axis, skipna, level, ddof, …])   | 返回标准误差                                                 |
| DataFrame.var([axis, skipna, level, ddof, …])   | 返回无偏误差                                                 |

### 函数应用&分组&窗口

| 方法                                           | 描述                                                         |
| :--------------------------------------------- | :----------------------------------------------------------- |
| DataFrame.apply(func[, axis, broadcast, …])    | 应用函数                                                     |
| DataFrame.applymap(func)                       | Apply a function to a DataFrame that is intended to operate elementwise, i.e. |
| DataFrame.aggregate(func[, axis])              | Aggregate using callable, string, dict, or list of string/callables |
| DataFrame.transform(func, *args, **kwargs)     | Call function producing a like-indexed NDFrame               |
| DataFrame.groupby([by, axis, level, …])        | 分组                                                         |
| DataFrame.rolling(window[, min_periods, …])    | 滚动窗口                                                     |
| DataFrame.expanding([min_periods, freq, …])    | 拓展窗口                                                     |
| DataFrame.ewm([com, span, halflife, alpha, …]) | 指数权重窗口                                                 |

### 从新定型&排序&转变形态

| 方法                                            | 描述                                                         |
| :---------------------------------------------- | :----------------------------------------------------------- |
| DataFrame.pivot([index, columns, values])       | Reshape data (produce a “pivot” table) based on column values. |
| DataFrame.reorder_levels(order[, axis])         | Rearrange index levels using input order.                    |
| DataFrame.sort_values(by[, axis, ascending, …]) | Sort by the values along either axis                         |
| DataFrame.sort_index([axis, level, …])          | Sort object by labels (along an axis)                        |
| DataFrame.nlargest(n, columns[, keep])          | Get the rows of a DataFrame sorted by the n largest values of columns. |
| DataFrame.nsmallest(n, columns[, keep])         | Get the rows of a DataFrame sorted by the n smallest values of columns. |
| DataFrame.swaplevel([i, j, axis])               | Swap levels i and j in a MultiIndex on a particular axis     |
| DataFrame.stack([level, dropna])                | Pivot a level of the (possibly hierarchical) column labels, returning a DataFrame (or Series in the case of an object with a single level of column labels) having a hierarchical index with a new inner-most level of row labels. |
| DataFrame.unstack([level, fill_value])          | Pivot a level of the (necessarily hierarchical) index labels, returning a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels. |
| DataFrame.melt([id_vars, value_vars, …])        | “Unpivots” a DataFrame from wide format to long format, optionally |
| DataFrame.T                                     | Transpose index and columns                                  |
| DataFrame.to_panel()                            | Transform long (stacked) format (DataFrame) into wide (3D, Panel) format. |
| DataFrame.to_xarray()                           | Return an xarray object from the pandas object.              |
| DataFrame.transpose(*args, **kwargs)            | Transpose index and columns                                  |

### Combining& joining&merging

| 方法                                          | 描述                                                         |
| :-------------------------------------------- | :----------------------------------------------------------- |
| DataFrame.append(other[, ignore_index, …])    | 追加数据                                                     |
| DataFrame.assign(**kwargs)                    | Assign new columns to a DataFrame, returning a new object (a copy) with all the original columns in addition to the new ones. |
| DataFrame.join(other[, on, how, lsuffix, …])  | Join columns with other DataFrame either on index or on a key column. |
| DataFrame.merge(right[, how, on, left_on, …]) | Merge DataFrame objects by performing a database-style join operation by columns or indexes. |
| DataFrame.update(other[, join, overwrite, …]) | Modify DataFrame in place using non-NA values from passed DataFrame. |

### 时间序列

| 方法                                            | 描述                                                         |
| :---------------------------------------------- | :----------------------------------------------------------- |
| DataFrame.asfreq(freq[, method, how, …])        | 将时间序列转换为特定的频次                                   |
| DataFrame.asof(where[, subset])                 | The last row without any NaN is taken (or the last row without |
| DataFrame.shift([periods, freq, axis])          | Shift index by desired number of periods with an optional time freq |
| DataFrame.first_valid_index()                   | Return label for first non-NA/null value                     |
| DataFrame.last_valid_index()                    | Return label for last non-NA/null value                      |
| DataFrame.resample(rule[, how, axis, …])        | Convenience method for frequency conversion and resampling of time series. |
| DataFrame.to_period([freq, axis, copy])         | Convert DataFrame from DatetimeIndex to PeriodIndex with desired |
| DataFrame.to_timestamp([freq, how, axis, copy]) | Cast to DatetimeIndex of timestamps, at beginning of period  |
| DataFrame.tz_convert(tz[, axis, level, copy])   | Convert tz-aware axis to target time zone.                   |
| DataFrame.tz_localize(tz[, axis, level, …])     | Localize tz-naive TimeSeries to target time zone.            |

### 作图

| 方法                                        | 描述                                                         |
| :------------------------------------------ | :----------------------------------------------------------- |
| DataFrame.plot([x, y, kind, ax, ….])        | DataFrame plotting accessor and method                       |
| DataFrame.plot.area([x, y])                 | 面积图Area plot                                              |
| DataFrame.plot.bar([x, y])                  | 垂直条形图Vertical bar plot                                  |
| DataFrame.plot.barh([x, y])                 | 水平条形图Horizontal bar plot                                |
| DataFrame.plot.box([by])                    | 箱图Boxplot                                                  |
| DataFrame.plot.density(**kwds)              | 核密度Kernel Density Estimate plot                           |
| DataFrame.plot.hexbin(x, y[, C, …])         | Hexbin plot                                                  |
| DataFrame.plot.hist([by, bins])             | 直方图Histogram                                              |
| DataFrame.plot.kde(**kwds)                  | 核密度Kernel Density Estimate plot                           |
| DataFrame.plot.line([x, y])                 | 线图Line plot                                                |
| DataFrame.plot.pie([y])                     | 饼图Pie chart                                                |
| DataFrame.plot.scatter(x, y[, s, c])        | 散点图Scatter plot                                           |
| DataFrame.boxplot([column, by, ax, …])      | Make a box plot from DataFrame column optionally grouped by some columns or |
| DataFrame.hist(data[, column, by, grid, …]) | Draw histogram of the DataFrame’s series using matplotlib / pylab. |



  参考文献：  

http://pandas.pydata.org/pandas-docs/stable/api.html#dataframe

本文参与[腾讯云自媒体分享计划](https://cloud.tencent.com/developer/support-plan)，欢迎正在阅读的你也加入，一起分享。
