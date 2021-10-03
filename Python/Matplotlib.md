https://mp.weixin.qq.com/s/eY1y-sbx5Dk-9cB1WxoliA

### 【python学习】-matplotlib图形设置（线宽、标签、颜色、图框、线类型、图例大小位置、图框大小及像素等）

https://blog.csdn.net/qq_40481843/article/details/106231257

```python 3
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Jupyter notebook绘图时，直接在python console里生成图像
%matplotlib inline
# 在Jupyter notebook中具体作用是 当调用matplotlib.pyplot的绘图函数plot()进行绘图的时候，或者生成一个figure画布的时候，可以直接在python console里面生成图像

# 双环图
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

size = 0.3
vals = np.array([[60., 32.], [37., 40.], [29., 10.]])

cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.set(aspect="equal", title='Pie plot with `ax.pie`')
plt.show()
```

[matplotlib.pyplot.colormaps色彩图cmap](https://matplotlib.org/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py)

![tab20b&tab20c.png](D:\MyDocuments\Typora\01-python\Matplotlib/tab20b&tab20c.png)

### 设置图形大小
```python
fig = plt.figure(figsize=(a, b), dpi=dpi)
# figsize 设置图形的大小，a 为图形的宽， b 为图形的高，单位为英寸
# dpi 为设置图形每英寸的点数
```