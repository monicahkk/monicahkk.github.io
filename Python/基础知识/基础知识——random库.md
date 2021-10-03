# random模块

## random.random() 
获取一个0-1之间的随机浮点数
```python
import random
# random.random(x,y) 从0-1区间取出一个浮点数,不包括0和1，区间不可改print(random.random()) # 范围 0-1 浮点型数据

# 输出结果
0.02808849040764705
```

## random.uniform() 
获取自定义区间的一个浮点数
```python
# random.uniform(x,y) 从x-y区间取出一个浮点数，不包x,y
print(random.uniform(1,2)) # 可以指定区间 取浮点型数据

# 输出结果
1.6776777381637533
```

## random.randint() 
获取自定义区间的一个整数，有始有终
```python
# random.randint(x,y) 从x-y区间取出一个整数，包括x,y
print(random.randint(1,3)) # 有始有终 1-3
```

## random.randrange() 
获取自定义区间一个整数，有始无终
```python
# random.randrange(x,y) 从x-y区间取出一个整数，包括x,不包括y
print(random.randrange(1,3)) # 有始无终 1-3
```

## random.choice() 
从列表中随机取出一个
```python
# random.choic([]) 随机取出一个
print(random.choice(["ass",5,6,"b"])) # 用列表(也可以是元组)中取值，数字字符都行
```

## random.sample() 
从列表中取出指定个数
```python
#random.sample([],n) 从前面的列表中随机取出n各
print(random.sample(["hello",5,9,"world"],2)) # 从列表中随机取出2个

# 输出结果
['hello', 9]
```

## random.suffle() 
是列表顺序随机
```python
# 洗牌功能，使列表顺序随机
a =[1,2,3,4,5,6,7,8,9]
random.shuffle(a)
print(a)

# 输出结果
[9, 1, 5, 2, 3, 6, 4, 8, 7]
```

## 小案例，数字字母验证码

```python
# 小案例 验证码 4位验证码 包括数字，大小写字母

""" chr() 65 - 122 获取adcii 中的大小写字母"""
check_code =""
for i inrange(4):  
	current =random.randrange(0,4) # 自定义逻辑  
	# 字母  
	if i ==current:    
		tmp =chr(random.randint(65,122))  
	else:    
		tmp =str(random.randint(0,9))  
	check_code+=tmp
print(check_code)

# 输出结果
22B5
```

参考文献：
https://www.cnblogs.com/bert227/p/9323983.html