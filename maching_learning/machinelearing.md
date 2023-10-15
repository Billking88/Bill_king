# 泰坦尼克号生存率预测

## **主要步骤：**

1. **提出问题（Business Understanding ）**
2. **理解数据（Data Understanding）**
3. **数据清洗（Data Preparation ）**
4. **构建模型（Modeling）**
5. **评估模型（Evaluation）**
6. **实施方案（Deployment）**

### 一、准备工作

**1.数据来源：**

[Titanic - Machine Learning from Disaster | Kaggle](https://www.kaggle.com/c/titanic)

**2.参考资料：**

https://www.zhihu.com/question/23987009/answer/285179721

### 二、预测流程

1.**提出问题**：能否利用Kaggle上现有的数据，对泰坦尼克号上的乘客作出生存率的预测？

2.导入Python模块和下载好的数据集，train.csv（训练数据）和test.csv（测试数据）；

3.填补缺失值；

**4.提取特征值；**

5.构建和评估模型；

6.选取模型进行预测。

### 三、数据处理工作

#### **1.导入导入Python模块和下载好的数据集，train.csv（训练数据）和test.csv（测试数据）**

```python
import numpy as np
import pandas as pd
import os

rain = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print('训练数据集大小：',train.shape,  '测试数据集大小', test.shape)

#如果数据读取失败，在Python提示符下使用这行代码查看运行路径，将数据文件放入其中。
#os.getcwd()
```

输出结果：

```python
训练数据集大小： (891, 12) 测试数据集大小 (418, 11)
```



#### **2.了解数据的整体情况**



```python
#Python命令行下：
#full.head()  #查看数据
#full.describe()  #获取数据描述统计信息。
#full.info()  #查看每一列的数据类型和数据总数
```



**a.查看数据：**



```python
>>> full.head()
    Age Cabin Embarked     Fare  \
0  22.0   NaN        S   7.2500   
1  38.0   C85        C  71.2833   
2  26.0   NaN        S   7.9250   
3  35.0  C123        S  53.1000   
4  35.0   NaN        S   8.0500   

                                                Name  Parch  PassengerId  \
0                            Braund, Mr. Owen Harris      0            1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...      0            2   
2                             Heikkinen, Miss. Laina      0            3   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)      0            4   
4                           Allen, Mr. William Henry      0            5   

   Pclass     Sex  SibSp  Survived            Ticket  
0       3    male      1       0.0         A/5 21171  
1       1  female      1       1.0          PC 17599  
2       3  female      0       1.0  STON/O2. 3101282  
3       1  female      1       1.0            113803  
4       3    male      0       0.0            373450  
```



**b.获取数据描述统计信息：**



```python
>>> full.describe()
               Age         Fare        Parch  PassengerId       Pclass  
count  1046.000000  1308.000000  1309.000000  1309.000000  1309.000000   
mean     29.881138    33.295479     0.385027   655.000000     2.294882   
std      14.413493    51.758668     0.865560   378.020061     0.837836   
min       0.170000     0.000000     0.000000     1.000000     1.000000   
25%      21.000000     7.895800     0.000000   328.000000     2.000000   
50%      28.000000    14.454200     0.000000   655.000000     3.000000   
75%      39.000000    31.275000     0.000000   982.000000     3.000000   
max      80.000000   512.329200     9.000000  1309.000000     3.000000   

             SibSp    Survived  
count  1309.000000  891.000000  
mean      0.498854    0.383838  
std       1.041658    0.486592  
min       0.000000    0.000000  
25%       0.000000    0.000000  
50%       0.000000    0.000000  
75%       1.000000    1.000000  
max       8.000000    1.000000  
```



**c.查看每一列数据的数据类型和数据总数**



```python
>>> full.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1309 entries, 0 to 1308
Data columns (total 12 columns):
Age            1046 non-null float64
Cabin          295 non-null object
Embarked       1307 non-null object
Fare           1308 non-null float64
Name           1309 non-null object
Parch          1309 non-null int64
PassengerId    1309 non-null int64
Pclass         1309 non-null int64
Sex            1309 non-null object
SibSp          1309 non-null int64
Survived       891 non-null float64
Ticket         1309 non-null object
dtypes: float64(3), int64(4), object(5)
memory usage: 122.8+ KB
```



#### **3.处理缺失值**

**a.数值缺失值处理**



```python
#使用fillna()填充平均数
#年龄(Age)python
full['Age'] = full['Age'].fillna(full['Age'].mean())
#船票价格
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
```



**b.字符串缺失值处理**



```python
#使用fillna()填充最频繁的值
#登船港口(Embarked)
full['Embarked'].head()
full['Embarked'] = full['Embarked'].fillna('S')
```



**c.当缺失值比较多，填充为U以表示未知**



```python
#船舱号(Cabin)，查看数据
full['Cabin'].head()
full['Cabin'] = full['Cabin'].fillna('U')
```



#### **4.特征工程**

**特征工程的目的是最大限度地从原始数据中提取特征以供算法和模型使用。**

特征工程是将原始数据转化为有用的特征，更好的表示预测模型处理的实际问题，提升对于未知数据的预测准确性。

**这是机器学习的关键，因为选取相关性越高的数据特征供模型算法训练，后面模型的正确率就越高。**

**特征工程的主要内容：**

- 特征处理：通过转换函数将原始数据转换成更加适合算法模型的特征数据。
- 特征选择：从给定的特征集合中筛选出对当前机器学习算法有用的特征。

3.2.1 特征处理

进行特征处理前，需要对数据分类，因为不同的数据类型，处理方法也不同：

- 数值类型：直接使用
- 分类数据：用数值代替类别（One_hot编码）
- 时间序列：转成单独年、月、日

**a.分类数据特征提取：性别**



```python
#分类数据特征提取：性别
sex_mapDict = {'male':1, 'female':0}

#map函数：对Series每个数据应用自定义的函数计算
full['Sex'] = full['Sex'].map(sex_mapDict)
```



**在Python提示符下输入：**



```python
>>> full['Sex'].head()
0    1
1    0
2    0
3    0
4    1
Name: Sex, dtype: int64
```



**b.分类数据特征提取：登船港口**

**在Python提示符下输入：**



```python
>>> full['Embarked'].head()
0    S
1    C
2    S
3    S
4    S
Name: Embarked, dtype: object
```



**存放提取后的特征**



```python
embarkedDf = pd.DataFrame()
#使用get_dumies进行one-hot编码，列名前缀是Embarked
embarkedDf = pd.get_dummies(full['Embarked'],prefix = 'Embarked')
```



**\#Python提示符下输入：**



```python
>>> embarkedDf.head()
   Embarked_C  Embarked_Q  Embarked_S
0           0           0           1
1           1           0           0
2           0           0           1
3           0           0           1
4           0           0           1
```



**把one-hot编码产生的虚拟变量dummy Variable添加到泰坦尼克数据集full**



```python
full = pd.concat([full,embarkedDf],axis = 1)
```



**c.分类数据特征提取：客舱等级(同b)**



```python
#存放提取后特征
pclassDf=pd.DataFrame()

#使用get_dummies进行oe-hot编码，列名前缀是pclass
pclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
#Python提示符下输入：
#pclassDf.head()

#把one-hot编码产生的虚拟变量dummy Variable添加到泰坦尼克数据集full
full=pd.concat([full,pclassDf],axis=1)
```



**d.分类数据特征提取：名字**

**Python提示符系输入：**



```python
>>> full['Name'].head()
0                              Braund, Mr. Owen Harris
1    Cumings, Mrs. John Bradley (Florence Briggs Th...
2                               Heikkinen, Miss. Laina
3         Futrelle, Mrs. Jacques Heath (Lily May Peel)
4                             Allen, Mr. William Henry
Name: Name, dtype: object
```



**字符串格式：名，头衔，姓**



```python
#Mr.Owen Harris
name1='Braund,Mr. Owen Harris'
str1=name1.split(',')[1]

#Mr.
Str2=str1.split('.')[0]

#strip()用于移除字符串头尾指定的字符（默认为空格）
Str3=Str2.strip()

#定义函数：从姓名中获取头衔

def getTitle(name):
    str1=name.split(',')[1] #Mr.OwenHarris
    str2=str1.split('.')[0] #Mr
    #strip()方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3

#存放提取后的特征
titleDf=pd.DataFrame()
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title']=full['Name'].map(getTitle)
```



**在Python提示符下输入：**



```python
>>> titleDf.head()
  Title
0    Mr
1   Mrs
2  Miss
3   Mrs
4    Mr
```



**在姓名中头衔映射字符串与定义头衔类别的映射关系**



```python
title_mapDict= {
                'Capt':       'Officer',
                'col':        'Officer',
                'Major':      'Officer',
                'Jonkheer':   'Royalty',
                'Don':        'Royalty',                            
                'Sir':        'Royalty',
                'Dr':          'Officer',
                'Rev':         'Office',
                'the Countess':'Royalty',
                'Dona':        'Royalty',
                'Mme':           'Mrs',
                'Mlle':         ' Miss',
                'Ms':            'Mrs',
                'Mr':             'Mr',
                'Mrs':            'Mrs',
                'Miss':           'Miss',
                'Master':        'Master',
                'Lady':          'Royalty'
                }
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title']=titleDf['Title'].map(title_mapDict)

#使用函数get_dummies进行one-hot编码
titleDf=pd.get_dummies(titleDf['Title'])
```



**在Python提示符下输入：**



```python
>>> titleDf.head()
    Miss  Master  Miss  Mr  Mrs  Office  Officer  Royalty
0      0       0     0   1    0       0        0        0
1      0       0     0   0    1       0        0        0
2      0       0     1   0    0       0        0        0
3      0       0     0   0    1       0        0        0
4      0       0     0   1    0       0        0        0
```



**e.分类数据特征提取：客舱号**

**Python提示符下输入：**



```python
>>> full['Cabin'].head()
0       U
1     C85
2       U
3    C123
4       U
Name: Cabin, dtype: object
```



**匿名函数语法**



```python
#定义匿名函数：对两个数相加
sum=lambda a,b:a+b

#调用sum函数
#print('相加后的值为：',sum(10,20))

#存放客舱号信息
cabinDf=pd.DataFrame()
'''
客场号的类别值是首字母，例如：
C85 类别映射首字母
'''

full['Cabin']=full['Cabin'].map(lambda c:c[0])
#使用get_dummies进行one-hot编码，列名前缀为Cabin
cabinDf=pd.get_dummies(full['Cabin'],prefix='Cabin')
```



**在Python提示符下输入：**



```python
>>> cabinDf.head()
   Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  Cabin_F  Cabin_G  Cabin_T  \
0        0        0        0        0        0        0        0        0   
1        0        0        1        0        0        0        0        0   
2        0        0        0        0        0        0        0        0   
3        0        0        1        0        0        0        0        0   
4        0        0        0        0        0        0        0        0   

   Cabin_U  
0        1  
1        0  
2        1  
3        0  
4        1  
```



**f.分类数据特征提取：家庭类别**



```python
#存放家庭信息
familyDf=pd.DataFrame()

'''
家庭人数=同代直系亲属数（parch）+不同代直系亲属数（sibSp）+乘客自己
（因为乘客自己也是家庭成员的一个，所以这里加1）
'''
familyDf['FamilySize']=full['Parch']+full['SibSp']+1

'''
家庭类别：
小家庭：family_single:家庭人数=1
中家庭：family_small:2<=家庭人数<=4
大家庭：family_large:家庭人数>=5
'''
#if条件为真时返回if前内容，否则返回0
familyDf['Family_Single']=familyDf['FamilySize'].map(lambda s:1 if s==1 else 0)
familyDf['Family_Small']=familyDf['FamilySize'].map(lambda s:1 if 2<=s<=4 else 0)
familyDf['Family_large']=familyDf['FamilySize'].map(lambda s:1 if s>=5 else 0)
```



**在Python提示符下输入：**



```python
>>> familyDf.head()
   FamilySize  Family_Single  Family_Small  Family_large
0           2              0             1             0
1           2              0             1             0
2           1              1             0             0
3           2              0             1             0
4           1              1             0             0
```



```python
#相关性矩阵
corrDf=full.corr()
corrDf
'''
查看各个特征与生成情况（Survived）的相关系数，
ascending=False表示按降序排列
'''
corrDf['Survived'].sort_values(ascending=False)
```



**在Python提示符下输入：**



```python
>>> corrDf['Survived'].sort_values(ascending=False)
Survived       1.000000
Pclass_1       0.285904
Fare           0.257307
Embarked_C     0.168240
Pclass_2       0.093349
Parch          0.081629
Embarked_Q     0.003650
PassengerId   -0.005007
SibSp         -0.035322
Age           -0.070323
Embarked_S    -0.149683
Pclass_3      -0.322308
Pclass        -0.338481
Sex           -0.543351
Name: Survived, dtype: float64
```



### 四、特征选择



```python
#特征选择
full_X=pd.concat(  [titleDf, #头衔
                    pclassDf, #客舱等级
                    familyDf, #家庭大小
                    full['Fare'], #船票价格
                    cabinDf,  #船舱号
                    embarkedDf,  #登船港口
                    full['Sex'], #性别
                    ] , axis=1 )
```



**在Python提示符下输入：**



```python
>>> full_X.head()
    Miss  Master  Miss  Mr  Mrs  Office  Officer  Royalty  Pclass_1  Pclass_2  
0      0       0     0   1    0       0        0        0         0         0   
1      0       0     0   0    1       0        0        0         1         0   
2      0       0     1   0    0       0        0        0         0         0   
3      0       0     0   0    1       0        0        0         1         0   
4      0       0     0   1    0       0        0        0         0         0   

  ...   Cabin_D  Cabin_E  Cabin_F  Cabin_G  Cabin_T  Cabin_U  Embarked_C  \
0 ...         0        0        0        0        0        1           0   
1 ...         0        0        0        0        0        0           1   
2 ...         0        0        0        0        0        1           0   
3 ...         0        0        0        0        0        0           0   
4 ...         0        0        0        0        0        1           0   

   Embarked_Q  Embarked_S  Sex  
0           0           1    1  
1           0           0    0  
2           0           1    0  
3           0           1    0  
4           0           1    1  

[5 rows x 29 columns]
```



### 五、构建模型

构建模型需要用到训练数据和机器学习算法，三者的关系如下：

## **数据 (原料)+ 算法(工具) = 模型(产品)**

训练数据就像原料，而算法就是工具，两者结合后产生模型，就是最终的产品。

```python
#构建模型
#原始数据集有891行
sourceRow=891
#原始数据集：特征
source_X=full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y=full.loc[0:sourceRow-1,'Survived']

#预测数据集：特征
pred_X=full_X.loc[sourceRow:,:]

from sklearn.cross_validation import train_test_split 

#建议模型用的训练数据集和测试数据集
train_X,test_X,train_y,test_y=train_test_split(source_X,#原始数据特征
                                               source_y,#原始数据标签
                                               train_size=0.8)

#输出数据集大小
print('原始数据集特征：',source_X.shape,
     '训练数据集特征：',train_X.shape,
     '测试数据集特征：',test_X.shape)
print('原始数据集标签：',source_y.shape,
     '训练数据集标签：',train_y.shape,
     '测试数据集标签：',test_y.shape)
```

#### 补充：**选择机器学习算法**

机器学习算法有很多种，选择哪种算法，需根据解决的具体问题而定，想详细了解如何选择的，可参考下面文章：

机器学习算法有很多种，选择哪种算法，需根据解决的具体问题而定，想详细了解如何选择的，可参考下面文章：

[](https://link.zhihu.com/?target=https%3A//blog.csdn.net/hellozhxy/article/details/80932736)

**输出结果：**



```python
原始数据集特征： (891, 29) 训练数据集特征： (712, 29) 测试数据集特征： (179, 29)
原始数据集标签： (891,) 训练数据集标签： (712,) 测试数据集标签： (179,)
```



```python
#建立模型
#导入逻辑回归算法：
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

#训练数据
#创建逻辑回归算法模型：
model.fit(train_X,train_y)

#评估
#分类问题，score得到的是模型的正确率
model.score(test_X,test_y)

#使用机器学习模型，对预测数据集中的生存情况进行预测
pred_Y=model.predict(pred_X)
'''
生成的预测值是浮点数（0.0,1.0）,
但是Kaggle要求提交的结果是整数（0,1）
所以要对数据类型进行转换
'''
pred_Y=pred_Y.astype(int)
#乘客id
passenger_id=full.loc[sourceRow:,'PassengerId']
#数据框：乘客id，预测生存情况
predDf=pd.DataFrame(
    {'PassengerId':passenger_id,
     'Survived':pred_Y})
```



**在Python提示符下输入：**



```python
>>> predDf.shape
(418, 2)
>>> predDf.head()
     PassengerId  Survived
891          892         0
892          893         1
893          894         0
894          895         0
895          896         1
```



**最后，将结果保存为csv文件：**



```python
#保存结果到titanic.pred.cvs
predDf.to_csv('titanic.pred.cvs',index=False)
```

以上就是”泰坦尼克号“机器学习案例的全部内容，总结如下：

### 六、全部代码



```python
from sklearn.cross_validation import train_test_split 
import numpy as np
import pandas as pd
import os

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#print('训练数据集大小：',train.shape,  '测试数据集大小', test.shape)

#如果数据读取失败，在Python提示符下使用这行代码查看运行路径，将数据文件放入其中。
#os.getcwd()

#合并数据集，这样可以同时对两个数据集进行清洗。
full = train.append(test, ignore_index = True)
#print('合并后的数据集： ', full.shape )

#Python命令行下：
#full.head()  #查看数据
#full.describe()  #获取数据描述统计信息。
#full.info()  #查看每一列的数据类型和数据总数

#使用fillna()填充平均数
#年龄(Age)
full['Age'] = full['Age'].fillna(full['Age'].mean())
#船票价格
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())

#使用fillna()填充最频繁的值
#登船港口(Embarked)
full['Embarked'].head()
full['Embarked'] = full['Embarked'].fillna('S')
#当缺失值比较多，填充为U以表示未知。
#船舱号(Cabin)，查看数据
full['Cabin'].head()
full['Cabin'] = full['Cabin'].fillna('U')

#分类数据特征提取：性别
sex_mapDict = {'male':1, 'female':0}

#map函数：对Series每个数据应用自定义的函数计算
full['Sex'] = full['Sex'].map(sex_mapDict)
#Python命令行输入：
#full['Sex'].head()

#分类数据特征提取：登船港口
#查看原始数据类型，在Python提示符下输入：
#full['Embarked'].head()

#存放提取后的特征
embarkedDf = pd.DataFrame()

#使用get_dumies进行one-hot编码，列名前缀是Embarked
embarkedDf = pd.get_dummies(full['Embarked'],prefix = 'Embarked')
#Python提示符下输入：
#embarkedDf.head()

#把one-hot编码产生的虚拟变量dummy Variable添加到泰坦尼克数据集full
full = pd.concat([full,embarkedDf],axis = 1)

#分类数据特征提取：客舱等级
#存放提取后特征
pclassDf=pd.DataFrame()

#使用get_dummies进行oe-hot编码，列名前缀是pclass
pclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
#Python提示符下输入：
#pclassDf.head()

#把one-hot编码产生的虚拟变量dummy Variable添加到泰坦尼克数据集full
full=pd.concat([full,pclassDf],axis=1)

#分类数据特征提取：名字
#查看数据
#分类数据特征提取：姓名
#Python提示符下输入：
#full['Name'].head()

#字符串格式：名，头衔.姓
#Mr.Owen Harris
name1='Braund,Mr. Owen Harris'
str1=name1.split(',')[1]

#Mr.
Str2=str1.split('.')[0]

#strip()用于移除字符串头尾指定的字符（默认为空格）
Str3=Str2.strip()

#定义函数：从姓名中获取头衔

def getTitle(name):
    str1=name.split(',')[1] #Mr.OwenHarris
    str2=str1.split('.')[0] #Mr
    #strip()方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3

#存放提取后的特征
titleDf=pd.DataFrame()
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title']=full['Name'].map(getTitle)
#Python提示符下输入：
#titleDf.head()

#姓名中头衔映射字符串与定义头衔类别的映射关系
title_mapDict= {
                'Capt':       'Officer',
                'col':        'Officer',
                'Major':      'Officer',
                'Jonkheer':   'Royalty',
                'Don':        'Royalty',                            
                'Sir':        'Royalty',
                'Dr':          'Officer',
                'Rev':         'Office',
                'the Countess':'Royalty',
                'Dona':        'Royalty',
                'Mme':           'Mrs',
                'Mlle':         ' Miss',
                'Ms':            'Mrs',
                'Mr':             'Mr',
                'Mrs':            'Mrs',
                'Miss':           'Miss',
                'Master':        'Master',
                'Lady':          'Royalty'
                }
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title']=titleDf['Title'].map(title_mapDict)

#使用函数get_dummies进行one-hot编码
titleDf=pd.get_dummies(titleDf['Title'])

#Python提示符下输入：
#titleDf.head()

#查看客舱号的内容
#Python提示符下输入：
#full['Cabin'].head()

#定义匿名函数：对两个数相加
sum=lambda a,b:a+b

#调用sum函数
#print('相加后的值为：',sum(10,20))

#存放客舱号信息
cabinDf=pd.DataFrame()
'''
客场号的类别值是首字母，例如：
C85 类别映射首字母
'''

full['Cabin']=full['Cabin'].map(lambda c:c[0])
#使用get_dummies进行one-hot编码，列名前缀为Cabin
cabinDf=pd.get_dummies(full['Cabin'],prefix='Cabin')

#Python提示符下输入：
#cabinDf.head()


#存放家庭信息
familyDf=pd.DataFrame()

'''
家庭人数=同代直系亲属数（parch）+不同代直系亲属数（sibSp）+乘客自己
（因为乘客自己也是家庭成员的一个，所以这里加1）
'''
familyDf['FamilySize']=full['Parch']+full['SibSp']+1

'''
家庭类别：
小家庭：family_single:家庭人数=1
中家庭：family_small:2<=家庭人数<=4
大家庭：family_large:家庭人数>=5
'''
#if条件为真时返回if前内容，否则返回0
familyDf['Family_Single']=familyDf['FamilySize'].map(lambda s:1 if s==1 else 0)
familyDf['Family_Small']=familyDf['FamilySize'].map(lambda s:1 if 2<=s<=4 else 0)
familyDf['Family_large']=familyDf['FamilySize'].map(lambda s:1 if s>=5 else 0)

#Python提示符下输入：
#familyDf.head()

#相关性矩阵
corrDf=full.corr()
corrDf
'''
查看各个特征与生成情况（Survived）的相关系数，
ascending=False表示按降序排列
'''
corrDf['Survived'].sort_values(ascending=False)

#特征选择
full_X=pd.concat(  [titleDf, #头衔
                    pclassDf, #客舱等级
                    familyDf, #家庭大小
                    full['Fare'], #船票价格
                    cabinDf,  #船舱号
                    embarkedDf,  #登船港口
                    full['Sex'], #性别
                    ] , axis=1 )

#Python提示符下输入：
#full_X.head()

#构建模型
#原始数据集有891行
sourceRow=891
#原始数据集：特征
source_X=full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y=full.loc[0:sourceRow-1,'Survived']

#预测数据集：特征
pred_X=full_X.loc[sourceRow:,:]

#建议模型用的训练数据集和测试数据集
train_X,test_X,train_y,test_y=train_test_split(source_X,#原始数据特征
                                               source_y,#原始数据标签
                                               train_size=0.8)

#输出数据集大小
'''
print('原始数据集特征：',source_X.shape,
     '训练数据集特征：',train_X.shape,
     '测试数据集特征：',test_X.shape)
print('原始数据集标签：',source_y.shape,
     '训练数据集标签：',train_y.shape,
     '测试数据集标签：',test_y.shape)
'''

#建立模型
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
#训练数据
model.fit(train_X,train_y)

#评估
#分类问题，score得到的是模型的正确率
model.score(test_X,test_y)

#使用机器学习模型，对预测数据集中的生存情况进行预测
pred_Y=model.predict(pred_X)
'''
生成的预测值是浮点数（0.0,1.0）,
但是Kaggle要求提交的结果是整数（0,1）
所以要对数据类型进行转换
'''
pred_Y=pred_Y.astype(int)
#乘客id
passenger_id=full.loc[sourceRow:,'PassengerId']
#数据框：乘客id，预测生存情况
predDf=pd.DataFrame(
    {'PassengerId':passenger_id,
     'Survived':pred_Y})

#在Python提示符下输入：
#predDf.shape
#predDf.head()

#保存结果到titanic.pred.cvs
predDf.to_csv('titanic.pred.cvs',index=False)
```