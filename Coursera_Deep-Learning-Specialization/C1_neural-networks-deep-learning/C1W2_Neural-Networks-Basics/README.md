C1W2L01_Binary Classification
=============

고양이 분류
1(cat) or 0(not cat)
RGB 채널 64*64
**n_x**(x의 차원) - 64*64*3 = 12288
**m** - 데이터의 수
**X** - 모든 학습 데이터, X.shape = (n_x, m)
Y - 모든 y 데이터, Y.shape = (1, m)


C1W2L02_Logistic Regression
=============

Y의 값이 0 또는 1인 경우 사용되는 알고리즘
**ŷ** - 예측한 y값 = P(y=1 | x)
파라미터 w, b
Output ŷ = σ(w^T*x + b)
σ(z) = 1 / (1 + e^-z)
z가 크면 e^-z는 0에 가까워진다 → 1 / 1.000... = 1에 가까워진다
z가 작으면 e^-z는 무한 → 1 / 무한 = 0에 가까워진다
w와 b를 따로 취급

> What are the parameters of logistic regression?<br><br>
_**W, an n_x dimensional vector, and b, a real number.**_

C1W2L03_Logistic Regression Cost Function
=============

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/02909275-a7fd-424d-af2e-41429200aa98/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/02909275-a7fd-424d-af2e-41429200aa98/Untitled.png)

Loss(error) function - 예측과 실제값의 불일치 정도

Cost function - 모든 훈련 데이터의 loss function의 평균

> What is the difference between the cost function and the loss function for logistic regression?<br><br>
_**The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.**_

C1W2L04_Gradient Descent
=============

Cost function J(w, b)를 가장 작게 만드는 w, b를 찾아야한다

**α** - Learning Rate

초기점에서 시작해 가장 가파른 방향으로 진행

w := w - α * dJ(w)/dw

> True or false. A convex function always has multiple local optima.<br><br>
_**False**_

C1W2L05_Derivatives / C1W2L06_More Derivative Examples
=============

도함수

f(a) = 3a

a = 2 → f(a) = 6

a = 2.001 → f(a) = 6.003

slope (derivative) of f(a) at a=2 is 3

> On a straight line, the function's derivative...<br><br>
_**doesn't change.**_

f(a) = a^2

a = 2 → f(a) = 4

a = 2.001 → f(a) = 4.004...

slope (derivative) of f(a) at a=2 is 4

a = 5 → f(a) = 36

a = 5.001 → f(a) = 4.010...

slope (derivative) of f(a) at a=5 is 10

f'(a) = 2a

f(a) = ln(a)

f'(a) = 1/a

C1W2L07_Computation Graph / C1W2L08_Derivatives with a Computation Graph
=============

J(a, b, c) = 3(a + bc)

순방향

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b134cd11-aef2-451a-bcce-73c73387bc47/Untitled.png)

> One step of ________ propagation on a computation graph yields derivative of final output variable.<br><br>
_**Backward**_

역방향
    J = 3v → dJ/dv = 3
    v = a + u → dJ/da = 3 = dJ/dv * dv/da : Chain Rule
    dv/da = 1
    d{FinalOutputVar}/d{var} → dvar (코드에서 변수명) 
    dJ/da = 3
    dJ/du = 3
    dJ/db = dJ/du * du/db = 3 * 2 = 6
    dJ/dc = dJ/du * du/dc = 3 * 3 = 9

> In this class, what does the coding convention dvar represent?<br><br>
_**The derivative of a final output variable with respect to various intermediate quantities.**_

C1W2L09_Logistic Regression Gradient Descent / C1W2L10_Gradient Descent on m Examples
=============

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3ef14acf-66e0-4e81-9d79-a1f0e9557591/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0e1d44dd-eeb3-4220-874d-5f9e9d7e5960/Untitled.png)

    da = dL/da = -y/a + (1-y)/(1-a)
    dz = dL/dz = dL/da * da/dz = {-y/a + (1-y)/(1-a)} * a(1-a) = a - y
    dw1 = dL/dw1 = x1 * dz
    dw2 = x2 * dz
    db = dz
값 갱신
    w1 := w1 - (α * dw1)
    w2 := w2 - (α * dw2)
    b := b - (α * db)

> In this video, what is the simplified formula for the derivative of the losswith respect to z?<br><br>
_**a - y**_

m개의 데이터에 대한 Logistic Regression

dJ/dw1 = 1/m * sum(dw1^(i))

```python
J = 0, dw1 = 0, dw2 = 0, db = 0
for i in range(0, m):
	zi = wi * xi + b
	ai = sigmoid(zi)
	J += -[yi * log(ai) + (1 - y(i)) * log((1-ai))]
	dzi = ai - ui
	dw1 += x1i * dzi
	dw2 += x2i * dzi
	db += dzi
J /= m
dw1 /= m
dw2 /= m
db /= m
# dw1 = dj / dw1 (overall)
w1 := w1 - α * dw1
w2 := w2 - α * dw2
b := b - α * db
```

for loop을 제거하기위해 Vectorization을 사용

더욱 더 큰 데이터를 학습하기에 용이

> In the for loop depicted in the video, why is there only one dw variable (i.e. no i superscripts in the for loop)?<br><br>
_**The value of dw in the code is cumulative.**_

C1W2L11_Vectorization / C1W2L12_More Vectorization Examples
=============

코드에서 for loop을 제거하는 기술

**np.dot(A, B)**

```python
import numpy as np

a = np.array([1,2,3,4])
print(a)
# [1 2 3 4]

import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print("Vectorized version: " + str(1000 * (toc-tic)) + "ms")
# Vectorized version: 1.502752...ms

c = 0
tic = time.time()
for i in range(1000000):
	c += a[i] * b[i]
toc = time.time()
print("For loop: " + str(1000 * (toc-tic)) + "ms")
# For loop: 474.295139...ms
```

SIMD: Single Instruction Multiple Data

병렬 처리를 빠르게 해줌

explicit for loop을 피하라.

True or false. Vectorization cannot be done without a GPU.
**False**

**np.exp(V)**

```python
import numpy as np
u = np.exp(v)

np.log(v)
np.abs(v)
np.maximum(v, 0)
v**2
1/v
# ...
```

Logistic regresion derivatives

```python
dw = np.zeros(n_x, 1)
dw += xi * dzi
dw /= m
```

C1W2L13_Vectorizing Logistic Regression / C1W2L14_Vectorizing Logistic Regression's Gradient Output
=============

순방향

Z = w^T * X + [b ... b]

X.shape, w.shape: (n_x, m)

→ Z = np.dot(w.t, x) + b

b가 Broadcasting되어 (1, m)이 된다

A = σ(Z)

What are the dimensions of matrix X in this video?
(n_x, m)

역전파

dz = [dz1, dz2, ... , dzm]

dz.shape: (1, m)

A = [a1, ... , am], Y = [y1, ... , ym]

dz = A - Y

db = (1 / m) * np.sum(dz)

dw = (1 / m) * X * dz^T

 = 1/m[x1*dz1 + ... + xm * dzm]

→ (n, 1)

```python
Z = np.dot(w.T, X) + b
A = σ(Z)
dz = A - Y
dw = 1 / m * X * dz.T
db = 1 / m * np.sum(dz)

w := w - α * dw
b := b - α * db
```

> How do you compute the derivative of b in one line of code in Python numpy?<br><br>
_**1 / m*(np.sum(dz))**_

C1W2L15_Broadcasting in Python
=============

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/57fdaf81-51e0-4609-8180-cd995162fd65/Untitled.png)

A.shape: (3, 4)

각 음식의 탄단지의 칼로리 비율 구하기 (without for loop)

```python
import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
							[1.2, 104.0, 52.0, 8.0],
							[1.8, 135.0, 99.0, 0.9]])
# 세로로 더하기
cal = A.sum(axis=0)
print(cal)
# [59. 239. 155.4 76.9]

percentage = 100 * A / cal.reshape(1, 4)
print(percentage)
```

> Which of the following numpy line of code would sum the values in a matrix A vertically?<br><br>
_**A.sum(axis = 0)**_

C1W2L16_A Note on Python/Numpy Vectors
=============

```python
a = np.random.randn(5)
# a.shape = (5,)
# rank 1 array -> 사용하지 않기

a = np.random.randn(5, 1)
a = np.random.randn(1, 5)
assert(a.shape == (5, 1))
```

> What kind of array has dimensions in this format: (10, ) ?<br><br>
_**A rank 1 array**_

Explanation of Logistic Regression Cost Function (Optional)
=============

???

> True or False: Minimizing the loss corresponds with maximizing logp(y|x).<br><br>
_**True**_