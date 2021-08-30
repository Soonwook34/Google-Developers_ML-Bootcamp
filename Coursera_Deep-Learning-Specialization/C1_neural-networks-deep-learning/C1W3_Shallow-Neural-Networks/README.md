C1W3L01_Neural Networks Overview
=============

- W^[1]에서 [1]은 레이어 번호를 의미

<br>

C1W3L02_Neural Network Representation / C1W3L03_Computing a Neural Network's Output
=============

    [그림]

- 2-Layered Neural Network
    - x1, x2, x3: Input Layer
    - Hidden Layer: 학습set에서 보이지 않음
    - Output Layer

    [그림]

    [그림]

- x는 a^[0]로 대체 가능
    - z^[1], a^[1] - (4, 1)
    - z^[2], a^[2] - (1, 1)

<br>

C1W3L04_Vectorizing Across Multiple Examples / C1W3L05_Explanation for Vectorized Implementation
=============

a^[2](1)는 1번째 데이터의 2번째 layer를 의미

    [그림]

- X - (n_x, m)
- A - (..., m)
- 행렬의 가로 - Training Example의 번호
- 행렬의 세로 - Hidden Units의 번호

    [그림]

<br>

C1W3L06_Activation Functions
=============

1. 시그모이드
    - a = 1 / ( 1 + e^-z)
    - 범위: 0 ~ 1
    - 이진 분류를 할 때 사용
2. tanh
    - a = (e^z - e^-z) / (e^z + e^-z)
    - 범위: -1 ~ 1
    - 시그모이드보다 대부분의 상황에서 좋다 → 중심값이 0이라서
- 시그모이드와 tahnk 모두 z가 극한으로 갈수록 기울기가 0에 가까워짐 → 경사 하강법이 느림

1. ReLU
    - a = max(0, z)
    - 이진 분류를 제외한 모든 경우에 ReLU가 가장 많이 사용됨
2. Leaky ReLU
    - z가 0이하일 때 약간의 기울기를 주는 것

    [그림]

<br>

C1W3L07_Why do you need Non-Linear Activation Functions? / C1W3L08_Derivatives of Activation Functions
=============

- 비선형 활성화 함수가 필요한 이유
  - 선형 활성화 함수를 사용하거나 활성화 함수를 사용하지 않는다면 Layer가 아무리 많아도 선형 Activation 함수를 산출하는 것이 된다 → Not Expressive (Hidden Layer가 없는 것과 같다)
- 선형 함수는 결과값 Layer에서 사용하는 경우가 있다
    - 값이 0 ~ 1이 아니라 실수집합일 때

1. 시그모이드
    - g'(z) = g(z)(1-g(z)) = a(1-a)
2. tanh
    - g'(z) = 1 - g(z)^2 = 1 - a^2
3. ReLU
    - g'(z) = 0 if z < 0, 1 if z ≥ 0
    - → z가 0이 될 확률이 매우 적기 때문에 성립

<br>

C1W3L09_Gradient Descent for Neural Networks
=============

- Forward Propagation
    - Z^[1] = W^[1]X + b^[1]
    - A^[1] = g^[1](Z^[1])
    - Z^[2] = W^[2]A^[1] + b^[2]
    - A^[2] = g^[2](Z^[2])

- Back Propagation
    - dZ^[2] = A^[2] - Y
    - dw^[2] = 1/m * dZ^[2]A^[1]T
    - db^[2] = 1/m * np.sum(dz^[2], axis=1, keppdims=True)
    - dZ^[1] = W^[2]TdZ^[2] * g[1]'(Z^[1])
    - dW^[1] = 1/m * dZ^[1]X^T
    - db[1] = 1/m * np.sum(dZ^[1], axis=1, keepdims=True)

<br>

C1W3L10_Backpropagation Intuition (Optional)
=============

- 꼭 복습하자

    [그림]

<br>

C1W3L11_Random Initialization
=============

- 0으로 weights를 초기화하면 안되는 이유
  - hidden units이 완전히 동일(Symmetric) → 동일한 결과를 산출 → 다음 layer에 주는 영향 동일
    - → hidden units의 수가 무의미해진다
- W^[1] = np.random.randn((2, 2,)) * 0.01
  - 너무 큰 값을 곱하지 않는 이유는 Z^[1]을 산출할 때 큰 값 → A^[1] 값 커짐 → 경사 하강법 느림 → 학습 속도 감소