C1W4L01_Deep L-layer Neural Network
==========

- shallow ↔ deep
- L = # layers
- n^[l] = #unit in layer l
- n^[0] = n_x
- a^[L] = y^

<br>

C1W4L02_Forward Propagation in a Deep Network
==========

- Z^[1] = W^[1]X + b^[1]
- A^[1] = g^[1](Z^[1])
- Z^[2] = W^[2]A^[1] + b^[2]
- A^[2] = g^[2](Z^[2])
- ...
- 일반화
  - Z^[l] = W^[l]A^[l-1] + b^[l]
  - A^[l] = g^[l](Z^[l]), A[0] = X
  - → layer를 진행하는 for loop을 없앨 수 없다

<br>

C1W4L03_Getting your Matrix Dimensions Right
==========

- 행렬의 차원 확인하기

    [그림]

- Layer 0
  - n^[0] = n_x = 2
- Layer 1
  - n^[1] = 3
  - z^[1] = W^[1]x + b^[1]
    - z^[1] → (n^[1], 1) = (3, 1)
    - x → (n^[0], 1) = (2, 1)
    - w^[1] → (n^[1], n^[0]) = (3, 2)
    - b^[1] → (n^[l], 1) = (3, 1)
- 일반화
- z^[l], a^[l] → (n^[l], 1)
- Z^[l], A^[l] → (n^[l], m)
- X → (n[0], m)
- dZ^[l], dA^[l] → n^[l], m)
- W^[l] → (n^[l], n^[l-1])
- b^[l] → (n^[l], 1)
  - bias는 broadcasting에 의해 (n^[l], m) 차원으로 복제
- dW^[l] → (n^[l], n^[l-1])
- db^[l] → (n^[l], 1)

<br>

C1W4L04_Why Deep Representations?
==========

- 작은 것으로부터 layer를 지날수록 취합?
- 간단한 것으로 시작해서 복잡한 것까지 (ex. 모서리 → 얼굴)
- Layer가 깊어질수록 Hidden Unit의 필요 수가 적어진다
  - "exponentially large"
    > Informally: There are functions you can compute with a "small" L-layer depp eneural network that shallower networks require exponentially more hidden units to compute.
- Deep → 딥러닝을 Rebranding

<br>

C1W4L05_Building Blocks of Deep Neural Networks
==========

- Forward
  - Input: a^[l-1], output: a^[l]
  - z^[l] = W^[l]a^[l-1] + b^[l]
  - a^[l] = g^[l](z^[l]) → cache z^[l]
- Backward
  - Input: da^[l], output da^[l-1]
  - use cache z^[l]
- Layer 1개

    [그림]

- 전체 NN

    [그림]

<br>

C1W4L06_Forward and Backward Propagation
==========

- Forward Propagation for layer l
  - 초기값: X
  - Input: a^[l-1]
  - Output: a^[l], cache (z^[l], W^[l], b^[l])
  - z^[l] = W^[l]a^[l-1] + b^[l] → Z^[l] = W^[l]A^[l-1] + b^[l]
  - a^[l] = g^[l](z^[l]) → A^[l] = g^[l](Z^[l])
- Backward Propagation for layer l
  - 초기값: da^[l] = -y/a + (1-y)/(1-a)
  - Input: da^[l]
  - Output: da^[l-1], dW^[l], db^[l])
  - dz^[l] = da^[l] * g[l]'(z^[l]) → dZ^[l] = dA^[l] * g^[l]'(Z^[l])
  - dw^[l] = dz^[l]a^[l-1] → dW^[l] = 1/m * dZ^[l]A^[l-1].T
  - db^[l] = dz^[l] → db^[l] → 1/m * np.sum(dZ^[l], axis=1, keepdims=True)
  - da^[l-1] = W^[l].Tdz^[l] → dA[l-1] = W[l].TdZ^[l]
  - → dz^[l] = W^[l+1].Tdz^[l+1] * g^[l]'(z^[l])

<br>

C1W4L07_Parameters vs Hyperparameters
==========

- Parameters: W^[1], b^[1], ...
- Hyperparameters: Learning rate α, #iterations, #hidden layers L, # hidden units n^[l], ... , choice of activation function
- Empirical Process → 경험적 과정
  - 학습률 조정, 레이어 수 조정, ...

<br>

C1W4L08_What does this have to do with the brain?
==========

    [그림]

- 별로 관련이 없지만
- "It's like the brain"
  - → 딥러닝을 단순화하는 경향이 있지만, 사람들에게 공식적으로 말하거나 매체에 알릴 때 매력적, 대중의 - 상상력을 자극
- 사람의 뉴런과 단일 로지스틱 유닛을 비교
  - → 뉴런의 작동 원리와 하는 일에 대해 불분명
- 주장이 점점 무너져간다