C2W1L01_Train / Dev / Test sets
==========

- Train set / Validation set (dev set) / Test set
- 데이터가 많아지면서 dev, test set의 %가 줄었다
  - 100만개일 때, 98 : 1 : 1
```
Make sure dev and test set come from same distribution
```
- Dev set을 이용하여 다양한 모델들을 평가하고 Dev set을 개선하려고 할 것이기 때문에 Dev와 Test set은 같은 분포를 가져야한다

<br>

C2W1L02_Bias / Variance
==========

- 편향과 분산(편차)
- Bias-variance tradeoff
  - 편향(Bias) - 학습 알고리즘에서 잘못된 가정을 했을 때 발생하는 오차 - 높은 편향값 → 과소적합(underfitting)
  - 분산(Variance) - 트레이닝 셋에 내재된 작은 변동때문에 발생하는 오차 - 높은 분산값 → 과적합(overfitting)

<br>

C2W1L03_**Basic Recipe for Machine Learning**
==========

- High bias?
  - Train data performance
  - → bigger network, train longer (NN architecture search)
- High variance?
  - Dev set performance
  - → More data, Regularization

<br>

C2W1L04_Regularization
==========

- 정규화 - Overfitting을 줄여준다

<br>

C2W1L05_Why Regularization Reduces Overfitting?
==========

- 정규화 $\lambda$ 값이 커지면 W값이 줄어들어 z값을 0에 가깝게 만들어서 선형처럼
  - overfitting을 방지한다

<br>

C2W1L06_Dropout Regularization
==========

- dropout이 0.8일 때
- keep_prob = 0.8
- d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
- a3 = np.multiply(a3, d3)
  - → a3 /= d3

<br>

C2W1L07_Understanding Dropout
==========

```
Can't rely on any one feature, so have to spread out weights.
```
- 드랍아웃을 통해 가중치를 분산시킨다
- CV에서 드랍아웃을 많이 사용
  - Input size가 커서???
  - 충분한 데이터가 없기 때문에 대부분의 경우 과적합이 발생
- 과적합이 발생하기 전까지는 dropout 사용을 안한다

<br>

C2W1L08_Other Regularization Methods
==========

- Overfitting을 방지하는 방법들 (정규화, 드롭아웃 제외)

1. Data augmentation
   - 사진 : 뒤집고 늘이고 확대하고 찌그러트리고 등등
2. Early stopping
   - dev set error가 증가하기 시작하면 stop
   - 정규화를 하는 방법이 있지만 lambda 값을 찾아야하기 때문에 비싼 방법
- Orthogonalization
  - Cost Function을 줄이는 방법
  - 과적합을 방지하는 방법을 별개의 문제로 생각

<br>

C2W1L09_Normalizing Inputs
==========

1. 평균을 빼기
2. 분산 정규화
- 학습 셋과 테스트 셋에 똑같은 방법을 적용할 것
- 경사하강법에서 왔다갔다 안해도 되기 때문에 learning rate를 크게 할 수 있다

<br>

C2W1L10_Vanishing / Exploding Gradients
==========

- weight가 1보다 작은 값이면 활성화가 기하급수적으로 줄어든다
- 반대로 1보다 큰 값이면 기하급수적으로 늘어난다

<br>

C2W1L11_Weight Initialization for Deep Networks
==========

- 특성의 개수(n)가 많을수록 w가 작아야함
- 편차(wi) = 2/n
- $W^{[l]}$ = np.random.randn(shape) * np.sqrt(2 / n[l-1])

<br>

C2W1L12_Numerical Approximation of Gradients
==========

- 미분
  - two-sided difference
    - 양쪽을 측정해서 $2\epsilon$을 사용하는 것이 error가 적다

<br>

C2W1L13_Gradient Checking
==========

- $W^{[1]}, b^{[1]},\;...\;, W^{[L]}, b^{[L]}$을 모아 $\theta$ 벡터로 reshape
- $\mathrm{d}W^{[1]}, \mathrm{d}b^{[1]},\;...\;, \mathrm{d}W^{[L]}, \mathrm{d}b^{[L]}$을 모아 - $\mathrm{d}\theta$ 벡터 reshape
- Is $\mathrm{d}\theta$ the gradient of cost function $J(\theta)$?
  - two-sided difference를 활용하여 
  - $\mathrm{d}\theta_{approx}[i] = \frac{J(\theta_1,\;\dots\;,\theta_i-\epsilon,\;\dots) - J(\theta_1,\;\dots\;,\theta_i+\epsilon,\;\dots)}{2\epsilon} \approx \mathrm{d}\theta[i] = \frac{\partial{J}}{\partial{\theta_i}}$
  - 를 모든 i에 대해 구한 다음
  - $\mathrm{d}\theta_{approx} \approx \mathrm{d}\theta$ 을 확인
    - $\epsilon = 10^{-7}$일 때,
    - $||a - b||_2$ = sqrt(요소들의 차의 제곱의 합)
    - $\frac{||\mathrm{d}\theta_{approx} - \mathrm{d}\theta||_2}{||\mathrm{d}\theta_{approx}||_2 + ||\mathrm{d}\theta||_2} \approx 10^{-7}$면 great!
  - 이보다 큰 값이 나오면 요소들의 차가 큰지 확인
- foreprop backprop grad-check

<br>

C2W1L14_Gradient Checking Implementation Notes
==========

- Don't use in training - only in debug
- If algorithm fails grad check, look at components to try to identify bug.
- Remember regularization
- Doesn\t work with dropout
- Run at random initalization; perhaps again after some training