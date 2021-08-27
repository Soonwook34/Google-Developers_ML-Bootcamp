### C1W1L01_What is Neural Network?

Linear Regression

    ex) Housing Price Prediction

        size → (neuron) → price

        x{size, #bedrooms, zip code, wealth} → y{price}

    ReLU (Rectified Linear Units) - 특정 부분까지 0, 이후 일차함수로 값이 증가하는 함수

> True or false? As explained in this lecture, every input layer feature is interconnected with every hidden layer feature.

> _True_

---

### C1W1L02_Supervised Learning with Neural Networks

Supervised Learning

예시

| Input(x) | Output(y) | Application | Neural Networks |
| :-----: | :---: | :---: | :---: |
| Home features | Price | Real Estate | Standard NN |
| Ad, user info | Click on ad? (0/1) | Online Advertising | Standard NN |
| Image | Object (1, ... , 1000) | Photo tagging | CNN |
| Audio | Text transcript | Speech recongnition | RNN |
| English | Chinese | Machine translation | RNN |
| Image, Radar info | Position of other cars | Autonomous driving | Custom, Hybrid |

|제목|내용|설명|
|------|---|---|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|

    Structured Data - Table 데이터

    Unstructured Data - Audio, Image, Text

        컴퓨터가 Unstructured Data를 이해하는데 어려웠지만, 딥러닝을 통해 가능성
        
> Would structured or unstructured data have features such as pixel values or individual words?

> _Unstructured data_

---

### C1W1L03_Why is Deep Learning taking off?

더 많은 데이터를 축적함으로써 새로운 네트워크의 장점을 극대화시킬 수 있었다
(m) - 데이터의 수
딥러닝의 초기 시점
    scaled data, scale computation이 중심적 역할 - 아주 큰 신경망 네트워크를 트레이닝 하는 방법
딥러닝의 현재
    algorithms의 혁신
        ex) 시그모이드 vs. ReLU
            시그모이드 함수의 기울기가 양끝으로 갈수록 0에 가깝기 때문에 학습이 느려짐
            ReLU는 양수에서 기울기가 1이기 때문에 0으로 줄어드는 확률 감소 → 학습이 빨라짐
학습 속도가 중요한 이유
    Idea → Code → "Experiement"의 주기가 빨라짐으로써 idea를 빨리 개선시킬 수 있다

> What will the variable m denote in this course?
> _Number of training examples_