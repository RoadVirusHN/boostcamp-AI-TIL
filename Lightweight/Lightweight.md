# Lightweight

> Naver AI boostcamp 내용을 정리하였습니다.

## Lightweight model 개요

**1. 결정 (Decision making)**

**연역적 (deductive) 결정**

이미 참으로 증명되거나 정의를 통하여 논리를 입증하는 것

삼단 논법과 조합 $\begin{pmatrix}
 n\\r
\end{pmatrix}=\frac{n!}{r!(n-r)!}$ 증명 등이 있다.

- 삼단 논법 예시 : 소크라테스는 사람이다. 사람은 모두 죽는다. 소크라테스는 죽는다.

$$
Definition:\begin{pmatrix}
 n\\r
\end{pmatrix}=\frac{n!}{r!(n-r)!}\\
n!=n\cdot(n-1)\cdot(n-2)\dots\cdot2\cdot1라면,\\
Theorem:1+2+\dots+(n-1)=\begin{pmatrix}
 n\\r
\end{pmatrix}이다.
$$

**[math. 조합의 가정]**

![image-20210315195200886](Lightweight.assets/image-20210315195200886.png)

**[img. 조합의 증명]**

**귀납적 (inductive) 결정**

반복된 관찰과 사례를 통하여 논리를 입증

많은 통계학적 기법의 증명이 속한다.

예시: 몇만년동안 해는 동쪽에서 떠서 서쪽에서 진다 -> 고로 내일도 그럴 것이다.

단점으로, 100% 보장하지 않는다. (ex) 만약, 내일 태양이 폭발한다면?)

**2. 결정기 (Decision making machine)**

이전의 기계들은 사람이 결정을 하면 해당 결정의 목적 달성을 위해 도움을 주는 형태였다.

최근의 인공지능이 포함된 기기는 이때의 결정 또한 해주거나, 결정에 도움을 제공.

(ex) 공기질이 나빠지면 자동으로 켜지는 공기청정기)

![image-20210315201319264](Lightweight.assets/image-20210315201319264.png)

**[img. 청소기를 예시로 든 결정기]**

![image-20210315201630448](Lightweight.assets/image-20210315201630448.png)

**[img. 가장 간단한 통계학적 결정기인 평균]**

![image-20210315202313895](Lightweight.assets/image-20210315202313895.png)

**[img. 세상에는 중요한 결정과 쉬운 결정, 그리고 결정할 수 없는것이 있다]**

- 근본적 가치에 관계된 일, 예측할 수 없는 일, 책임질 수 없거나 책임이 너무 큰일 등은 결정하기 힘들거나 결정할 수 없다.

![image-20210315202153583](Lightweight.assets/image-20210315202153583.png)

**[img. 과거에는 성능 부족으로 추천 정도만 했지만 최근에는 의사결정에 관여 가능한 수준]**

![image-20210315201936039](Lightweight.assets/image-20210315201936039.png)

**[img. 현재 기술로는 잘해봐야 Rules & conditions 까지만 결정 가능, 가치와 중요 결정은 불가]**

**3. 가벼운 결정기 (Lightweight decision making machine)**

경량화(Lightweight)는 성능이나 가치가 차이가 없거나 적게 희생하고 규모와 비용 등을 줄이는 것이 것

소형화(Miniaturization)은 말 그대로 물건의 크기를 줄이는 것

![image-20210315203300348](Lightweight.assets/image-20210315203300348.png)

**[img. 소형 기기에 ML 모델을 넣기 위한 Cycle]**

최근에는 TinyML이라 하여 칩 수준의 소형기기를 위한 메모리 사용량과 연산량이 적은 Project 또한 진행되고 있다.

![image-20210315203415533](Lightweight.assets/image-20210315203415533.png)

**[img. Edge Computing의 장점 시나리오]**

Edge Computing은 네트워크에서 끝 부분에 해당하는 보통 소형기기에서의 연산하는 서비스를 의미하며, 많은 장점을 가지고 있다

예를 들어, 자율주행의 연산을 Cloud Computing으로 처리할 시, 보안 문제, 연결안전성 문제, 대규모 연산에 의한 전력과 기기 소모 등이 필요하지만

Edge Computing은 이러한 문제를 해결할 수 있다.

**4. Backbone & dataset for model compression**

성능좋고 안정성 있는 Pre-trained 모델이 많이 나와있다.

이러한 Pre-trained 모델을 경량화하여 사용하면, 검증된 성능과 문제 해결, 빠른 개발 등의 장점이 있다.

![image-20210315204427413](Lightweight.assets/image-20210315204427413.png)

**[img. 최근에는 다양한 pretrained model이 많다.]**

MiB(메비바이트) : 2진법 기준 용량 계산법, 2^20 바이트 

![image-20210315204446909](Lightweight.assets/image-20210315204446909.png)

**[img. Classification을 위한 Dataset들, 점점 커진다]**

**5. Edge devices**

![image-20210315210230781](Lightweight.assets/image-20210315210230781.png)

**[img. 각종 computing 방법]**

Cloud는 높은 가격과 보안 문제, 네트워크 연결이 필수이며, 연결이 불안전하거나 부하가 걸리면 사용하기 힘들다.

On-premise는 직접 서버를 두는 방식으로, 법적으로 정보가 네트워크에 누출 되지 않은 경우나, 대규모로 구현할 수 있는 회사등에서만 사용 가능하며, 유지보수 비용이 크다.

Edge device는 저비용, 높은 보안성, stand-alone 동작과 네트워크 연결 여부를 선택할 수 있지만, 연산량이나 전력 소모, 메모리가 제한된다는 단점이 있다.(Dumb and fast)

최근에는 Rasberry Pi, Jetson nano 처럼 합리적인 가격으로 사용해 볼 수 있다.

**6. Edge intelligence**

![image-20210315212913774](Lightweight.assets/image-20210315212913774.png)

**[img. Cloud intelligence vs edge intelligence]**

![image-20210315213029425](Lightweight.assets/image-20210315213029425.png)

**[img. edge intelligence의 내부 분류]**

1) Edge training

Edge device에서 모델을 학습하는 것,

아직 Edge device 들의 성능이 떨어져 산업에서 사용하진 않는다.

![image-20210315214625508](Lightweight.assets/image-20210315214625508.png)

**[img. Inference vs Training에 드는 비용 비교]**

2) Edge caching

Edge가 처리하기 힘들지만 외부에서 연결을 통해 가져오기 힘든 정보를 연결없이 저장공간에 저장, 또는 불러오기 

이번 강의에서 다루진 않음

컴퓨터 구조에서 메모리에서 hit과 miss에 대한 개념을 비슷하게 사용

3) Edge offloading

Edge와 가까운 곳에 있는 Edge 서버(클라우드와 Edge의 중간 형태)로부터 연결하여 정보를 가져옴, 하드웨어와 비슷한 개념

4) Edge Inference

수업에서 주로 다룰 분야, Edge에서 Output을 출력

![image-20210315214927165](Lightweight.assets/image-20210315214927165.png)

**[img. model 설계의 컴파일 수준]**

## 동전의 뒷면(The flip-side of the coin: on-device AI)