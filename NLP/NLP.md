[TOC]

# NLP

## Intro to Natural Language Processing(NLP)

- 컴퓨터가 주어진 단어나 문장, 문단, 글을 이해하는 NLU(Natural Language Understading)
- 이런 한 글을 생성할 수 있는 NLG(Natural Language Generation)로 이루어짐

- NLP의 영역(ACL, EMNLP, NAACL 등에서 연구)
  - 우리가 주로 다룰 분야
  - Low-level parsing 
    - Tokenization: 문장을 이루는 각 단어를 정보 단위(Token)로 쪼개나가는 것
    - stemming: 단어의 다양한 표현 변형을 없애고 의미만 남기는 것
      - 맑고, 맑지만, 맑았는데, 맑은 = 맑다.
  - word and phrase level
    - Named entity recognition(NER) : 고유 명사 인식 
      - (Newyork times는 newyork + times가 아니다)
    - part-of-speech(POS) tagging : 단어의 품사, 성분이 무엇인가?(명사, 목적어, 형용사 등)
  - Senetence level
    - 문장의 감정분석, 기계번역
  - Multi-sentence and paragraph level
    - Entailment prediction : 두 문장의 모순관계, 논리적 내포를 확인
      - 문장이 앞서 했던 말과 다른가?
    - 질의 응답, 문서 요약
    - dialog systems: 챗봇

- Text mining의 영역(KDD, WSDM, WWW 등에서 연구중)
  - 트랜드 분석, 상품 반응 분석, SNS 사회과학 분석 등

- Information retrieval(검색 기술)
  - 많이 성숙해진 상태, 상대적으로 느림,
  - 추천 시스템, 개인화 광고 등

### NLP의 트랜드

- 문장의 단어들을 vector로 표현하여 그래프 내의 점으로 바꾸어 처리하게 됨(Word Embedding).
- 자연어 처리에서 RNN 모델을 위한 LSTM, GRU 유닛이 사용되다 최근 구글의 attention module과 Transformer model의 발표로 성능이 크게 늘어남.
- 더이상 rule base 기계 번역은 딥러닝 base 기계번역에 성능으로 뒤쳐지고 있음.
- 최근에는 자가 지도 학습 또는 pretrained model을 통하여 더 이상 labeling이 필요하지 않은 BERT, GPT-3 같은 기술이 나타남.
- 다만 이러한 기술 들은 대규모의 컴퓨팅 능력과 데이터가 필요함.

### Bag-of-Words and NaiveBayes Classifier

- Bag-of-Words는 딥러닝 기반 이전 시대 때, 단어와 문서를 숫자형태로 나타내는 표현 기법
- NaiveBayes Classifier : Bag-of-Words 방식을 이용한 전통적인 문서 분류 기법

#### Bag-of-Words Representation

- Bag-of-Words의 과정

  1. 주어진 문장의 단어가 각각 포함되어있는 사전(vocabulary) 생성
     - "Jon really really loves this movie", "Jane really likes this song"에서
     - Vocabulary: {"John", "really", "loves",  "this","movie","Jane","likes","song"} 생성

  2. 사전에 포함된 유일한 단어들을 one-hot vector로 인코딩
     - Vocabulary: {"John", "really", "loves",  "this","movie","Jane","likes","song"} 
     - 차례대로 one-hot vector 형식으로 인코딩
       - 예를 들어  John:[1 0 0 0 0 0 0 0], really: [0 1 0 0 0 0 0 0] ... song: [0 0 0 0 0 0 0 1]
       - word- embedding 방법과의 차이점
     - 모든 vector쌍의 거리는 $\sqrt 2$, cosine similarity 는 0으로 계산된다.

- 이렇게 구성된 문장이나 단어는 one-hot vector들의 합으로 표현할 수 있다.

  - 예시 1. "John really really loves this movie"
    - John + really + really + loves + this + movie: [1 2 1 1 1 0 0 0]
  - 예시 2. "Jane really likes this song"
    - Jane + really + likes + this + song: [0 1 0 1 0 1 1 1]

#### NaiveBayes Classifier for Document Classification

- NaiveBayes Classifier란, Bag of words 방법으로 표현된 문서를 구분할 수 있는 전통적인 방법
- d개의 문서와 C의 문서분류 집합이 있다고 가정하고, c 분류는 C 문서분류 집합의 원소일 때, 각각의 클래스에 d문서가 속할 확률 분포는 아래와 같이 표현된다.

$$
C_{MAP}=argmax\ P(c|d) :"maximum\ a\ posteriori" = most\ likely\ class \\ = argmax\ \frac{P(d|c)P(c)}{P(d)}: Bayes\ Rule \\ = argmax\ P(d|c)P(c):Dropping\ the\ denominator\\
where\ c \in C
$$

**[math. Bayers' Rule Applied to Documents and Classes ]**

- P(d)는 d document가 뽑힐 확률 이며, 고정된 값으로 볼 수 있어서 무시할 수 있음

- 단어들 w로 이루어진 c 분류의 d 문서에서 특정 분류 c가 고정일 때, 문서 d가 나타날 확률은
  - $P(d|c)P(c) = P(w1,w2,\dots,w_n|c)P(c)\rightarrow P(c)\prod_{w_i\in W}P(w_i|c)$
  - 각 단어(w~1~,w~2~,...,w~n~)가 나타날 확률을 독립적이라고 가정

##### NaiveBayes 분류 예시

|          | Doc(d) | Document(words, w)                                   | Class(c) |
| -------- | ------ | ---------------------------------------------------- | -------- |
| Training | 1      | Image recognition uses convolutional neural networks | CV       |
|          | 2      | Transformer can be use for image classification task | CV       |
|          | 3      | Language modeling uses transformer                   | NLP      |
|          | 4      | Document classification task is language task        | NLP      |
| Test     | 5      | Classification task uses transformer                 | ?        |

**[fig. NaiveBayes 예시]**

- 위 표의 상황에서 class의 속할 확률은
  - $P(c_{cv})=\frac{2}{4}=\frac{1}{2}$
  - $P(c_{NLP}) = \frac {2}{4}=\frac {1}{2}$
- 이며, 이때 각 단어가 나타날 확률은
-  $P(w_k|c_i)=\frac {n_k}{n}$, where n~k~ is occurreneces of w~k~ in documents of topic c~i~

| Word                             | Prob            | Word                              | Prob            |
| -------------------------------- | --------------- | --------------------------------- | --------------- |
| $P(w_{"classification"}|c_{CV})$ | $\frac {1}{14}$ | $P(w_{"classification"}|c_{NLP})$ | $\frac {1}{10}$ |
| $P(w_{"task"}|c_{CV})$           | $\frac {1}{14}$ | $P(w_{"task"}|c_{NLP})$           | $\frac {2}{10}$ |
| $P(w_{"uses"}|c_{CV})$           | $\frac {1}{14}$ | $P(w_{"uses"}|c_{NLP})$           | $\frac {1}{10}$ |
| $P(w_{"transformer"}|c_{CV})$    | $\frac {1}{14}$ | $P(w_{"transformer"}|c_{NLP})$    | $\frac {1}{10}$ |

**[fig. 각 단어가 class 내에 나타날 확률들]**

- document d~5~="Classification task uses transformer"를 통하여 class에 속할 확률을 구해보면
  - $P(c_{CV}|d_5)=P(c_{CV})\prod_{w\in W}P(w|c_{CV})=\frac{1}{2}\times \frac{1}{14}\times \frac{1}{14}\times \frac{1}{14}\times \frac{1}{14}\times = 0.000013$
  - $P(c_{NLP}|d_5)=P(c_{NLP})\prod_{w\in W}P(w|c_{NLP})=\frac{1}{2}\times \frac{1}{10}\times \frac{2}{10}\times \frac{1}{10}\times \frac{1}{10} = 0.0001$
- 여기서 가장 높은 확률인 분류를 뽑아 확정하게 된다(argmax).
- 다만 학습 데이터에 없는 단어가 포함될 경우 다른 단어들이 있더라도 확률이 0이 되버리므로 regularization 같은 다른 방법을 강구해야함.

### Word Embedding: Word2Vec, GloVe

- Word2Vec과 GloVe는 하나의 차원에 단어의 모든 의미를 표현하는 one-hot-encoding과 달리 단어의 distributed representation을 학습하고자 고안된 모델.
- 문장을 단어라는 단위 정보로 이루어진 sequence data라고 가정했을 때, 각 단어들을 공간상의 한 점이나 벡터로 표현하는 기술을 Word Embedding이라고 한다.
  - 예를 들어 cat과 kitty는 비슷한 의미를 가지므로 비슷한 좌표값이나 벡터를 가지고 있겠지만, hamburger는 전혀 다른 좌표나 벡터에 있을 것이다.

#### Word2Vec

![image-20210215225944488](NLP.assets/image-20210215225944488.png)

**[img. Word2Vec 의 예시]**

- 같은 문장에 자주 포함되는 단어들은 관련성이 많다는 가정 하에  Word Embedding을 진행하는 알고리즘
  - 예를 들어 "The furry cat hunts mice." 에서 cat은 furry하고 mice를 hunts 하므로 연관이 있다.

- 문장 내의 단어 w가 나타날 때, 그 주위에 각각의 단어가 나타날 확률분포를 구하여 이용한다.
- Skipgram 방법과 CBOW 방법 두가지로 나뉨 (실습 2_word2vec.ipynb) 참조

##### Word2Vec Algorithm과 예제

1. 주어진 문장을 Tokenization한 후, 사전(Vocabulary) 생성하고 단어별 one-hot 벡터 생성
   - Sentence: "I study math."
   - Vocabulary: {"I", "study", "math"}
   - Input: "study" [0, 1, 0]
   - Output: "math" [0, 0, 1]
2. Sliding Window 기법을 이용해 앞 뒤로 나타난 단어들의 쌍들로 학습 데이터 구성.

![image-20210216024706818](NLP.assets/image-20210216024706818.png)

**[img. Window size가 1인 Sliding Window 기법의 그림]**

3. 2 layer 구성의 심층 신경망을 통해 word embedding 한다.

   - Input layer, output layer 차원 수 : vocabulary 단어 수(one-hot vector 차원)

   - hidden layer 차원 수 : hyper parameter(word embedding 차원 수)

   ![image-20210216005047259](NLP.assets/image-20210216005047259.png)

   **[img. 2 layer 신경망의 word embedding 원리]**

   - input vector "study"의 경우, [0,1,0]의 벡터를 가지고 있으며, hidden layer의 차원수가 2라고 가정할 때.
   - linear transform matrix W~1~의 경우 3차원의 벡터를 2차원의 벡터로 바꿔야 하므로 2X3 차원을 가진다.
     - W~1~의 input vector가 one-hot vector 이므로 내적을 구하기 보단 one-hot vector의 인덱스에 해당하는 W~1~의 Column을 가져오는 형식으로 계산한다.
   - linear transform matrix W~2~의 경우 다시 3차원의 벡터를 가져야 하므로 3X2 차원을 가진다.
     - softmax 함수를 통과시키기 전의 이상적인 logic값은 ground-truth의 내적값은 $\infin$, 그 이외의 내적 값은 $-\infin$이 되어야 결과값도 one-hot vector의 형태로 나온다. 

4. softmax 함수를 통과시켜 word embedding 값을 가져온다.

****

- 이를 통하여 의미론적 관점에서 단어간의 유사도를 알 수 있는 word vector를 구할 수 있다.

![image-20210216020142926](NLP.assets/image-20210216020142926.png)

**[img. 단어 간의 관계가 비슷하면 두 단어의 벡터의 방향도 유사하다.]**

- https://ronxin.github.io/wevi/ 에서 Word2Vec을 시연해볼 수 있다.
- https://word2vec.kr/search/ 에서 한국어 Word2Vec 결과값을 알아볼 수 있다.
- Word2Vec으로 문맥에 어색한 단어를  찾아내는 Intrusion Detection 또한 가능하다.
- Word2Vec을 이용해 다음과 같은 분야에 활용해 성능을 향상시킬 수 있다.
  - 기계번역
  - PoS tagging
  - 고유명사 태깅
  - 감정 분석
  - Image Captioning
  - 기타 등등

#### GloVe: Another Word Embedding Model

- Global Vectors for Word Representation

- Word2Vec과 함께 많이 쓰이는 Word Embedding 방법
- 카운트 기반 방법론(LSA)과 예측 기반의 방법론(Word2Vec) 두 가지를 모두 사용하는 방법론
  - (입력어의 임베딩 벡터와 출력어의 임베딩 벡터의 내적값)과 (윈도우에서 두 단어 i, j의 동시 출연 빈도에 log를 씌운 것)을 loss 함수로써 fitting하여 word embedding 값을 구하는 방식
  - $u_i$: 입력어의 임베딩 벡터, $v_j$: 출력어의 임베딩 벡터, $P_{ij}$: 윈도우 기반 두 단어 i, j의 동시 등장 빈도

$$
J(\theta)=\frac{1}{2}\sum^w_{i,j=1}f(P_{ij})(u_{i}^Tv_j-logP_{ij})^2
$$

​		**[math. GloVe의 손실함수]**

- 윈도우 기반 동시 등장 빈도($P_{ij}$)는 전체 단어 집합 들의 단어들이 윈도우 크기 내에서 단어가 등장한 횟수를 의미하며, 보통 전체 단어들의 등장 빈도를 행렬로 다음과 같이 표현한다.
  - "I like deep learning", "I like NLP", "I enjoy flying" 세 문장이 주어지고 윈도우 크기가 1일 때,

| 카운트   | I    | like | enjoy | deep | learning | NLP  | flying |
| :------- | :--- | :--- | :---- | :--- | :------- | :--- | :----- |
| I        | 0    | 2    | 1     | 0    | 0        | 0    | 0      |
| like     | 2    | 0    | 0     | 1    | 0        | 1    | 0      |
| enjoy    | 1    | 0    | 0     | 0    | 0        | 0    | 1      |
| deep     | 0    | 1    | 0     | 0    | 1        | 0    | 0      |
| learning | 0    | 0    | 0     | 1    | 0        | 0    | 0      |
| NLP      | 0    | 1    | 0     | 0    | 0        | 0    | 0      |
| flying   | 0    | 0    | 1     | 0    | 0        | 0    | 0      |

**[fig. 예제의 윈도우 기반 동시 등장 행렬(Window based Co-occurrence Matrix) https://wikidocs.net/22885]**

- 중복 되는 계산이 적어 상대적으로 빠를 수 있고, 적은 데이터로도 성능이 좋다.(실제 성능은 비등비등)
- https://nlp.stanford.edu/projects/glove/ 오픈소스 glove 모델

## Recurrent Neural Networks(RNNs)

### Basics of Recurrent Neural Networks(RNNs)

![image-20210216101229026](NLP.assets/image-20210216101229026.png)

**[img. RNN의 구조, 좌측은 Rolled RNN, 우측은 UnRolled RNN이다.]**

- Sequence 입력 벡터 X과 이전 time step의 RNN 모듈에서 계산한 $h_{t-1}$을 입력으로 받아, 현재의 $h_t$를 출력하는 구조.

  | 수식과 구조                                                  | 구성                                                         |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ![image-20210217032720485](NLP.assets/image-20210217032720485.png) | $$h_{t-1}:\ 이전\ hidden-state\ vector\\x_t:\ 해당 모듈의\ input\ vector\\h_t:\ 생성된\ hidden-sate\ vector\\\cdot 최종\ output(y_t)\ 출력시\ h_t를\ 바탕으로 계산  \\f_W:\ 파라미터\ W를\ 포함한\ RNN 함수\\y_t:\ 해당\ 모듈의\ output\ vecotr$$ |

  **[fig. RNN에서의 hidden state 계산과 구성 요소]**


- 매 time step 마다 같은 함수와 Parameter 구성을 공유함

$$
h_t = f_W(h_{t-1},x_t)\\
\downarrow\\
h_t=tanh(W_{hh}h_{t-1}+W_{xh}x_t)\\
y_t =W_{hy}h_t
$$

**[math. RNN의 자세한 $f_W$함수]**

- Hidden State Vector($h_t$)의 차원수는 Hyper parameter이다.

  - 즉, 개발자 판단하에 주어져야 함.

  ![image-20210217040854308](NLP.assets/image-20210217040854308.png)
**[img. $W$의 차원 수는 $h_{t-1}$ 차원수 X ($h_{t-1}$ 차원수 + $x_t$ 차원수)가 된다.]**
  
- 마지막으로 output $y_t$의 경우 binary classification이면 1차원으로 바꾼 뒤 sigmoid 함수를 이용하며, Multi Class classification이면 n개의 차원으로 바꾼 뒤 SoftMax Layer를 통과 시킨다. 

### Types of RNNs

![image-20210216102655561](NLP.assets/image-20210216102655561.png)

**[img. 여러 종류의 RNN 구조]**

- one-to-one 구조 

  - 키, 몸무게, 나이를 통해 저혈압, 정상혈압, 고혈압을 판단
  - Time Step이 존재하지 않음

- one-to-many 구조
  
  - Input이 첫 time step에 한번, 출력은 매번,
  - 나머지 Input은 무의미한 같은 차원의 zero vector가 들어간다.
  - Image Captioning
  
- many-to-one 구조
  
  - Input은 매번, Output은 마지막 한번
  
  -  감정 분석
  
- sequence-to-sequence 구조 1
  
  - Input이 일부 time step에, output이 또 일부 time step에 존재
  - 기계번역
  
- sequence-to-sequence 구조 2
  
  - Input과 Output이 매번 존재
  - video classification per frame

### Character-level Language Model

- Language Model이란, 문자열이나 단어를 입력받고, 그 다음에 나올 문자나 단어를 출력하는 모델

#### RNN 예시(hello Language Model)

1. 사전(Vocabulary) 구축 및 one-hot vector 생성

- 사전(Vocabulary)
  - 주어진 문장에서 정보 단위(단어 또는 여기서는 글자)들이 유일하게 이루어진 사전을 구축한다.
  - 예시의 "hello"의 경우 "[h, e, l, o]"라는 Vocabulary가 구축됨.
- one-hot vector 생성
  - index 부분이 1인 사전의 갯수 만큼의 dimension을 가지는 one-hot vector 생성
  - 마지막 "o"의 경우 그 뒤로 예측해야할 필요 없으므로 생성하지 않아도 좋다. 

![image-20210216104022300](NLP.assets/image-20210216104022300.png)

**[img. hello의 one-hot vector]**

2. 선형결합과 비선형 함수 tanh를 이용한 $h_t$ 학습

   ![image-20210216104043006](NLP.assets/image-20210216104043006.png)

   **[img. $h_t$의 dimension이 3이라고 가정하고 학습]**

   - $h_t=tanh(W_{hh}h_{t-1}+W_{xh}x_t+b)$ b: bias
   - 최초의 RNN 모델의 경우 $h_{t-1}$입력이 없으므로 h와 같은 차원의 zero vector를 채워준다.
   - 자세한 방법은 위의 RNN의 구조 참조

3. output 출력

![image-20210216104133143](NLP.assets/image-20210216104133143.png)

**[img. output vector 출력]**

- 각 RNN 모듈의 output은 hidden state($h_t$)에 $W_{hy}$를 곱한 후 bias를 더해서 구하게 된다. 
  - $Logit = W_{hy}h_t+b$
- 이후 Output을 softmax 함수에 통과 시켜 가장 큰 값을 결과값으로 확정한다.
- 이후 ground-truth(실제 값)의 one-hot vector와 비교하여 loss를 줄이는 방향으로 학습시킨다. 

4. 학습이 끝난 뒤 Test inference 시행

   - 최초의 Time step에만 입력을 넣어주고, 이후 Time step 부터는 이전 모듈의 output을 Input으로 넣어주어 추론한다.

   - 다음날 주식을 예측하는 RNN 모델이 그 다음날, 다음 다음날에도 예측 가능한 이유

![image-20210216104612495](NLP.assets/image-20210216104612495.png)

**[img. 출력을 다음 모듈의 input으로 넣어주는 Inference 구조]**

#### RNN 예제

![image-20210217105519872](NLP.assets/image-20210217105519872.png)

**[img. 셰익스피어 희곡 생성]**

![image-20210217105602228](NLP.assets/image-20210217105602228.png)

**[img. 영화 대본 생성]**

![image-20210217105630027](NLP.assets/image-20210217105630027.png)

**[img. C언어 code 생성]**

#### Backpropagation through time (BPTT)

-  모든 RNN Time step에 대해 Backpropagtion 또는 ForwardPropagtion을 진행하는 것은 성능상 한계가 있다. 

![image-20210216110815279](NLP.assets/image-20210216110815279.png)

**[img. RNN에서의 Forward와 backpropagation]**

- Truncation : 전체를 propagation 하면 너무 느리므로 구간 별(chunks of the sequence)로 잘라서 한다.

![image-20210216110934728](NLP.assets/image-20210216110934728.png)

**[img. Truncation  : 구간별로 잘라서 propagation]**

- Hidden state Vector($h_t$)는  이전 처리 결과, 문맥 등의 필요한 정보를 포함하고 있다.
  - 이러한 Hidden State Vector의 변화를 분석하는 것으로 RNN의 특성을 알 수 있다.

![image-20210217115424822](NLP.assets/image-20210217115424822.png)

**[img. Hidden state의 변화를 시각화한 것.]**

#### Vanishing/Exploding Gradient Problem in RNN

- RNN에는 Long-term problem과 Vanishing/Exploding Gradient Problem을 가지고 있다.
  - 가중치가 계속 곱해져서 Gradient가 기하급수적으로 커지거나 작아지는 문제

| math                                                         | propagation                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $h_t = tanh(w_{xh}x_t+w_{hh}h_{t-1}+b), t=1,2,3\\For\ w_{hh}=3,w_{xh}=2,b=1\\h_3=tanh(2x_3+3h_2+1)\\h_2=tanh(2x_2+3h_1+1)\\h_1=tanh(2x_1+3h_0+1)\\\dots\\h3=tanh(2x_3+3tanh(2x_2+3h_1+1)+1)$ | ![image-20210216111919764](NLP.assets/image-20210216111919764.png) |

**[fig. hidden state 중첩에 따른 $W_{hh}$의 중첩 ]**

![image-20210216111919764](NLP.assets/vanishing gradient.gif)

**[gif. LSTM과 RNN의 cell 진행에 따른 gradient 감소 비교]**

### Long Short-Term Memory(LSTM) and Gated Recurrent Unit(GRU)

- 기존의 RNN 모델에 비해 Vanishing/Exploding gradient 문제를 해결하고, 성능 상에 더 좋은 결과를 보인다.

#### Long Short-Term Memory(LSTM)

![image-20210216113818583](NLP.assets/image-20210216113818583.png)\

**[img. LSTM 구조의 도식화]**

- 정보를 좀 더 오래 남기게 하고 기존의 RNN 모델을 개선한 모델.
- Vanishing/Exploding  Gradient Problem, Long-Term dependency 문제를 해결한 모델
- Hidden state vector($h_t$)를 기억 소자로 보고, 단기 기억을 길게 기억하도록 개선하여 LSTM이라고 이름 붙임.
- 이전 Time step에서 넘어오는 정보(Cell state($C_t$),  Hidden state($h_{t-1}$) )가 2개이며, 총 Input은 3개이다. 
  - $\{C_t, h_t\}= LSTM(x_t,C_{t-1},h_{t-1})$
  - $C_t$ : Cell state vector, 핵심 정보
  - $h_{t-1}$: Hidden state vector, Cell state 정보를 필요한 것만 Filtering 한 정보.

#### Long short Term Memory의 구성과 동작 과정

- 입력을 받은 후 선형 변환 후 나온 결과물에 4개의 activation 함수를 통과 시켜 gate값들을 만든다.
- 이렇게 나온 gate 들은 Cell state 및 Hidden State를 계산할 때의 중간 결과물 역할을 한다.

| LSTM 구성                                                    | 수식                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20210216115034645](NLP.assets/image-20210216115034645.png) | $\begin{pmatrix}i \\f\\o\\g \end{pmatrix}=\begin{pmatrix}\sigma \\\sigma\\\sigma\\tanh \end{pmatrix}W\begin{pmatrix}h_{t-1}\\x_t\end{pmatrix}\\c_t=f\odot c_{t-1}+i\odot g\\h_t=o\odot tanh(c_t)$ |

**[fig. LSTM 내부의 연산, $x_t, h_t$의 dimension을 h라고 가정]**

- i: Input gate, Whether to write to cell
- f: Forget gate, Whether to erase cell
- o: Output gate, How much to reveal cell
  - sigmoid를 통해 나오는 Input gate, Forget gate, Output gate는 0~1사이를 가지며 다른 벡터와 곱해져 0~1 사이의 일부로 축소해주는 역할을 함(gate)
- g: Gate gate, How much to write to cell

#### Gate 들의 구체적인 예시

- gate들은 이전의 $c_{t-1}$을 적절하게 가공하는 역할을 함

1. Forget gate

| 수식                                      | 도식                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| $f_t=\sigma(W_f \cdot [h_{t-1},x_t]+b_f)$ | ![image-20210216135905387](NLP.assets/image-20210216135905387.png) |

**[fig. forget gate 수식과 그림]**

- forget gate는 sigmoid 함수를 통해 $h_{t-1}$과 $x_t$의 일부 데이터를 축소하는 역할

2. Gate gate & Input gate

| 수식                                                         | 도식                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $i_t=\sigma(W_i\cdot [h_{t-1},x_t]+bi)\ (input\ gate) \\\widetilde{C_t}=tanh(W_c\cdot [h_{t-1},x_t]+b_C)\  (gate\ gate)\\C_t=f_t\cdot C_{t-1}+i_t\cdot \widetilde{C_t}\  (input\ gate \times gate\ gate)$ | ![image-20210216135817334](NLP.assets/image-20210216135817334.png) |

**[fig. Input gate와 Gate gate의 수식과 그림]**

- Input gate와 Gate gate의 결과물을 곱한 뒤, 해당 결과물을 Forget gate와 이전 Cell state를 곱한 것을 더하여 현재의 Cell state를 만들게 된다.
  - 2번의 선형변환를 거친 현재 입력정보($h_{t-1}, x_t$)를 이전 Cell state와 합쳐 새로운 state를 만드다는 의미이다.

3. Output gate

| 수식                                                         | 도식                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $o_t=\sigma(W_o[h_{t-1},x_t]+b_o) \\ h_t=o_t\cdot tanh(C_t)$ | ![image-20210216135945524](NLP.assets/image-20210216135945524.png) |

**[fig. Output gate의 수식과 그림]**

- 생성한 Output gate의 결과를 tanh함수를 통과시켜, 현재 필요한 정보만 filtering한 뒤, 현재 Cell state와 곱하여  Hidden state를 만든다.
- tanh 함수를 미분할 시 미분값이 sigmod에 비해 커서 vanishing/exploding gradient problem을 해결할 수 있다.

#### Gated Recurrent Unit(GRU)

![image-20210216133902631](NLP.assets/image-20210216133902631.png)

**[img. GRU 구조]**

- LSTM을 간소화하여 더욱 적은 메모리 요구량과 빠른 계산 시간을 가능하게 한 모델
- LSTM과 달리 Cell state가 존재 하지 않고, hidden state 벡터 하나만 다음 Time step으로 보낸다.
- LSTM과 더불어 많이 사용되며 성능상 뒤지지 않으면서 빠르게 계산이 가능하다.

- GRU의 $h_{t-1}$이 LSTM의 Cell state 이전 time step 들의 정보와 결과값을 함께 포함한다.
  - $z_t=\sigma(W_z\cdot [h_{t-1},x_t])$ (Input gate)
  - $r_t=\sigma(W_r\cdot [h_{t-1},x_t])$
  - $\widetilde{h_t}=tanh(W\cdot[r_t\cdot h_{t-1},x_t])$ (LSTM의 Gate gate 역할)
  - $h_t=(1-z_t)\cdot h_{t-1}+z_t\cdot \widetilde{h_t}$
  - c.f) $C_t=f_t\cdot C_{t-1}+i_t\cdot \widetilde{C_t}$ in LSTM, 
    - Input gate와 Forget gate의 대신인 1 - Input gate한 값으로 hidden state를 구한다.(가중 평균의 형태)
    - 내부적으로 1개의 gate로 통합하면서 계산량을 줄임

##### Backpropagation in GRU

![image-20210216134033070](NLP.assets/image-20210216134033070.png)

**[img. GRU propagation]**

- 곱셈이 아니라 덧셈으로 연산해서 Vanishing/Exploding gradient problem을 해결함
  - $h_t=(1-z_t)\cdot h_{t-1}+z_t\cdot \widetilde{h_t}$
- Long term dependency 문제 해결

#### Summary on RNN/LSTM/GRU

- RNN은 다양한 길이를 가질 수 있는 Sequence data에 특화된 유연한 형태의 딥러닝 모델 구조
- Vanilla RNN은 간단한 구조지만 학습시 문제가 많고 학습이 잘 안된다.
- LSTM과 GRU는 long term problem과 vanishing\exploding gradient 문제를 해결했음
- LSTM과 GRU는 덧셈 기반 학습? BPTT