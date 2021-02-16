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

  1. 주어진 문장의 단어가 각각 포함되어있는 사전 생성
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
- 여기서 가장 높은 확률인 분류를 뽑아 확정하게 된다.
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