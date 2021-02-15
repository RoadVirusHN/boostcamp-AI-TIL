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



#### Glove: Another Word Embedding Model

