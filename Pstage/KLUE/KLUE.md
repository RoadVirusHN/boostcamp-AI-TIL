# KLUE(한국어 언어 모델 학습 및 다중 과제 튜닝)

> Naver AI Boostcamp의 KLUE 강의를 정리한 내용입니다.

## 01. 인공지능과 자연어 처리
**인공지능과 자연어 처리에 대하여**

*인공지능* : 인간의 지능이 가지는 학습, 추리, 적응, 논증 따위의 기능을 갖춘 컴퓨터 시스템

*ELIZA (1966)* : 기계에 지능이 있는지 판별할 수 있는 튜링 테스트(이미테이션 게임)을 적용할 수 있는 최초의 Human-like AI

자연어 처리에는 문서 분류, 기계 독해, 챗봇, 소설 생성, 의도 파악, 감성 분석 등의 응용 분야가 있다.

![인간의 자연어 처리](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412102428484.png)

인간의 자연어 처리 (대화)의 경우

1. 화자는 자연어 형태로 객체를 인코딩
2. 메세지의 전송
3. 청자는 본인 지식을 바탕으로 자연어를 객체로 디코딩

하는 과정을 거친다.

![사람의 자연어 처리](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412102441208.png)

비슷하게 컴퓨터의 자연어 처리는

1. Encoder는 벡터의  형태로 자연어를 인코딩
2. 메세지의 전송
3. Decoder는 벡터를 자연어로 디코딩

하는 과정을 거친다.

즉, *자연어 처리는 컴퓨터를 이용하여 인간 언어의 이해, 생성 및 분석을 다루는 인공 지능 기술*

![분류로 생각할 수 있는 자연어 처리 문제](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412103028063.png)

대부분의 자연어 처리 문제는 분류 문제로 생각할 수 있으며, 분류를 위해 데이터를 수학적으로 표현되어야 한다.

이를 위해 가장 먼저,

1. 분류 대상의 특징 (Feature)을 파악(Feature extraction)
2. 특징을 기준으로 분류 대상을 그래프 위에 표현 뒤, 대상들의 경계를 수학적으로 나눔(Classification) 
3. 새로 들어온 데이터를 기존 Classification을 통해 그룹 파악 가능

![특징 추출과 분류](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412103743433.png)

과거에는 위의 과정을 사람이 직접 했지만 이제는 컴퓨터가 스스로 Feature extraction과 Classification 하는 것이 기계학습의 핵심이다.

![one-hot encoding 방식](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412103900112.png)

자연어를 좌표 평면 위에 표시하는 방법으로 one-hot encoding 방식이 있지만 단어 벡터가 sparse 해지므로 해당 방식은 의미를 가지지 못한다.

![Word2Vec](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412105044270.png)

Word2Vec 알고리즘은 자연어의 의미를 벡터 공간에 임베딩하여, 단어의 의미를 포함

- 비슷한 의미의 단어는 비슷한 Word2Vec 벡터를 가진다.
  - 예를 들어, '개'와 '고양이'는 '먹는다' 보다 비슷한 벡터를 가진다.

subword information : 서울시는 서울을 포함하는 단어이다.

- Word2Vec은 이러한 것을 무시함

Out of Vocabulary(OOV) : 학습하지 않은 단어는 전혀 예측할 수 없음.

![Word2Vec의 Skip-gram 방식](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412104332817.png)

Skip-gram 방식이라는 주변부의 단어를 예측하는 방식으로 학습한다. 



![다양한 용언의 형태](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412105232695.png)

위의 단점을 보완하기 위해 FastText라는 새로운 임베딩 방법이 존재

- Facebook에서 공개, word2vec과 유사하나 단어를 n-gram으로 나누어 학습 수행
- n-gram 2-5일시, "assumption" = {as,ss,su, ..., ass,ssu,sum, ump,...,assumption}
- 별도의 n-gram vector를 형성하며, 입력 단어가 vocabulary 사전에 있을 경우 word2vec과 비슷하게 return 하지만, OOV일 경우 n-gram vector 들의 합산을 return

![Orange의 n-gram vector](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412105840903.png)

FastText는 Word2Vec에 비해 오탈자, OOV, 등장 회수가 적은 학습 단어에 대해 강세

![FastText VS Word2Vect 비교](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412110218288.png)

단, Word2Vec이나 FastText 같은 word embedding 방식은 동형어, 다의어 등에 대해서 embedding 성능이 좋지 못하며, 주변 단어를 통해 학습이 이루어지므로 문맥 고려가 불가능하다는 단점이 있다.

**딥러닝 기반의 자연어 처리와 언어 모델**

*모델(model)*이란, 어떤 상황이나 물체 등 연구 대상 주제를 도면이나 사진 등 화상을 사용하거나 수식이나 악보와 같은 기호를 사용하여 표현한 것.

- 일기예보 모델, 데이터 모델, 비즈니스 모델, 물리 모델 등이 존재

자연 법칙을 컴퓨터로 모사함으로써 시뮬레이션 가능, 미래의 state를 올바르게 예측하는 방식으로 모델 학습이 가능



Markove 기반의 언어 모델, 혹은 마코프 체인 모델(Markov Chain Model)은 초기 언어 모델로, 다음의 단어나 문장이 나올 확률을 통계와 단어의 n-gram을 기반으로 계산

최근의 딥러닝 기반의 언어 모델은 해당 확률을 최대로 하도록 네트워크를 학습하며, RNN (Recurrent Neural Network) 기반의 언어 모델이 그러하다.

![RNN 노드](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412111606822.png)

RNN은 히든 노드가 방향을 가진 엣지로 연결돼 순환구조를 이룸(directed cycle)

이전 state 정보가 다음 state를 예측하는데 사용됨으로써, 시계열 데이터 처리에 특화

마지막 출력은 앞선 단어들의 문맥을 고려해 만들어진 최종 출력 vector (Context vector라고 함)이며, 이 위에 classification layer를 붙이면 문장 분류를 위한 신경망 모델이 됨.

![context vector 수에 따른 task](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412111820015.png)



**Seq2Seq(Sequence to Sequence)**

RNN 구조를 통해 Context Vector를 획득하는 Encoder와 획득된 Context vector를 입력으로 출력을 예측하는 Decoder layer를 이용한 RNN 구조

![Seq2Seq 구조](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412112852783.png)

Seq2Seq 구조는 모든 자연어처리 task에 적용가능하다.

단, RNN의 경우 입력 sequence의 길이가 매우 길 경우, 앞단의 단어의 정보는 희석이 되며, 고정된 context vector 사이즈로 인해 긴 sequence에 대한 정보를 함축하기 어려움

또한, 모든 token이 영향을 미치니, 중요하지 않은 token 도 영향을 줌

이를 방지하기 위해 Attention이 개발됨

**Attention**

문맥에 따라 동적으로 할당되는 encode의 Attention weight로 인한 dynamic context vector를 획득

이를 통해 Seq2Seq의 encoder, decoder 성능을 비약적으로 향상시킴, 단 여전히 느림



![seq2seq vs transformer](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412113814040.png)

Self-attention 구조의 경우, hidden state가 순차적으로 RNN에 전해지는 것이 아니라 모든 RNN이 서로의 output에 영향을 주게끔 설계되어 더욱 빠르다.

![Transformer 구조](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\KLUE\KLUE.assets\image-20210412114058591.png)

이를 응용한 구조가 Transformer 구조이며, 각자 따로 RNN 구조를 가지던 Seq2Seq model과 달리, 하나의 네트워크를 공유함.

이 Transformer 구조로 인해 다양한 구조와 언어 모델이 생성됨

