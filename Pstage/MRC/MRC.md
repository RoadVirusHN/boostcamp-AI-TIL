# 기계 독해(MRC)

> Naver AI boostcamp 기계 독해 강의를 정리한 내용입니다.

## 1. MRC Intro & Python Basics

### Introduction to MRC

**Machine Reading Comprehension(MRC. 기계독해)란?**

![MRC의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426111436197.png)

주어진 지문(Context)를 이해하고, 주어진 질의 (Query/Question)의 답변을 추론하는 문제



![Extractive Answer Datasets](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426111736186.png)

1) Extractive Answer Datasets

: 질의 (qeustion)에 대한 답이 항상 주어진 지문 (context)의 segment (or span)으로 존재

MRC 종류로 Cloze Tests(CBT), Span Extraction(SQuAD, KorQuAD, NewsQA) 등이 존재



![Descriptive/Narrative Answer Datasets, MS MARCO](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426112238977.png)

2) Descriptive/Narrative Answer Datasets

: 답이 지문 내에서 추출한 span이 아니라, 질의를 보고 생성된 sentence (or free-form)의 형태

MRC 종류로 MS MARCO, Narrative QA 등이 존재



![Multiple-choice Datasets, MCTest](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426112702926.png)

3) Multiple-choice Datasets

: 질의에 대한 답을 여러 개의 answer candidates 중 하나로 고르는 형태

MRC 종류로 MCTest, RACE, ARC, 등이 존재

![다양한 종류의 Dataset들](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426144735600.png)

**MRC의 해결해야할 점**

![paraphrasing과 Coreference Resolution](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426151837948.png)

1. 단어들의 구성이 유사하지는 않지만 동일한 의미의 문장을 이해 (paraphrasing)

2. 대명사가 지칭하고 있는 것이 무엇인가?(Coreference Resolution)
3. 답변이 존재하지 않는 경우는?

4. 여러 문서에서 supporting fact를 찾아야 답을 알 수 있는 경우(Multi-hop reasoning)

**MRC의 평가 방법**

1)  Exact Match / F1 Score : For extractive answer and multiple-choice answer datasets

- Exact Match (EM) or Accuracy : 예측한 답과 Ground-truth가 정확히 일치하는 샘플의 비율, (Number of correct samples) / (Number of whole samples)
- F1 Score : 예측한 답과 ground-truth 사이의 token overlap을 F1으로 계산
  - F1 = $\frac{2\times Precision \times Recall}{Precision+Recall}$, $Precision=\frac{num(same\ token)}{num(pred\ tokens)}$, $Recall=\frac{num(same\ token)}{num(ground_tokens)}$

예를 들어, 답이 5 days, 예측이 for 5days 이면, EM은 0이지만 F1: 0.8이다.

2) ROUGE-L / BLEU : For descriptive answer datasets, Ground-truth와 예측한 답 사이의 overlap을 계산

- ROUGE-L Score : 예측한 값과 ground-truth 사이의 overlap recall (ROUGE-L => LCS (Longest common subsequence) 기반)
- BLEU (Bilingual Evaluation Understudy) : 예측한 답과 ground-truth 사이의 Precision (BLEU-n => uniform n-gram weight)

### Unicode & Tokenization

Unicode란, 전 세계의 모든 문자를 일관되게 표현하고 다룰 수 있도록 만들어진 문자셋으로, 각 문자마다 숫자 하나에 매핑됨.

![about unicode](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426141501946.png)

인코딩이란, 문자를 컴퓨터에서 저장 및 처리할 수 있게 이진수로 바꾸는 것

UTF-8(Unicode Transformation Format)란, 현재 가장 많이 쓰는 인코딩 방식, 문자 타입에 따라 다른 길이의 바이트를 할당한다.

1 byte: Standard ASCII

2 bytes: Arabic, Hebrew, most European scripts

3 bytes: BMP(Basic Multilngual Plane) - 대부분의 현대 글자 (한글 포함)

4 bytes: ALL Unicode characters - 이모지 등

한국어의 경우, 모든 한글 경우의 수를 따진 완성과, 조합하여 나타낼 수 있는 조합형으로 나뉘어 분포되어 있다.

![토크나이징 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426154514362.png)

토크 나이징은 텍스트를 토큰 단위로 나누는 것으로, 단어, 형태소, subword 등 여러 기준이 존재한다.

Subword 토크나이징은 자주 쓰이는 글자 조합은 한 단위로 취급하고, 자주 쓰이지 않는 조합은 subword로 쪼갠다. 

- ##은 디코딩 (토크나이징의 반대 과정)을 할 때 해당 토큰을 앞 토큰에 띄어쓰기 없이 붙인다는 것을 의미

BPE(Byte-Pair Encoding)은 데이터 압축용으로 제안된 알고리즘으로, NLP에서 토크나이징용으로 활발하게 사용되고 있다.

- 사람이 직접 짠 토크나이징보다 성능이 좋은 경우가 많음

1. 가장 자주 나오는 글자 단위 Bigram (or Byte pair)를 다른 글자로 치환한다.
2. 치환된 글자를 저장해둔다, 1로 다시 반복

ex) aaabdaaabac -> Z=aa, ZabdZabac -> Y=ab, Z=aa, ZYdZYac -> X=ZY, XdXac

### Looking into the Dataset

![KorQuAD](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426155330638.png)

KorQuAD란, LG CNS가 AI 언어지능 연구를 위해 공개한 질의응답/기계독해 한국어 데이터셋

- 인공지능이 한국어 질문에 대한 답변을 하도록 필요한 학습 데이터셋
- 위키피디아 문서 + 크라우드 소싱을 통해 제작한 질의응답 쌍으로 구성되어 있음
- 누구나 데이터를 내려받고, 학습한 모델을 제출하고 공개된 리더보드에 평가 받을 수 있음
- 2.0은 보다 긴 분량의 문서, 복잡한 표와 리스트 등을 포함하는 HTML 형태로 표현 

Title과 Context, Question-Answer Pairs로 이루어져 있다.

![KorQuAD의 데이터 수집 과정](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426162043978.png)

HuggingFace datasets 라이브러리에서 'squad_kor_v1', 'squad_kor_v2'로 불러올 수 있음

- 여러 라이브러리에 호환됨, memory-mapped, cached 등의 메모리 공간 부족, 전처리 과정 등을 피할 수 있음
- 기본적인 데이터셋 함수 구현되어 있음.

![korQuAD 질문 유형](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426155603301.png)

![KorQuAD 답변 유형](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210426155624334.png)

## Extraction-based MRC

### Extraction-based MRC

![Extraction-based MRC의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427094642197.png)

질문(question)의 답변(answer)이 항상 주어진 지문(context)내에 span으로 존재하는 문제

MRC 데이터 셋으로 SQuAD, KorQuAD, NewsQA, Natural Questions 등이 존재한다.

- Hugging Face Dataset에서 구하면 쉽다.

Text를 생성하는 것이 아니라, 답의 시작점과 끝점을 찾는 문제가 된다.

![F1 score 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427095523808.png)

분류(Classification) 문제와 비슷하며, 주로 이전에 배웠던 F1 score나 EM을 지표로 삼는다.

- 답이 여러개가 될 수 있을 때는 보통 가장 높은 것을 인정해준다.

### Pre-processing

다음과 같은 예시의 data를 전처리한다고 생각하자.

![MRC 입력 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427134801860.png)

**Tokenization 단계**

텍스트를 작은 단위(Token)으로 나누는 단계

띄어쓰기, 형태소, subword 등 여러 단위 토큰 기준이 사용되지만 최근 Out of Vcabulary(OOV) 문제를 해결하고 정보학적으로 이점을 가진 Byte Pair Encoding(BPE)을 주로 사용함

BPE 방법론에는 WordPiece Tokenizer가 존재.

"미국 군대 내 두번째로 높은 직위는 무엇인가?" => ['미국', '군대', '내', '두번째', '##로', '높은', '직', '##위는', '무엇인가', '?']

Tokenizing과 Speicial Token을 이용해 Tokenizing 하면 결과가 다음과 같다.

![Special Token을 포함한 데이터](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427135043627.png)

![실제 토근화된 결과](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427135126093.png)

**Attention Mask**

입력 시퀀스 중에서 attention을 연산할 때 무시할 토큰을 표시

0은 무시, 1은 연산에 포함되며, 보통 [PAD]와 같은 의미가 없는 특수토큰을 무시하기 위해 사용

![pad 토큰이 0으로 표시되어 무시된다.](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427135320797.png)

**input_ids 또는 input_token_ids**

![인덱스로 바뀐 질문](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427162608805.png)

입력된 질문의 형태를 인덱스의 형태로 바꾸어 학습을 용이하게 만든다.



**Token Type IDs**

입력이 2개 이상의 시퀀스(예: 질문 & 지문),일때, 각각에게 ID를 부여하여 모델이 구분해서 해석하도록 유도

![Context는 1로, 그외 질문이나 토큰을 0으로 바꾼 경우](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427135402202.png)



**모델 출력값**

정답은 문서내 존재하는 연속된 단어토큰(span)이므로, span의 시작과 끝 위치를 알면 정답을 맞출 수 있음

Extraction-based에서 답안을 생성하기 보다, 시작위치와 끝위치를 예측하도록 학습한. 즉 Token Classification 문제로 치환. 

![모델 출력의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427135519037.png)

### Fine-tuning

![Extraction-based MRC Overview](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427103633663.png)



BERT의 output vector의 각 element는 해당 token이 답의 시작 또는 끝일 확률을 나오게 만든다.

![Fine-tuning BERT](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427103609832.png)

### Post-processing

**불가능한 답 제거**

- End position이 start position보다 앞에 있는 경우 제거 (예상 답변 부분의 앞부분이 뒷부분 보다 뒤에 있을 경우)
- 예측한 위치가 context를 벗어난 경우 제거(ex) question이 있는 곳에 답이 태깅)

- 미리 설정한 max_answer_length 보다 길이가 더 긴 경우

**최적의 답 찾기**

1. Start/end position prediction에서 score(logits)가 가장 높은 N개를 각각 찾는다.

2. 불가능한 start/end 조합 제거

3. 가능한 조합들 중 score의 합이 큰 순서대로 정렬

4. Score가 가장 큰 조합을 최종 예측으로 선정

5. Top-k 가 필요한 경우 차례대로 내보낸다.





## Generation-based MRC

### Generation-based MRC

![Generation의 예시](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427104025265.png)

주어진 지문과 질의(question)을 보고, 답변을 생성하는 Generation 문제

지표는 동일하게 EM과 F1을 쓸 수 있다, 추가적으로 ulgi? blue? 같은 다른 지표를 사용할 수도 있다.

![Generation-based MRC의 overview](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427150536911.png)

Extraction base와 달리 seq 2seq 구조를 이용하며, output의 형태 또한 답의 위치가 아닌 생성된 답을 사용하게 된다.

![gen MRC vs EXT MRC](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427154709478.png)

### Pre-processing

오히려 Extraction base 보다 더욱 쉬워졌다.

입력의 경우, 동일하게 WordPiece Tokenizer를 활용하며, Special Token 사용이 조금 다르다.

Extraction-based MRC에선 CLS, SEP, PAD 토큰을 사용 했지만

Generation-based MRC에서는 토큰을 자연어를 이용한 텍스트 포맷으로 사용한다.

![Generation based MARC에서의 토큰 사용](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427163156309.png)

Attention mask는 이전 BERT와 같지만 우리가 사용할 Generation-based MRC를 위한 BART 모델의 경우 입력에 token_type_ids가 존재하지 않는다, 즉, 여러 입력 sequence 간의 구분이 없음.

- 직접 제공하지 않아도 모델이 충분히 구분 가능하고, 성능 차이가 없어서 지워짐.

![Generation base MRC의 Pretraining(좌)과 출력 형태](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427221135106.png)

### Model

![BART의 Encoder Decoder 구조](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427224631617.png)

**BART**

기계 독해, 기계 번역, 요약, 대화 등 sequence to sequence 문제의 pre-training을 위한 denoising autoencoder (noise(mask)를 없애는 방식으로 학습하는 것)

BART의 인코더는 BERT처럼 bi-directional이며, BART의 디코더는 GPT처럼 uni-directional(autoregressive)

- 아래 1일 경우 정보가 주워진다는 의미이다.

![BART의 인코더와 디코더](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427224423080.png)

**Pre-training BART**

![BART에서의 Pretraing](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427231032326.png)

BART는 텍스트에 노이즈를 주고 원래 텍스트를 복구하는 문제를 푸는 것으로 Pre-training함



### Post-processing

**Decoding**

![autoregressive Decoder](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427231613878.png)

디코더에서 이전 스텝에서 나온 출력이 다음 스텝의 입력으로 들어감, 이를 autoregressive라고 함

맨 첫 입력은 문장 시작을 뜻하는 스페셜 토큰

![Searching](C:\Users\roadv\Desktop\AI_boostcamp\BoostCamp AI TIL\Pstage\MRC\MRC.assets\image-20210427231721860.png)

output을 생성할 때는 대부분 Beam Search 방법을 사용한다.