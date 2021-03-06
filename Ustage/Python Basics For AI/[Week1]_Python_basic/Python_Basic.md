# Python Basic

[TOC]
---
- 과거 정리했던 것 + 해당 강의에서 들은 것을 추가하였다.
## Python의 특징과 종류, 프레임워크 예
---
### Python의 특징
---

>_폭 넓은 사용자 층의 다양한 영역에서 사용하는 프로그래밍 언어, 여러 교육기관, 연구기관, 빅데이터 분석, 머신 러닝 등에 이용_

>_아름다운 것이 추한 것보다 낫다._
>_명시적인 것이 묵시적인 것보다 낫다._
>_단순한 것이 복잡한 것보다 낫다._
>_복잡한 것이 난해한 것보다 낫다._
>_가독성이 중요하다._

- 가독성을 중요시하며, 풍부한 라이브러리, 오픈소스이며, 유니코드[^1]를 기본으로 프로그램 패러다임을 지원한다.
-  C언어 같은 native 2진 파이 언어보다 성능상 뒤지지만, 컴퓨터 성능의 발달과 c언어 모듈화로 극복할 수 있다. 
- 들여쓰기로 코드블록을 구분을 강제하며, 한줄에 79글자를 넘지 않도록 권장한다. 스페이스바 4번이 1들여쓰기인 것이 관례이며, 알기쉽고 가독성 좋은 코드를 만들 수 있다.
- 보통 메인 코드 맨 위에 **\# coding : utf-8 ** 또는  # -\*- coding : utf-8 -\*-  를 적어 소스코드 인코딩을 알려준다
-  다른 언어와 달리 세미콜론[^6]을 사용하지 않아도 된다. 하지만 한줄에 2 코드 이상이며, 서로 구분되야 할 때는 써도 된다.
-  라이브러리 성격의 기능, 프로그램의 진입점 역할, .py로 끝나며, 원칙적으로 1 모듈 = 1.py 이다.

1. _`독립적`_ : 오픈소스이며 상업적 사용 가능
2. _`인터프리터 언어`_ : 실행시간에 명령을 해석해 실행
3. _`동적 언어`_ : 동적 타이핑[^2] 지원, 생성된 객체에 대한 메모리 관리는 Garbage Collector[^3]로 이루어짐
4. _`대화형 성격`_ : 대화형 실행으로 결과값과 버그를 바로 확인 가능
5. _`학습 용이성`_ : API[^4]를 많이 지원하고, 읽기 쉽고 직관적이며, 효율적인 코드를 짜는 고급 언어 
6. _`내장 스크립트 언어`_ : 언어로 쓰인 모듈을 연결하려는 목적, 스크립트 언어[^5]로 활용

---
### Python의 종류
---
#### 버전별
---
1. Python의 시작
1980년대 고안, 1991년 발표, 몬티파이선에서 따옴

2. Python 2.0
- 유니코드 지원, 가비지 콜레터 지원, 재단 설립 등- 

3.  Python 3.0
- __Python 2.0과 하위 호환되지 않음__
- 일부 자료형 구성요소, 내장자료형의 내부적인 변화 제거
-  표준 라이브러리 패키지 재배치
- 향상된 유니코드[^1] 지원
---
#### 구현 언어별
---
1. `CPython` : c로 작성된 파이썬, 표준
2. `IronPython` : .Net과 Mono용, C#으로 구현
3. `Jython` : 자바로 구현된 파이썬, 자바 가상머신에서 동작
4. `PyPy` : 파이썬으로 구현, Cpython보다 빠름

---

### 대표 프레임워크
---
>__Flask __: 간단한 웹 서비스 또는 모바일 서버 구축에 적합
>__Beautiful Soup __: 웹 크롤링 라이브러리, 데이터 수집 분야, 문서 수집, HTML 문서에 대한 구문 분석, DOM 트리 탐색등의 기능, 문서 분석 및 정보 추출 기능
>__Scrapy __: 웹 크롤링 프레임워크, 웹 문서에서 데이터 추출 규칙 작성 -> 문서 수집 및 필요 데이터 자동 추출
>__Numpy__ : 과학분야 컴퓨팅 패키지, 다차원 배열 객체, 선형대수, 푸리에 변환, 난수 생성 기능
>__pandas__ : 데이터 분석 시 사용하는 표준라이브러리, 데이터 구조, 분석도구 제공
>__SciPy.org__ : 수학, 과학, 엔지니어링 분야에서 활용,
>__scikit-learn__ : 데이터 마이닝과 데이터 분석을 위한 도구, 분류, 회귀 군집, 차원축소, 머신러닝 지원
>__tensorFlow__ : 구글이 공개한 머신러닝, 딥러닝 프레임 워크, GPU를 이용한 연산 지원, 신경망 모델을 쉽게 구현
>__PYTORCH__ : 머신러닝 및 딥러닝 프레임워크, GPU 연산, 간결한 코드, 신경망 모델, 빠른 모델 훈련, 결과값 시각화
>그외 __YOLO__, __Faker__ 등
---
## 개발환경과 코드 작성
---
###  통합개발환경 (IDE)[^7]
---
1. __오픈소스 IDE 들__
	Pycharm, Visual Studio Code, PyDev 등
2. __프로젝트란[^8]?__
3. __디버깅[^9]__이란?
---
### 주석 처리 방법
---
> 주석 처리 예제
```Python
# 주석 처리란?
'''
컴파일이 되지 않는 코드를 알려주어 제외하는 것,
주로 디버깅이나, 일부 코드를 잠시 지우거나, 메모를 하기 위해 사용
'''
# 1줄 주석처리 방법
'''
다중
라인
주석
처리 
방법
'''
#VS 코드 기준으로 원하는 코드를 블록으로 지정하고 Ctrl + /로 주석처리 on/off 가능
```
---
## 기초 문법의 이해와 문자열 포맷팅
---
### 자료형의 특징과 활용법
---
1. 리터럴(Literal)[^10]이란?
> 자료형의 예시들
```Python
# 각 자료형의 예시와 특징
#동적 타이핑 언어이므로  컴파일시 대입된 값에 따라 결정됨

#------- 숫자형 ------- 숫자형 사이의 _(언더스코어)는 무시됨

intType = 15 # 정수형 : 메모리가 허용하는 길이까지 사용가능, 
binaryInt = 0x_0010_1011 # 43, 0x는 2진수 접두어 0o는 8진수, 0b는 16진수 접두어

floatType = .14 # 부동소수점형, 부호 사용가능, 소수부와 정수부 생략가능
floatTpye2 = 3.00
floatType3 = 3.14e-2 #3.14/100 == 3.14 * (10**2) 지수 표기법 사용가능 

imaginaryType =  1 + j # 허수형, i 대신 j로 표현 j**2는 -1

boolType = True # 부울형

strType = 'Hello, World!' # 문자열 : '' 또는, "" 사이로 표시

ListType = [1,2,3] # 컬렉션형, 변수 챕터의 컬렉션 자료형 참조

print(type(intType)) #타입형을 확인하는 함수
```
---
### 문자열의 특징과 이스케이프 시퀀스 예제
_'' 또는, "" 사이로 표시, 다른 언어에는 있는 문자형(char) 자료형 없음, 길이가 1인 문자열_

>문자열 표시법과 예제
```Python
example1 = '시간의 역사를 남기고 떠난 호킹'
#또는
example1 = "시간의 역사를 남기고 떠난 호킹"
#또는
example1 =
'''
시간의 역사를
남기고 떠난
호킹
'''
# 다중 라인 문자열 포함법, 줄바꿈 또한 포함됨
example1=
"""
시간의 역사를 
남기고 떠난
호킹
"""
```
>문자열 이스케이프 시퀀스[^11]와 사용 예
```Python
example1 = "'시간의 역사'를 남기고 떠난 호킹" # 큰따옴표 안에 작음 따옴표를 넣어 표현할 수 있음 
# 출력 결과 : '시간의 역사'를 남기고 떠난 호킹
example1 = '"시간의 역사"를 남기고 떠난 호킹' # 작은따옴표 안에 큰 따옴표를 넣어 표현할 수 있음
# 출력 결과 : "시간의 역사"를 남기고 떠난 호킹
example1 = ''시간의 역사'를 남기고 떠난 호킹' # 같은 따옴표 표시 안됨, 오류
example1 = ""시간의 역사"를 남기고 떠난 호킹" # 같은 따옴표 표시 안됨, 오류
example2 = '시간의 역사를\n 남기고 떠난\n 호킹' #\n은 줄바꿈으로 해석됨
example2 = # 이것과 위의 출력 결과가 같음
'''
시간의 역사를
남기고 떠난
호킹
'''
example3 = "\t\'시간의 역사\'를 \\남기고 떠난\\호킹"
# \를 통하여 \,'," 같은 문자 표현 가능, \t는 탭
# 출력 결과 : 	'시간의 역사'를 \남기고 떠난\ 호킹
```
### 문자열 포맷팅[^12]과 콘솔 입력 받기
_문자열 내에 사용된 문자열 표시 유형을 특정값으로 변경하는 기법_
> 문자열 포맷팅 예제
```Python
# 방법 1. %를 사용
print("이름 : %s, 나이 : %d세" % ("홍길동",20) )
# 형식 : "%s %d" % ("해당 문자열",해당 변수), 앞에서부터 순서대로 자리에 맞춰서 출력
# %s : 문자열, %c : 문자 하나, %d : 10진수, %o : 8진수, %x : 16진수, %f: 부동소수점형 6자리, %%: % 문자 출력
print("이름 : %(name)s, 나이 : %(age)s 세" % {"name":"홍길동","age":20})
# 요런 방법도 가능
# 문자열 포맷팅
# 형식 : %(-:좌측정렬)(padding number)(.부동소수점 자릿수)(s,c,x,f 등)
print("이름 : %-10s, 나이 : %10.2f세" % ("홍길동",20))
# 출력 :이름 : 홍길동       , 나이 :      20.00세

# 방법 2. f-string을 이용한 방법, *제일 실행속도 빠름!* 3.0 버전 이상에서만

name = input("이름을 입력하세요 : ") # 문자열 입력받기, Enter로 구분, 매개변수 문자열을 출력한 뒤, 그 뒤에 입력됨
age = int(input("나이를 입력하세요 : ")) # 입력받은 값을 int형으로 형변환

print(f"이름 : {name}, 나이 : {40/2} 세")
#문자열 맨앞에 f를 주고 내용이나 변수명 기입
# 문자열 포맷팅 형식 : {변수이름, 내용:(빈 padding에 채울내용,단, 빈칸이 있으면 안됨)(<왼쪽정렬,>오른쪽정렬,^중앙정렬)(padding)(.부동소수점 자릿수)(형식)}
print(f"이름 : {name:*^10s}, 나이 : {age:<10.4f}세")
# 출럭 : 이름 : ***홍길동****, 나이 : 20.0000   세

# 방법 3. str.format()을 사용
print("이름 : {0}, 나이 : {1}세".format("홍길동",20))
#string 뒤에 각자 자리 순서에 맞는 내용이나 변수 대입, {} 사이 숫자는 생략 가능
# 형식 바꾸는 법은 방법 2와 같음
```

## 변수
`변수`란,_어떠한 값을 저장하는 그릇, 값을 저장할 때 사용하는 식별자_

* 파이썬에서는 동적 타이핑에 의해 변수의 자료형이 바뀌므로 주의하자
* 하나의 변수에는 하나의 자료형만 들어가도록 설계하자.

### 변수의 규칙
1. __한글__, __숫자__와_(__언더스코어__), __영어 대소문자__로 변수명 생성 가능

2. __예약어[^13]__는 불가능

3. __숫자로 시작하는 변수 생성 불가, 대소문자 구별__

4. 모든 변수는 객체를 참조하는 구조이다. 같은 값을 가지고 있다면, is 연산자로 비교하면 참을 리턴
>변수 이름 생성 예제
```python
   # 1VAR,while = 20,5 
   # 숫자 앞에 불가능, 예약어 불가능
   VAR1 = 30
   # 대소문자 구별
   var1 = 10 
   값2 = 10 # 한글 변수 가능
   _result = var1 is 값2 # is 연산자 : 둘은 같은 객체인가? # _(언더스코어) 가능

   print(_result) # 결과값 True Bool형 자료형, True or False를 반환함
   #즉, 같은 10을 참조하고 있다. 객체를 참조하는 구조이다.
```

### 컬렉션 자료형
#### Tuple

_( )안에 서로 다른 자료형의 값을 콤마(,)로 구분해 하나 이상 저장할 수 있는 컬렉션 자료형_
* 0부터 시작하는 인덱스를 이용해 접근, __한 번 저장된 항목은 변경 불가능__

> 튜플 자료형 예제
```Python
student1 = ("홍길동",20)
student2 = "김말자",25 #위와 동일
print(student1) # 결과값 : ('홍길동', 20)
print(student2) # 결과값 : ('김말자', 25)
print(student1[0])#인덱스를 이용한 접근, 0부터 인덱스 시작, 결과값 : 홍길동
print(student2[1])# 결과값 : 25
print(student1[2])# 결과값: 에러, 0,1 두가지 밖에 없음
student1[1] = "서른살" # 결과값 : 에러, 한 번 저장된 항목은 변경 불가능이기 때문에
student1[99] = False# 결과값 : 에러, 위와 같음
student1= ("홍길동","서른살") # 가능, 왜냐하면 기존의 것을 버리고 새로 참조하는 것이므로
```
#### List
_[ ]안에 서로 다른 자료형의 값을 콤마(,)로 구분해 하나 이상 저장할 수 있는 컬렉션 자료형_

* 0부터 시작해서 접근, __변경 가능__
>List 자료형 예시
```Python
student1 = ["홍길동",20]
student1[1] = False # 리스트이므로 변경됨, 
```
##### List Comprehension(리스트 내포)
- 기존의 배열을 이용해 다른 배열을 만드는 기법
- for + append 보다 빠르다.
```python
data_list3 = []

data_list3 = [item for item in data_list1] 
# 리스트 내포기능을 통하여 list1과 동일한 항목으로만듬, 반복자료형의 경우 리터럴 안에 for문을 사용하여 내포가능

data_list5 = [item for item in data_list1 if item % 2 == 1] # 홀수항목만 저장하는 법

[x * y for x in data_list1 if x % 2 ==1 for y in data_list1 if y % 2 ==0]
# 리스트 내포 안에 for문의 중첩 가능, if 문 사용 가능
```


#### Set
- _{ } 안에 서로 다른 자료형의 값을 콤마로 구분해 저장, __순서 없음__, __데이터 항목의 중복을 허용하지 않음__, 중복 입력시 하나만 입력됨, 그래서 인덱스로 접근 불가_
- 수학에서 활용하는 union, intersection, difference등의 함수를 사용 가능
- List 보다 성능상 빠르며, 실수로 값을 변경하는 것을 방지

>Set 자료형 예시
```Python
student1 = {False,"홍길동",20,"이순신",False}
print(student1) # 결과값 : {False, '이순신', '홍길동', 20} False 하나는 중복이므로 사라짐,
# 순서 무작위로 다시 나타남
student1[1] = False # 셋은 순서가 없으므로 인덱스를 이용해 접근 못하므로 에러 
```
_Set은 집합의 개념을 가지고 있는 자료구조이므로 합집합 연산이 가능함_ 
_연산 대상 중 하나라도 List, Tuple이면 불가능, 즉 List, Tuple은 합집합 연산 불가능_
>Set의 합집합 연산 예시
```Python
student = {False,"홍길동",20,"이순신",False}
student |= {3.14,True} # 오른쪽 값을 왼쪽 값과 합연산하라는 의미
print(student) # 결과값 :  {False, True, 3.14, 20, '이순신', '홍길동'}
# 두 값 중 중복된 값을 제외하고 나옴
```
#### Dictionary
_{ } 안에 __키: 값__ 형식의 항목을 콤마로 구분해 하나 이상 저장할 수 있는 컬렉션 자료형_

__항목 추가시  동일키가 없으면 새로운 항목추가, 있으면 저장된 항목을 변경함__
__순서 개념 없으므로 인덱스로 접근 불가하며, 키를 이용해 값을 읽어 올 수 있다__
>Dictionary 자료형 예시
```Python
dogs = {1:"골든리트리버",2:"진돗개",3:"보더콜리"}
print(dogs) # 결과값 : {1: '골든리트리버', 2: '진돗개', 3: '보더콜리'}
print(dogs[0]) # 에러, 인덱스로 접근 불가능
print(dogs[1]) # 결과값 : 골든리트리버  인덱스가 아닌 키값 1로 접근
dogs[1] = "치와와"
print(dogs[1]) # 결과값 : 치와와 키 1의 값인 골든리트리버가 치와와로 바뀜
# dogs["4"] != dogs[4], dictionary 키 값이 뭔지 알고 접근하자
```
#### None 객체와 변수 생성 초기화
_파이썬에서는 None 객체를 이용해 널(null) 객체 상태를 표현, 비어있음을 표현_
_ 최초 변수 선언시 초기화 하지 않으면 에러가 나므로, None 객체 저장 필요_
>None 자료형과 변수 생성 초기화 예시
```python
obj # 에러
obj =  None # 에러 안남
obj is None # 동일한 객체인가? True 주로 이걸 쓰자
obj == None # 동일한 값인가? True
x=10
y=20
x,y = y,x # 값이 바귐
x ,y = [10,20]
[x,y] = 10,20
[x,y] = [10,20]
# 어떤 결과든 x = 10, y = 20
```
> Garbage collector가 자동으로 변수를 제거해서 메모리 관맇지만 del()함수로 직접 객체를 지울 수도 있음 


#### collections
- List, Tuple, Dict에 대한 Python Built-in 확장 자료 구조(모듈)
- deque, Counter, OrderedDict, defaultdict, namedtuple 등의 모듈을 포함
##### deque
- stack과 queue를 지원하는 모듈
- List에 비해 빠르다.
- 기존 list의 함수들, stack, queue를 위한 appendleft 함수 뿐만 아니라, rotate, reverse 등 Linked List의 특성 지원
```python
from collections import deque

deque_list = deque()
for i in range(5):
	deque_list.append(i)
print(deque_list)
deque_list.appendleft(10)
print(deque_list)
deque_list.rotate(2)
print(deque_list)
deque_list.rotate(2)
print(deque_list)
print(deque(reversed(deque_list)))

deque_list.extend([5,6,7])
print(deque_list)
deque_list.extendleft([5,6,7])
print(deque_list)
```
##### OrderedDict
- 입력 순서에 따른 출력 순서를 보장하고(최신 Python은 기존 dict도 가능함), 정렬이 가능한 딕셔너리
##### defaultdict
- Dict type의 값에 기본 값을 지정, 신규값 생성시 사용하는 방법
```python
from collections import defaultdict, OrderedDict
text= "a to a and the press release that of your a to and the press that of and the press".lower().split()
word_count = defaultdict(lambda: 0) # Default 값을 0으로 설정
for word in text:
    word_count[word] += 1  # 굳이 word_count[word] = 0 으로 생성안해도 됨

for i, v in OrderedDict(sorted(word_count.items(), key=lambda t: t[1], reverse=True)).items():
    print(i,v)
###
a 3
and 3
the 3
press 3
to 2
that 2
of 2
release 1
your 1
###
    
```
##### Counter
- Sequence type의 data element들의 갯수를 dict 형태로 반환
```python
from collections import Counter

c = Counter()
c = Counter('gallahad')

print(c) # Counter({'a': 3, 'l': 2, 'g': 1, 'h': 1, 'd': 1})

c = Counter({'red': 4, 'blue': 2})
print(c) # Counter({'red': 4, 'blue': 2})
print(list(c.elements())) # ['red', 'red', 'red', 'red', 'blue', 'blue']

c = Counter(cats=4, dogs=8)
print(c) # Counter({'dogs': 8, 'cats': 4})
print(list(c.elements())) # ['cats', 'cats', 'cats', 'cats', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs', 'dogs']

c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
print(c+d) # Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2})
print(c&d) # Counter({'b': 2, 'a': 1})
print(c|d) # Counter({'a': 4, 'd': 4, 'c': 3, 'b': 2})
c.subtract(d) # c- d
print(c) # Counter({'a': 3, 'b': 0, 'c': -3, 'd': -6})

# word counter 구현
text= "a to a and the press release that of your a to and the press that of and the press".lower().split()
print(Counter(text)) # Counter({'a': 3, 'and': 3, 'the': 3, 'press': 3, 'to': 2, 'that': 2, 'of': 2, 'release': 1, 'your': 1})
print(Counter(text)["a"]) # 3
```
- Dict type, keyword parameter 등도 모두 처리 가능, Set 연산 지원
##### namedtuple
- Tuple 형태로 Data 구조체를 저장하는 방법
- 저장되는 data의 variable을 사전에 지정해서 저장
```python
from collections import namedtuple

point = namedtuple('Point', ['x', 'y'])
p = point(11, y=22)
print(p.x,p.y) # 11, 22

```
## 연산자
### 산술 연산자
```python
a = 3 + 2 # + 연산자, 양변의 값을 더하기, a 값은 5, 우선순위가 -와 함께 다른 연산자에 비해 낮음
a = 3 - 2 # - 연산자 양변의 값을 빼기, a 값은 1
a = 3 * 2 # * 연산자 양변의 값을 곱하기, a 값은 6
a = 3 / 2 # / 연산자 양변의 값을 나누기, a 값은 1.5
a = 3 // 2 # // 연산자 양변의 값을 나눈 몫, a 값은 1
a = 3 % 2 # % 연산자 양변의 값을 나눈 나머지, a 값은 1
a = 3 ** 2 # ** 연산자 양변의 값을 제곱, a 값은 9

# 문자열 접합 연산 # 문자열을 더하면 접합 연산됨
a = '파이썬'
b = '1'
print(a + b) # 결과 '파이썬1'

c = '2'

print (b+c) # 결과 '12'
print(int(b)+ int(c)) # 결과 3, 문자열을 정수로 더하려면 형변환하고 난뒤에
```
### 대입연산자
```python
a = 3 # 우변의 값을 좌변에 대입, a 값은 3
a += 2 # 우변의 값을 좌변의 변수에 더한 뒤, 좌변의 변수에 대입, a 값은 5
a -= 2 # 우변의 값을 좌변의 변수에 뺀 뒤, 좌변의 변수에 대입, a 값은 3
a *= 2 # 우변의 값을 좌변의 변수와 곱한 뒤, 좌변의 변수에 대입, a 값은 6
a /= 2 # 우변의 값을 좌변의 변수와 나눈 뒤, 좌변의 변수에 대입, a 값은 3
a //= 2 # 우변의 값을 좌변의 변수와 나눈 몫을 좌변의 변수에 대입, a 값은 1
a = 3 # a에 다시 3 대입
a %= 2 # 우변의 값을 좌변의 변수와 나눈 나머지를 좌변의 변수에 대입, a 값은 1
a = 3 # a에 다시 3 대입
a **= 2 # 우변의 값으로 좌변의 변수를 제곱해서 좌변의 변수에 대입, a 값은 9

# 문자열 대입연산 예시
a = '파이썬'
a += '은 재밌어'
print(a)#결과 값: '파이썬은 재밌어'
```
### 관계  연산자(비교 연산자)
```python
a, b = 3, 2
print(a == b) # 양변의 값이 같으면 True 반환 결과값 : False
print(a != b) # 양변의 값이 다르면 True 반환 결과값 : True
print(a > b) # 좌변의 값이 우변의 값보다 크면 True 반환 결과값 : True
print(a < b) # 좌변의 값이 우변의 값보다 작으면 True 반환 결과값 : False
print(a >= b) # 좌변의 값이 우변의 값보다 크거나 같으면 True 반환 결과값 : True
print(a <= b) # 좌변의 값이 우변의 값보다 자거나 같으면 True 반환 결과값 : False
```
### 논리 연산자

```python
a,b = True, false
print(a and b) # 양변의 값이 모두 True일 경우에만 True를 반환 결과값 : False
print(a or b) # 양변의 값 중 하나라도 True일 경우 True 반환 결과값 : True
print(not a) # 해당 bool 값의 반대를 출력 결과값 : False
```

### 비트 연산자

```python
x, y = 1,0
print(x&y) # 비트 논리곱 연산자, 양변 비트 값 모두 일 경우에만 1을 반환 결과값 : 0
print(x|y) # 비트 논리합 연산자, 양변 비트 값 모두 0일 경우에만 0을 반환 결과값 : 1
print(x^y) # 비트 베타적 논리합 연산자, 양변의 값이 다를 경우 1, 같을 경우 0을 반환 결과값 : 1
print(~x) # 비트 부정 연산자, 비트값들을 반전 시킴, 결과값 : -2 (2의 보수법에 의한 결과, 이유는 밑의 컴퓨터의 음수와 양수 표현 참조)
x = 4
print(x<<1) # 비트 왼쪽 쉬프트 연산자, 좌변의 값을 우변의 값만큼 비트를 왼쪽으로 이동
print(x>>1) # 비트 오른쪽 쉬프트 연산자, 좌변의 값을 우변의 값만큼 비트를 왼쪽으로 이동
```

#### 컴퓨터의 음수와 양수 표현
 컴퓨터는 음수와 양수를 구별하기 위해서 __부호 절대값__, __1의 보수__, __2의 보수__ 등의 방법을 사용한다.
 파이썬은 __2의 보수법__을 사용

 ##### 부호 절대값 방법
 	최상위 비트를 부호 비트로 사용하는 방법, 최상위 비트가 0이면 양수, 1이면 음수이다.
 	하지만 치명적인 문제점이 있다

 > 부호 절대값 방법의 문제점
 ```부호 절대값 방법
 4비트 기준
 1001(십진수 시 -1) + 0001(십진수 시 1) = 1010(-2)
 하지만 실제 십진수 값은 0이 되어야 맞다.
 또한 0을 표현 하는 방식도, +0(1000), -0(0000) 두 가지가 존재하게 된다.
 ```

#### 1의 보수법
 최상위비트로 음수와 양수를 구별하는 법은 같으나, 주어진 이진수의 보수를 음수로 표현하는 방법이다. 음수와 양수를 단순히 더하는 방식으로 뺄셈을 구현할 수 있다.
> 1의 보수법 예시
```1의 보수법
4비트 기준
2 + 3 = 0010 + 0011 = 0101 = 5
-2 + -3 = 1101 + 1100 = 1001 + 0001(자리올림에 의한 최하위비트 덧셈)= 1010 = 5(0101)의 보수 = -5

8비트 기준
3 + 3 = 0000 0011 + 0000 0011 = 0000 0110
-3 + -3 = 1111 1100 + 1111 1100 = 1111 1000 + 0000 0001 = 1111 1001  

단, 위의 예와 같이 자리올림이 발생하면 최하위 비트에 1을 더해주어야 한다.
하지만 이때도 0은 +0과 -0이 존재하며, 자리올림이 발생하였는지 확인해야하는 소모가 발생한다.
```

#### 2의 보수법
 2의 보수는 __1의 보수에 1을 더해서__ 구할 수 있다.
 (또는)
 주어진 이진수 보다 한 자리 높고 가장 높은 자리가 1인 2의 제곱수에서 주어진 수를 빼서 얻는다

> 2의 보수를 구하는 예시와 2의 보수 덧셈
```2의 보수법
4비트 기준

3(0011)의 2의보수
= 1100(3의 1의보수) + 0001(1을 더해준다) = 1101 = -3
또는
10000 - 0011(3) = 1101

0의 경우
+0 = 0000
- 0 = 1111(1의 보수의 -0)
이를 2의 보수로 표현하면, 1111+ 0001(1을 더해준다) = 0000 = +0과 같다.

2 + 3 = 0010 + 0011 = 0101 = 5
-2 + -3 = 1110(2의 보수)+1101(2의보수) = 1011 = 5의 2의보수 = -5

2의 보수법을 이용하여 -0 문제와 자리올림 발생을 처리하였다, 그리고 하드웨어적으로 뺄셈을 효율적으로 구현할 수 있다.
```
### 연산자 우선순위
> 연산자 별로 우선 적용되는 순위가 따로 있으므로 가독성과 정확한 결과값을 위해 괄호()를 이용하자
```
1. ()  : 괄호
2. ** : 제곱
3. +,-,~ : 부호, 비트 부정
4. *,/,//,% : 곱하기, 나누기, 몫, 나머지
5. +, - : 더하기, 빼기
6. <<. >> : 비트 왼쪽 시프트, 비트 오른쪽 시프트
7. & : 비트 논리곱
8. ^ : 비트 배타적 논리합
9. | : 비트 논리합
10. <.<=,>,>=,==,!= : 크기 비교자
11. not : 부정
12. and : 논리곱
13. or : 논리합 
```
* 괄호가 최우선
* 산술 연산이 비트 연산보다 우선
* 관계 연산이 논리 연산보다 우선

## 흐름과 제어문
### if문, else문, if\~elif\~else 문
_어떤 조건을 만족하는 경우 명령문을 수행하기 위해 사용_

>if~elif~else문의 형태와 예제
```Python
if 조건식 : # 조건식이 참일때 아래 명령문을 실행
	명령문() # 들여쓰기로 코드 블록을 구분함 중요
	명령문2() # 1 들여쓰기 = 4 공백(Spacebar)
elif 조건식 2 : # 위 조건식이 거짓, 여기는 참일때
	명령문3() 	# 위 명령문은 무시되고 이 명령문이 실행
	명령문4()	# elif문을 중첩해서 여러 조건식도 가능
else :	# 만약 위의 모든 조건식 이외의 조건에서 발동
	명령문5()
조건식에영향받지않는명령문() # 코드 블록 바깥이므로 위 조건문들을 통과한 뒤 무조건 실행된다.
score = 10

if score==100 :
	print("wow! genius!")
elif score >= 66 : 
	print("pretty goodjob!")
elif 66 >= score >= 33 : #Python만 가능한 코드
	print("not good")
else :
	print("try again!")
print("test result!")# 무조건 실행
```
### 반복문과 반복 처리문 : for 문, while 문, break 문, continue 문

 반복문 : 특정 작업을 반복해서 수행하기 위해 사용

#### for 문

> for 문의 형식과 예시
```python
for 변수 in 순회할객체 :
	명령문
	명령문
	
# range()함수 : 첫번재 인자는 범위 시작 값 초기값은 0으로 생략가능, 두번째 인자는 종료값으로 생략불가능, *범위에 해당 종료값은 포함 안됨*, 세번째 인자는 증감치로 초기값은 1로 생략가능
# range(0,10,2) == [0,2,4,6,8]
# range(1,5) = [1,2,3,4]
# range(4) = [1,2,3]

for i in range(10) : # 10번 순회함
	명령문
	명령문

dogs = {1:"치와와",2:"진돗개",3:"고양이"}
# 각 인자를 변수로 사용할 수 있음
for key in dogs: #매 반복마다 key값이 바뀜
	print(f"{dogs[key]}: 멍멍!")
# 결과값:
"치와와: 멍멍!"
"진돗개: 멍멍!"
"고양이: 멍멍!"

# 중첩된 for 문의 문법
for 변수1 in 순회할객체1:
	for 변수2 in 순회할객체2: #들여쓰기에 신경써서 블록을 구분해야함
		명령문1
		명령문2

dan = range(2,10)
num = range(1,10)

for i in dan:
	for k in num:
		print("{0} X {1} = {2:>2}".format(i,k,i*k))
	print() # for i in dan 밑에 속해잇음
			
#결과값 : 구구단 출력
```
#### while 문

bool 값을 반환하는 조건식의 결과에 의해 반복 결정
_ 무한 반복에 주의 _
> while 문 형식과 예시
```python
while 조건식 :
	명령문 1
	명령문 2
	
i = 1
while i < 10: 
	print(f"{i}번째 문장")
	i += 1
# 결과값 :
1번째 문장
2번째 문장
.
.
.
10번째 문장
```

#### break 문과 continue 문

##### break 문
_논리적으로 반복문을 빠져나갈 때 사용_
> break문 예시
```python
i = 1
while True: # break문을 만나기 전까지 무한 루프임
	if i >= 5
    	break # while 문에서 벗어남
    print(f"{i},")
	i += 1
# 결과값 : 1,2,3,4, 
```
#### continue 문
_이후 코드는 건너뛰고 반복문을 계속 실행을 반복할 때 사용_

>continue 문 예시
```python
for n in range(10)
	i += 1
	if (n = 3) or (n = 5) :
    	continue # while 문에서 벗어남
    print(f"{i},")
# 결과값 : 1,2,4,6,7,8,9, 
# 3과 5는 출력부분을 건너뛰게 됨
```
## 함수
프로그램에서 어떤 특정 기능을 수행할 목적으로 만들어진 재사용 구조의 코드 부분 

파이선의 경우 Call by Object Reference[^14] 라는 방식으로 구현되었다.

_효율적이고 구조화된 프로그램 생성 가능_

### 함수의 기초

* 함수의 장점

1. 하나의 큰 프로그램을 여러 부분으로 나눌 수 있기 때문에 구조적 프로그래밍이 가능해짐
2. 동일 함수를 여러 곳에서 필요할 때마다 호출할 수 있음
3. 수정이 용이함

#### 함수의 개념과 목적

__매개 변수__
* 함수 호출 시 입력 값을 전달 받기 위한 변수
* 전달받은 인자의 값에 의해 타입이 결정됨
* 선언된 매개변수의 개수만큼 인자 전달 가능
* 만약 함수의 매개변수와 숫자가 일치하지 않으면 오류 일어남
	- __Scope__ : 변수의 유효범위
		_전역 스코프_ : 어디서나 접근 가능한 전역 변수
		_함수 스코프_ : 함수 내에서만 접근 가능한 지역 변수
			-매개변수, 함수 내부에서 정의된 변수 등
>변수의 유효범위 예시
```python
#만약 전역 변수와 지역변수 이름이 같으면 함수 내의 경우,
#먼저 함수 스코프 내에서 변수를 찾고, 
#그 뒤 전역 스코프에서 변수를 찾음
#그러므로 변수 이름을 같이 하고 싶을 경우 변수 앞에 __global__을 붙여 전역 스코프에서의 변수임을 선언하면 됨
def change_global_var():
	global x
	x += 1
x = 5
change_global_var()
# x값은 6으로 바뀜

```
 인자 : 매개변수로 전달되는 값
 반환값 : 함수가 기능을 수행한 후 반환되는 값
 순수 함수: 결과값 반환 외에 외부에 영향을 주지 않는 함수

### 함수의 호출 및 선언
#### 함수의 호출
>함수 호출의 형식과 예
```python
#함수 호출은 함수명(매개인자1,매개인자2,...) 의 형태를 띔
a,b=2,3
print(a+b)#받은 인자를 출력해서 보여줌, 결과값 : 5
```
#### 함수의 선언
>함수 선언의 형식과 예
```python
def 함수명(매개변수1,매개변수2,...):
	명령문1 #코드 블록 들여쓰기로 구분
	명령문2
	return문 # 반환값이 없으면 없을 수 도 있음

def calc_sum(x,y): # 인터프리터 언어이므로 함수 선언이 실제 사용보다 먼저 해야함
	return x + y

a, b = 2,3

c = calc_sum(a,b) #반환값 c = 5

#기본값을 갖는 매개변수 예시
def calc(x,y, operator="+"): # operator 인자의 값을 쓰지 않으면 +를 기본으로 입력됨
	if operator == "+":
		return x+y
	else:
		return x-y
ret_val = calc(10,5,"+") # 결과값 15
ret_val = calc(10,5,"-") # 결과값 10
ret_val = calc(10,5) # 결과값 15 생략해서 기본값인 "+"가 입력됨

```
### 함수의 유형

__매개변수의 유무__와 __반환값의 유무__에 따라 유형이 나눠짐

> 매개변수가 있고 반환값도 있는 함수 예시
```python
def func_parameters_return(x,y):
	print(f"매개변수로 {x}와 {y}가 전달된 반환값이 있는 함수")
	return x+y

print(func_parameters_return("판젤라틴","커튼"))
'''
결과값 :
매개변수로 판젤라틴과 커튼가 전달된 반환값이 있는 함수
판젤라틴커튼
'''
```
> 매개변수가 있고 반환값이 없는 함수 예시
```python
def func_parameters_noreturn(x,y):
	print(f"매개변수로 {x}와 {y}가 전달된 반환값이 없는 함수")

func_parameters_noreturn("판젤라틴","커튼")

# 결과값 : 매개변수로 판젤라틴과 커튼가 전달된 반환값이 없는 함수	
```
> 매개변수가 없고 반환값이 있는 함수 예시
```python
def func_noparameters_return():
	print("매개변수가 없는 함수입니다.")
	return "반환값은 있다니깐!"	

print(func_noparameters_return())
'''
결과값 : 
매개변수가 없는 함수입니다.
반환값은 있다니깐!
'''
```
> 매개변수가 없고 반환값도 없는 함수 예시
```python
def func_noparameters_noreturn():
	print("매개변수가 없고 반환값도 없는 함수 예시")

func_noparameters_noreturn()
#결과 값: 매개변수가 없고 반환값도 없는 함수 예시
```
#### 언팩 연산자와 키워드 언팩 연산자

1. \*언팩 연산자
- 매개변수의 개수를 가변적으로 사용할 수 있도록 언팩 연산자(*) 제공
- 매개변수에 적용시 인자를 튜플 형식으로 처리

> 튜플 매개변수 입력과 튜플 반환 예시
```python
def calc_sum(strings,*params): # 다른인자와 도 쓸수 있음, 가변 매개변수 인자 앞에 *넣기, 마지막 인자에 1개만 적용 가능, C#의 params 같은 느낌?
	total = 0
    word = ""
    word = strings
	for val in params: # 인자 갯수만큼 params라는 튜플로 나옴
		total += val
	return word, total1	# 결과값도 콤마로 구분하여 여러개 낼 수 있음, 튜플 형식으로 반환
	
val = def calc_sum("더하기",1,2) # *붙여서 튜플형식 반환값 나타내기
print("{0}, {1}".format(*val))# 1+2 = 결과값 :더하기, 3
val = def calc_sum("덧셈",1,2,3,4,5)
print(val[0] +", "+ val[1])  # 1+2+3+4+5 = 결과값 : 덧셈, 15
```

2. \*\*키워드 언팩 연산자
- 매개변수의 개수를 가변적으로 사용할 수 있도록 함
- 키워드 인자들을 전달해 매개변수를 딕셔너리 형식으로 처리함
> 키워드 언팩 연산자 예시
```python
def use_keyword_arg_unpacking(**params): # 키=값 형식의 인자값들이 params 매개변수에 딕셔너리형식으로 전달
	for k in params.keys():
		print("{0}: {1}".format(k, params[k])) # 키는 전달된 매개변수 이름, 값은 전달된 인자 값
use_keyword_arg_unpacking(a=1,b=2,c=3) #딕셔너리 형식으로 전달
# 결과값
'''
a: 1
b: 2
c: 3
'''
```

### 고급 함수 사용법

1. __중첩 함수__ : 함수 내에 중첩함수를 선언해 사용 가능
	- 중첩 함수를 포함하는 함수 내에서만 호출이 가능,
	- 중첩 함수를 포함하는 함수의 스코프에도 접근이 가능
	
> 매개변수에 함수 전달하기 예제
```python
# 매개변수에 함수를 전달하여 프로그램의 유연성을 높일 수 있음
def calc(operator_fn,x,y):
	return operator_fn(x,y)

def plus(op1,po2):
	return op1 + op2

def minus(op1, op2):
	return op1 - op2

ret_val = calc(plus,10,5) # plus 함수 집어넣음
print(ret_val) # plus 함수를 매개변수로 넘겨주었기 때문에 결과값 : 15
ret_val =calc(minus,10, 5) # minus 함수 집어넣음
print(ret_val) # minus 함수를 매개변수로 넘겨주었기 때문에 결과값 : 5
```
2. __람다식__ : Lambda 매개변수 : 반환값
	- 변수에 저장해 재사용이 가능한 함수처럼 사용함
	- 기존의 함수처럼 매개변수의 인자로 전달함
	- 함수의 매개변수에 직접 인자로 전달할 수 있음
	- 최근에는 reduce 함수와 함께 가독성 문제로 사용을 권장하지 않음
> 람다식 예제
```python
def calc(operator_fn,x,y):
	return operator_fn(x,y)

ret_val = calc(lambda a, b: a + b,10,5) # 람다식, 두 매개변수를 더함
print(ret_val) # 더하는 람다식을 매개변수로 넘겨주었기 때문에 결과값 : 15
ret_val =calc(lambda a, b: a - b,10, 5) # 람다식, 두 매개변수를 뺌
print(ret_val) # 빼는 람다식을 매개변수로 넘겨주었기 때문에 결과값 : 5
```
3. __클로저__ : 
	1) 중첩함수에서 중첩함수를 포함하는 함수의 scope에 접근 가능
	2) 중첩함수 자체를 반환값으로 사용하면
		가. 정보 은닉 구현 가능
		나. 전역변수의 남용 방지
		다. 메서드가 하나밖에 없는 객체를 만드는 것보다 우아한 구현 가능
> 중첩함수 반환 클로저 예제
```python
def outer_func():
	id = 0 # 지역변수 해당 함수 또는 중첩 함수에서만 접근 가능
	
	def inner_func():
		nonlocal id #nonlocal을 통하여 inner_func의 지역변수가 아님을 알림, 즉 outer_func()의 변수를 찾게 만듬
		id += 1
		return id
	
	return inner_func # 함수 호출이 아닌 함수에 대한 참조를 반환함
	
make_id = outer_func() # inner_func()의 반환값이 아닌, make_id는 inner_func 함수가 됨
print(make_id()) # 결과값 : 1
```
#### 함수 타입 힌트와 함수 문서화

```python
def do_function(var_name: var_type) -> return_type:
``\`
    # function description
    {{#args}}                       - iterate over function arguments
        {{var}}                     - variable name
        {{typePlaceholder}}         - [type] or guessed type  placeholder
        {{descriptionPlaceholder}}  - [description] placeholder
    {{/args}}
    
    {{#kwargs}}                     - iterate over function kwargs
        {{var}}                     - variable name
        {{typePlaceholder}}         - [type] or guessed type placeholder
        {{&default}}                - default value (& unescapes the variable)
        {{descriptionPlaceholder}}  - [description] placeholder
    {{/kwargs}}
    
    {{#exceptions}}                 - iterate over exceptions
        {{type}}                    - exception type
        {{descriptionPlaceholder}}  - [description] placeholder
    {{/exceptions}}
    
    {{#yields}}                     - iterate over yields
        {{typePlaceholder}}         - [type] placeholder
        {{descriptionPlaceholder}}  - [description] placeholder
    {{/yields}}
    
    {{#returns}}                    - iterate over returns
        {{typePlaceholder}}         - [type] placeholder
        {{descriptionPlaceholder}}  - [description] placeholder
    {{/returns}}
    python Docstring Generator 기준
    ``\`
    pass
```


- 함수 타입 힌트

  - 파이썬의 특징인 dynamic typing을 포기하고 형을 강제할 수 있다.
  - python 3.5 버전 이후로 사용 가능
  - 원치 않는 결과와 버그를 많이 방지할 수 있다.
- 함수 문서화

  - IDE에서 함수 작성시 팝업으로 뜨는 함수 설명을 커스터마이징하여 적을 수 있다.
  - VScode 등의 extension을 이용하면 좀더 쉽게 작성할 수 있다. 

## Pythonic code

- 파이썬 스타일의 효율적인 코딩 기법, 고급 코드 작성할 수록 많이 필요해짐

- lamda, map, reduce, list comprehension 등이 있다.

### enumerate
- list의 element를 추출할 때 번호를 붙여서 추출
```python
for idx, value in enumerate(['tic', 'tac', 'toe']):
	print(idx, value)
###
0 tic
1 tac
2 toe
###	
```

### zip
- 두 개의 list의 값을 병렬적으로 추출
```python
alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']
for a, b in zip(alist, blist):
	print(a,b)
###
a1 b1
a2 b2
a3 b3
###	
```
### iterable object
- Sequence형 자료형에서 데이터를 순서대로 추출하는 object
- list, set, 문자열 등에 있음
- \_\_iter\_\_ 함수와 \_\_next\_\_ 함수로 iterable 객체를 iterator object로 사용
- iter 함수 : 배열을 iterator object로 변경
```python
citieds = ["Seoul", "Busan", "Jeju"]
iter_obj = iter(cities)

print(next(iter_obj))# Seoul
print(next(iter_obj))# Busan
print(next(iter_obj))# Jeju
next(iter_obj) # stopIteration 오류 발생
```
### generator
- iterable ojbect를 특수한 형태로 사용해주는 함수
- element가 사용되는 시점에 값을 메모리에 반환, :yield를 사용해 한번에 하나의 element만 반환
```python
def generator_list(value):
	result = []
	for i in range(value):
		yield i
```
#### generator comprehension
- [] 대신 ()를 사용하는 generator 형태의 list
- 일반적인 list보다 적은 메모리를 사용함
- list 타입 데이터 반환시, 큰 데이터, 파일 데이터를 사용할 때 많이 사용함
```python
gen_ex = (n*n for n in range(500))
print(type(g))
```
## Object-Oriented Programming(OOP, 객체지향 프로그래밍)
- 객체 개념을 프로그램으로 표현하여 문제를 해결하는 방법
- 객체 : 실생활에서 일조의 물건, 속성(Attribute)와 행동(Auction)을 가짐
	- 클래스 : 객체를 만들기 위해 미리 속성과 행동이 적혀있는 설계도, 청사진, 템플릿
	- 인스턴스 : 클래스를 사용하여 만든 구현 객체
```python
# class 이름은 Camelcase로, 함수, 변수는 snake_case로
class SoccerPlayer(object):#object는 상속받는 객체명으로 생략가능
	# __init__은 객체 초기화 예약 함수
	def __init__(self, name, position, back_number):
		# 객체 자신을 뜻하는 self를 꼭 넣어야 class 함수로 인정된다.
		self.name = name		
		self.position = position
		self.back_number = back_number
		
	def change_back_number(self, new_number): # method
		print("등번호 변경")
		self.back_number = new_number
jinhyun = SoccerPlayer("Jinhyun", "MF", 10) # 객체 생성 및 초기화
# 객체명 = class명(__init__함수 interface(초기값))
jinhuyn.change_back_number(5) #메소드 사용
````
- 이외에도 \_\_main\_\_, \_\_add\_\_, \_\_str\_\_,\_\_eq\_\_  등의 예약함수가 존재

### Inheritance(상속)
- 부모 클래스로부터 속성과 Method를 물려받은 자식 클래스를 생성하는 것
```python
class Person(object):
	def __init__(self, name, age):
		self.name = name
		self.age = age
	
	def introduce(self):
		print(self.name, self.age)

class Korean(Person):
	pass
	
first_korean = Korean("Sungchul", 35)
print(first_korean.name) # korean 클래스에 없는 속성과 Method를 Person(부모 객체)것을 이용해 사용가능
first_korea.introduce()
```
### Polymorphism(다형성)
- 같은 이름 메소드의 내부 로직을 다르게 작성
- Method Overriding을 의미함
```python
class Animal:
	def talk(self):
		raise NotImplementError("Subclass must implement abstract method")
		
class Cat(Animal):
	def talk(self): # 부모의 메소드를 사용하지 않고 같은 이름으로 새로 바꾸어 사용
		return 'Meow!'
		
class Dog(Animal):
	def talk(self):
		return 'Woof! Woof!'
		
nabi = Cat()
yebbi = Dog()

print(nabi.talk()) # Meow!
print(yebbi.talk()) # Woof! Woof!
		
```

### Visibility(가시성)
- 객체의 정보를 볼 수 있는 레벨을 조절함.
- 객체를 사용하는 사용자가 임의로 정보를 수정하거나 필요 없는 정보에 접근, 소스의 보호를 위해 사용
- **Encapsulation(캡슐화, 정보은닉)** 라고도 함
```python
class Inventory(object):
    def __init__(self):
        self.__items = [] # private 변수로 타객체가 접근 불가

    def add_new_item(self, product):
        self.__items.append(product)

    def get_number_of_items(self):
        return len(self.__items)

    @property
     # property decorator, private 변수를 반환하게 해줌
    def items(self):
        return self.__items

my_inventory = Inventory()
my_inventory.add_new_item('potion')
print(my_inventory.get_number_of_items()) # 1
items = my_inventory.items # 함수로 변수처럼 호출하여 반환 가능
print(items) # ['potion']
print(my_inventory.__items) # AttributeError: 'Inventory' object has no attribute '__items'
```
### decorate
#### first-class objects
- 파이썬의 함수들의 특징, 일등함수 또는 일급 객체
- 변수나 데이터 구조에 할당이 가능한 객체
- 함수의 파라메터로 전달이 가능 + 리턴값으로 사용
```python
def square(x):
	return x * x
	
	
f = square
print(f(5)) # 25

```
#### inner function
- 함수 내에 또다른 함수가 존재 가능
```python
def print_msg(msg):
	def printer(): # inner function
		print(msg)
	return printer # inner function을 return 값으로 반환


another = print_msg("Hello, python")
another()
```
#### decorator
- 클로져 함수를 좀더 간단하게 쓰게 해줌.
- @함수명을 함수 위에 놓음으로 inner함수로 만들 수 있음
```python
def star(func):
	def inner(*args, **kwargs):
		print('*' * 30)
		func(*args, **kwargs)
		print('*' * 30)
	return inner
	
@star
def printer(msg):
	print(msg)
printer("Hello")
# ******************************
# Hello
# ******************************
```
## 모듈과 패키지
### Module
- 미리 정의된 함수와 클래스의 집합
- 새로 정의한 커스텀 모듈과 파이썬이 기본제공하는 Built-in Module이 있다.
- 모듈이름으로 할 같은 폴더 내에 **\_\_init\_\_.py**를 만들고, 기타 .py파일을 만들어 생성가능
- **import 모듈명 from 모듈경로** 로 사용 가능
	- 절대 참조나 현재 디렉토리 기준, 부모 디렉토리 기준으로 모듈 경로 지정 가능
### Package
- 모듈들을 모아놓은 단위, 하나의 프로그램
- 모듈들의 합, 즉 폴더들의 합으로 연결됨
> \_\_init\_\_.py 예시
```python
__all__ = ['image', 'stage', 'sound']

from . import image
from . import stage
from . impoart sound
```

> \_\_main\_\_.py 예시
```python
from stage.main import game_start
from stage.sub import set_stage_level
from image.character import show_character
from sound.bgm import bgm_play

if __name__ == '__main__':
	game_start()
	set_stage_level(5)
	bgm_play(10)
	show_character()
```
- 가상환경을 이용해 프로젝트 진행 시 필요한 패키지만 설치 가능
	- virtualenv + pip, conda 등이 있음 

## Exception/File/Log Handling

### Exception

- 프로그램 사용할 때 일어나는 오류들
- 발생 여부를 사전에 인지할 수 있는 예외는 명시적으로 정의해줘서 해결
- 예상 불가능한 예외는 인터프리터 과정에서 발생 가능한 예외, index값의 리스트 범위 넘어가기, 0으로 나누기 등...

#### Exception 문법

```python
try:
    	예외 발생 가능 코드
except <Exception Type> as e: # as e 생략 가능
    	print(e)
    	예외 발생 시 대응 코드
else: # 생략 가능
    예외가 발생하지 않을 때 동작 코드
finally:
    예외 발생 여부와 상관없이 실행
```

- Built-in Exception

| Exception 이름                          | 내용                          |
| --------------------------------------- | ----------------------------- |
| indexError                              | List의 Index 범위를 넘어감    |
| NameError                               | 존재하지 않는 변수 호출       |
| ZeroDivisionError                       | 0으로 숫자를 나눌 때          |
| ValueError                              | 변환할 수 없은 문자/숫자 변환 |
| FileNotFoundError                       | 존재하지 않는 파일 호출       |
| 기타 등등, + 커스텀 Excpetion 제작 가능 |                               |

#### raise, assert 구문

```python
if a == 0:
    raise ZeroDivisionError: # raise <exception type>(예외정보)
        print("Divided by 0")
```

- raise : 필요에 따라 강제로 Exception 발생

```python
def get_binary_number(decimal_number):
    assert isinstance(decimal_number, int) # assert 예외조건
    # true 일시 코드 정지
    return bin(decimal_number)
```

- assert : 특정 조건에 만족하지 않으면 발생

### File Handling
- File system: OS에서 파일을 저장하는 트리구조 저장체계
- 파일은 text 파일과 binary 파일로 나눔

| Binary 파일                                                  | Text 파일                                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 컴퓨터만 이해할 수 있는 이진법 형식 파일, 메모장으로 내용 확인 불가, (ex) 엑셀, 워드 등 | 인간도 이해할 수 있는 문자열(ASCII/UNICODE) 형식 파일, 메모장으로 내용확인 가능, (ex) HTML, 코드 등 |

- 파이썬은 파일 처리를 위해 "open" 키워드 이용

```python
with open("<파일이름.확장자>", "<접근 모드>", encoding="<인코딩 종류>") as my_file: # with와 함께 사용하면 해당 구문 완료후 파일을 종료함.
    # encoding의 종류 : "utf8", "utf16", "cp949" 등
    contents = my_file.read() #my_file.readlines() : 파일 한줄 한줄을 list로 변환
    print(contents)
    my_file.write("기록 시작\n") # 기록 시작이라는 한줄이 파일에 적혀짐
    # with를 안쓰면 f.close()로 닫아줘야함.
```

- 접근 모드의 종류 
  - r : 읽기 모드, w: 쓰기 모드, a : 추가 모드

- **os, pathlib, shutil 모듈**을 이용하여 directory를 다룰 수 있음

```python
import os, pathlib, shutil
if not os.path.isdir("log"): #log directory가 아니면
    os.mkdir("log") # log 폴더를 생성하라.
if not os.path.exists("log/count_log.txt"): # log폴더 내에 count_log.txt 파일이 없으면
    f = open("log/count_log.txt", "w", encoding="utf8") #파일 생성 
	f.close
    os.path.isfile("log/count_log.txt") # 파일 존재 여부, true
   
cwd = pathlib.Path.cwd() # pathlib 모듈을 이용해 path를 객체로 다룰 수 있음
print(cwd) # WindowsPath('D:/workspace')
print(cwd.parent) # WindowsPath('D:/')
print(list(cwd.glob("*"))) # workspace 내의 폴더들...

source = "i_have_a_dream.txt"
dest = os.path.join("abc", "sungchul.txt") # abc\sungchul.txt 경로 저장
shutil.copy(source, dest) # source 파일을 dest로 파일 복사
# shutil : 파일 조작 간편한 모듈
```

- **Pickle**
  - 파이썬의 객체를 영속화(persistence)하는 built-in 객체
  - 데이터, object 등 실행중 정보를 저장-> 불러와서 사용
  - 저장해야하는 정보, 계산 결과(AI 모델) 등에 활용

```python
import pickle

class Multiply(object):
    def __init__(self, multiplier):
        self.multiplier = multiplier
        
    def multiply(self, number):
        return number * self.multiplier
    
multiply = Multiply(5)
print(multiply.multiply(10)) # 50
    
f = open("multiply_object.pickle", "wb")
pickle.dump(multiply, f) # 객체 저장
f.close()

f = open("multiply_object.pickle", "rb")
multiply_pickle = pickle.load(f) # 객체 불러오기
multiply_pickle.multiply(5) # 25
```


### Loggin Handling

- 로그 : 프로그램이 실행되는 동안 일어나는 정보를 기록에 남김
  - 유저의 접근, 프로그램 Exception, 특정함수 사용 등
  - 파일, DB의 형태로 남겨 분석하여 사용함
  - console 창에 print하는 것은 남지 않으므로 체계적인 로그 모듈 사용을 권장

```python
import logging

logger = logging.getLogger("main") # logger 선언
stream_handler = logging.StreamHandler() # logger output 방법 선언 콘솔에 출력하는 방법

stream_handler2 = logging.FileHandler(
	"my.log", mode="w", encoding="utf8" # my.log라는 파일에 저장
)

logger.addHandler(stream_handler) # logger output 등록

# logger.setLevel(logging.DEBUG) # 레벨 debug 급이후 부터만 로깅됨, 3.8버전 부터 바뀜
logging.basicConfig(levle=logging.DEBUG)  # 3.8버전 이후 
logger.debug("틀림") 
logger.info("확인해")
logger.warning("조심해")
logger.error("에러났어!!!")
logger.critical("망했다...")
# logger.setLevel(logging.CRITICAL) # 레벨 CRITICAL급이후 부터만 로깅됨, 3.8버전 부터 바뀜
```

- logging level 종류

| 레벨     | 개요                                                 | 예시                                       |
| -------- | ---------------------------------------------------- | ------------------------------------------ |
| debug    | 개발시 처리 기록을 남겨야하는 로그 정보를 남김       | A 객체 호출, 변수값 변경                   |
| info     | 처리가 진행되는 동안의 정보를 알림                   | 서버 시작, 서버 종료, 접속                 |
| warning  | 잘못 입력한 정보나 처리 가능한 의도치 않는 정보 알림 | 자료형 틀림, argument 타입 이상            |
| error    | 잘못된 처리로 인한 에러지만 프로그램 동작 가능       | 기록할 파일이 존재 하지 않음, 연결 불가 등 |
| critical | 데이터 손실이나 프로그램 동작 불가                   | 파일 삭제, 개인정보 유출, 강제종료 등      |

#### 프로그램 설정

- 데이터 파일 위치, 파일 저장 장소, Opertaion Type 등 여러가지를 설정해줘야함

1) configparser - 파일에 프로그램 설정해서 알려줌

	- Section, Key, Value 값의 형태로 설정된 설정 파일을 사용
	- 설정파일을 Dict Type으로 호출 후 사용

> configparser 출력 코드 예시

```python
import configparser
config = configparser.ConfigParser()
config.sections()

config.read('example.cfg')
config.sections()

for key in config['SectionOne']:
    print(key)

config['SectionOne']["status"]
```



> config 파일(exmaple.cfg) 출력 예시

```
[SecionOne]
Status: Single # key: Value 형식
Name: Derek
Value: Yes
Age: 30
Single: True

[SectionTwo]
FavoriteColor = Green # =을 사용해도됨 

[SectionThree]
FamilyName: Johnson
```



2) argparser - 실행시점에 프로그램 설정을 쉘에서 알려줌

- 거의 모든 Console 기반 Python 프로그램 기본 제공
- Command-Line Option 이라고도 부름

```python
import argparse

parser = argparse.ArgumentParser(description='Sum two inegers.')

parser.add_argument('-a', "--a_value", dest= "A_value", help="A integers", type=int)
parser.add_argument('-b', "--b_value", dest= "B_value", help="B integers", type=int)
# 짧은 이름, 긴 이름, 표시명, Help 설명, Argument Type

args = parser.parse_args()
print(args)
print(args.a)
print(args.b)
print(args.a + args.b)
```

> 포멧 제시
```python
# 로깅 방식시
formatter = logging.Formatter('%(asctime)s %(levelname)s %(process)d $(message)s')


# 파일 설정 방식시
logging.config.fileConfig('logging.conf')
logger = logging.getLogger()
logger.info('Open file {0}'.format("customers.csv",))

```

> logging 예시

```python

try:
    with open("customers.csv", "r") as customer_data:
        customer_reader = csv.reader(customer_data, delimiter=',', quotechar='"')
        for customer in customer_reader:
            if customer[10].upper() == "USA": #customer 데이터의 offset 10번째 값
                logger.info('ID {0} added'.format(customer[0],))
                customer_USA_only_list.append(customer) #즉 country 필드가 " USA " 것만
except: FileNotFoundError as e:
        logger.error('File Not Found {0}'.format(e,))
```

## Python Data handling

**1) CSV(Comma(or character) seprate values) Handling**

- 필드를 쉼표(,)로 구분한 텍스트 파일, 엑셀 양식의 데이터를 쓰기위해 사용
- 쉼표 대신 탭이나 빈칸으로 구분하기도 함으로 Character-separated values 라고도 부름
- 파이썬에서는 **파일 처리**를 이용하거나 **csv 모듈**을 활용 

> 파일처리 예제

```python
line_counter = 0
data_header = []
employee = []
customer_USA_only_list = []
customer = None

with open("customers.csv", "r") as customer_data:
    while 1:
        data = customer_data.readline()
        if not data:
            break
        if line_counter==0:
            data_header = data.split(",") # 데이터를 , 를 구분으로 나눔
        else:
            customer = data.split(",")
            if customer[10].uppper() == "USA": #customer 데이터의 offset 10번째 값
                customer_USA_only_list.append(customer) # 즉 country 필드가 "USA" 것만
        list_counter += 1 #customer_USA_only_list에 저장
print("Header:\t", data_header)
for i in range(0, 10):
    print("Data: \t\t", customer_USA_only_list[i])
print(len(customer_USA_only_list))

with open("customers_USA_only.csv", "w") as customer_USA_only_csv:
    for customer in customer_USA_only_list:
        customer_USA_only_csv.write(",".join(customer).strip('\n')+"\n")
        #customer_USA_only_list 객체에 있는 데이터를 customer_USA_only.csv 파일에 , 를 구분으로 쓰기   
```

``` python
import csv
reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
# delimiter: 글자를 나누는 기준
# linterminator: 줄 바꿈 기준 (기본값: \r\n)
# quotechar: 문자열을 둘러싸는 신호문자(기본값: ")
# quoting: 데이터를 나누는 기준이 quotechar에 의해 둘러쌓인 레벨(기본값: QUOTE_MINIMAL)
with open("파일 명.csv", "w", encoding="utf9") as f:
    writer = csv.writer(f, delimiter="\t", quotechar="'", quoting=csv.QUTOTE_ALL)
    writer.writerow(heade)# 제목 필드 헤더 쓰기
    for row in seung_namdata:
        write.writerow(row)# 한줄 씩 쓰기
```

**2) HTML Handling**

- HTML : 데이터 표시를 위한 텍스트 형식, 웹 상의 정보를 구조적으로 표현키 위한 언어
- 자세한 것은 web, HTML TIL 참조
- 최근 활발히 사용하고 있음, 규칙을 분석하여 방대한 양의 데이터를 분석 가능
- 정규식 (regular expression)
  - regexp 또는 regex 등으로 불림
  - 복잡한 문자열 패턴을 정의하는 문자 표현 공식
  - 특정한 규칙을 가진 문자열의 집합 추출
  - http://www.regexr.com/
  - 정규식 정리 참조

> 정규식 in 파이썬

```python
import re
import urlib.request

url = "https://bit.ly/3rxQFS4"
html = urllib.request.urlopen(url)
html_contents = str(html.read().decode("utf8")) #utf8으로 인코딩해서 문자열 추출
id_results = re.findall(r"([A-Za-z0-9]+\*\*\*)", html_contents) # findall 전체에서 패턴대로 데이터 찾기
for result in id_results:
	...    
```

- beautifulsoup4, selenium을 이용해도 쉽게 추출할 수 있다.

**3) XML Handling**

- eXtensible Markup Language, 데이터 구조와 의미를 설명하는 TAG(MarkUp)을 사용하여 표시하는 언어
- Tag와 Tag 사이에 값이 표시되고, 구조적인 정보를 표현 가능, HTML과 비슷함
- 정보의 구조에 대한 정보인 스키마와 DTD 등으로 정보에 대한 정보(메타정보)가 표현되며, 용도에 따라 다양한 형태로 변경 가능

> XML 예시

![1611285918373](Python_Basic.assets/1611285918373.png)

```xml
<?xml version="1.0"?>
<books>
	<book>
    	<author>Carson</author>
        <price
     format="dollar">31.95</price>
        <pubdate>05/01/2001</pubdate>
    </book>
    <pubinfo>
    	<publisher>MSPress</publisher>
        <state>WA</state>
    </pubinfo>
</books>
```

- BeautifulSoup
  - HTML, XML 등 Markup 언어 스크립팅을 위한 도구
  - lxml, html5lib 같은 parser 이용

> 파이썬에서의 beautifulsoup를 이용한 XML 처리

```python
from bs4 import BeautifulSoup #beautifulsoup 인스톨

wiht open("books.xml", "r", encoding="utf8") as books_file:
    books_xml = books_file.read() # File을 String으로 읽어오기
    
soup = BeautifulSoup(books_xml, "lxml") # lxml Parser를 사용해서 데이터 분석

# author가 들어간 모든 element 추출
for book_info in soup.find_all("author"): 
    print(book_info)
    print(book_info.get_text())
```





**4) JSON Handling**

- JavaScript Object Notation, 원래 웹 언어인 Java Script의 데이터 객체 표현 방식
- 간결하고 인간, 기계 둘다 이해 쉬움, 데이터 용량 적고, code로 전환 쉬움
- 최근 XML을 대체하는 중
- javascript의 dict 타입과 비슷

> JSON 예시

```json
{
    "employees":
    	[
            {
            	"name":"Shyam",
         		"email":"shyamjaiswal@gmail.com"
         	},
            {
            	"name":"Bob",
         		"email":"Bob32@gmail.com"
         	},
            {
            	"name":"Jai",
         		"email":"Jai87@gmail.com"
         	},
        ]
}
```



> Python Json 처리

- **json 모듈**을 사용하여 손쉽게 파싱 및 저장 가능

```python
import json
with open("json.json","r", encoding="utf8") as f: # json 불러오기
    contents = f.read()
    json_data = json.loads(contents)
    print(json_data["employees"])
     
with open("data.json", "w") as f: # json 파일 쓰기
    json.dump(json_data, f)
```



> 트위터 크롤링 예제

```python
import requests
from requests_oauthlib import OAuth2

consumer_key = ENV.KEY
consumer_secret = ENV.SECRET
access_token = ENV.TOKEN
access_token_secret= ENV.TOKEN_SECRET

oauth = OAuth2(client_key=consumer_key, client_secret=consuer_secret, resource_owner_key=access_token, resource_owner_secret=access_token_secret)

url="api.url"
r = requests.get(url=url,auth=oauth)
statuses= r.json()

for status in statuses:
    print(status['text'], status['created_at'])

```



[^1]: 각 나라별 언어를 모두 표현하기 위해 만든 통합 코드체계, 최대 65,536자를 표현 가능
[^2]: 변수의 자료형을 미리 선언하지 않고, 실행 시간에 값에 의해 결정
[^3]: 메모리 관리 기법 중의 하나로, 프로그램이 동적으로 할당했던 메모리 영역 중 필요없게 된 영역을 해제하는 기능
[^4]: 운영체제가 제공하는 함수의 집합체
[^5]: 응용 소프트웨어를 제어하는 프로그래밍 언어
[^6]: ; <-요거
[^7]: (Integrated Development Environment), 코드 편집기, 컴파일러( 인터프리터), 디버거 등 도구들이 통합되어 개발 생산성을 위한 SW
[^8]:  파이썬 개발 작업을 통합 관리하기 위한 논리적 개념 (파이썬 코드+ 리소스 파일)
[^9]: 컴퓨터 프로그램의 정확성이나 논리적인 오류를 찾아내는 테스트 과정, 자동화된 디버거 소프트 웨어가 필요함, 중단점 지정, 프로그램 실행정지, 메모리에 저장된 값 확인, 
[^10]: 소스 코드 상에서 내장 자료형의 상수값
[^11]: 소스 코드내에 사용할수 있도록 백슬래쉬(\\)기호와 조합해서 사용 하는 사전에 정의해둔 문자 조합, 문자열의 출력 결과를 제어하기 위해 사용함
[^12]: 문자열 내에 사용된 문자열 표시 유형을 특정값으로 변경하는 기법
[^13]: 파이썬에서 명령어나 연산자 등으로 사용하도록 되어있는 단어들
[^14]: 객체의 주소가 함수로 전달되는 방식, 전달된 객체를 참조해 변경시 호출자에게 영향을 주나 새로운 객체를 만들어 변수에 넣은 후로는 영향이 가지않는다.
