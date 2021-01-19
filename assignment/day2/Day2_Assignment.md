# 2일차 과제 정리

## basic-math

```python
def get_greatest(number_list):
    greatest_number = 0
    for i in number_list:
        greatest_number = i if i > greatest_number else greatest_number
    return greatest_number


def get_smallest(number_list):    
    smallest_number = 987654321
    for i in number_list:
        smallest_number = i if i < smallest_number else smallest_number
    return smallest_number


def get_mean(number_list):
    mean = 0
    for i in number_list:
        mean += i
    mean /= int(len(number_list))
    return mean


def get_median(number_list):   
    number_list.sort()    
    mid = len(number_list)//2
    if len(number_list) & 1:
        median = number_list[mid]
    else:
        median = sum(number_list[mid-1:mid+1])/2
    return median
```

- 제출 완료
- 각각 기능에 해당하는 라이브러리를 최대한 배제하고 구현하였다.
- 중간값에 대하여 찾아보았으며, 통계학적 의미에 대해서도 알아보았다.

## text-processing

```python
def normalize(input_string):
    input_string = input_string.lower().strip(' ')
    while("  " in input_string):
        input_string = input_string.replace('  ', ' ')
    normalized_string = input_string
    return normalized_string

def no_vowels(input_string):
    vowels = 'aeiou'
    upperVowels = vowels.upper()
    for vowel in vowels:
        input_string = input_string.replace(vowel, '')
    for vowel in upperVowels:
        input_string = input_string.replace(vowel, '')
    no_vowel_string = input_string
    return no_vowel_string

```

- 제출 완료
- no_vowels 함수의 경우 2~3번 정도 제출한 뒤 성공했는데, 대소문자 구분이 문제였다.
- 또한 문자열은 리스트로 변환한 뒤에야 원소의 값을 바꾸는 것이 가능하단걸 깨달았다.

## text-processing2

```python
def digits_to_words(input_string):
    words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    digit_string = ''
    for char in input_string:
        if char.isdigit():
            digit_string += words[int(char)] + ' '
    return digit_string.rstrip(' ')


def to_camel_case(underscore_str):
    if '_' in underscore_str:
        underscore_str = list(underscore_str.lower())
        for idx in range(len(underscore_str)):
            if underscore_str[idx] == '_':
                if (idx+1 < len(underscore_str)):
                    underscore_str[idx+1] = underscore_str[idx+1].upper()
        while('_' in underscore_str):
            underscore_str.remove('_')
        if (underscore_str):
            underscore_str[0] = underscore_str[0].lower()
        camelcase_str = ''.join(underscore_str)
    else:
        camelcase_str = underscore_str
    return camelcase_str
```

- 제출 완료
- to_camel_case 함수가 많이 아쉽다.
- 좀더 코드를 다듬고 적절한 내장함수를 쓰는게 맞았다.
  - 예를 들어 첫글자를 대문자로 바꿀때 titlize 함수를 쓰면 굳이 크기 체크와 리스트화가 필요 없을 것이다.

## baseball

```python
import random
import sys

sys.setrecursionlimit(15010)

def get_random_number():
    return random.randrange(100, 1000)


def is_digit(user_input_number):
    nums = ['1','2','3','4','5','6','7','8','9','0']
    for char in user_input_number:
        if char not in nums:
            return False
    result = True
    return result

def is_between_100_and_999(user_input_number):
    result = True if int(user_input_number) >= 100 and 1000 > int(user_input_number) else False
    return result


def is_duplicated_number(three_digit):
    three_digit = str(three_digit)
    for idx in range(len(three_digit)):
        if three_digit[idx] in three_digit[idx+1:]:
            return True
    result = False
    return result


def is_validated_number(user_input_number):
    user_input_number = str(user_input_number)
    result = is_digit(user_input_number) and is_between_100_and_999(user_input_number) and not is_duplicated_number(user_input_number)
    return result


def get_not_duplicated_three_digit_number():
    result = get_random_number()
    while(is_duplicated_number(result)):
        result = get_random_number()
    return result


def get_strikes_or_ball(user_input_number, random_number):
    result = [0,0]
    user_input_number = list(user_input_number)
    random_number = list(random_number)
    for i in range(3):
        if random_number[i] == user_input_number[i]:
            result[0] += 1
        elif user_input_number[i] in random_number:
            result[1] += 1
    return result

def is_yes(one_more_input):  
    one_more_input = one_more_input.upper()
    if one_more_input == "Y" or one_more_input == "YES":
        return True
    result = False
    return result


def is_no(one_more_input):
    one_more_input = one_more_input.upper()
    if one_more_input == "N" or one_more_input == "NO":
        return True
    result = False
    return result


def game():
    user_input = 999
    random_number = str(get_not_duplicated_three_digit_number())
    print("Random Number is : ", random_number)
    play(user_input, random_number)


def play(user_input, random_number):
    user_input = None
    while not is_validated_number(user_input):
        user_input = input('Input guess number : ')
        if user_input=='0':
            return
        if is_validated_number(user_input):
            break
        else:
            print('Wrong Input, Input again')
    stOrB = get_strikes_or_ball(user_input, random_number)
    print(f'Strikes : {stOrB[0]} , Balls : {stOrB[1]}')
    if (stOrB == [3, 0]):
        if not regame(stOrB): return
    else:
        play(user_input, random_number)


def regame(stOrB):
    user_input = input('You win, one more(Y/N) ?')
    if user_input=='0':
        return False
    if is_yes(user_input):
        return game()
    elif is_no(user_input):
        return False
    else:
        print(f'Wrong Input, Input again')
        return regame(stOrB)


def main():
    print("Play Baseball")
    game()
    print("Thank you for using this program")
    print("End of the Game")

if __name__ == "__main__":
    main()

```

- 제출 완료
- 솔직히 파이썬은 주 사용 언어라서 자신 있었지만 의외로 많은 시간이 걸린 과제였다.
- 대부분의 막혔던 이유는 
  - 코드를 제대로 읽지 않았거나
  - 문제를 제대로 읽지 않아서이다.

1. Stop Iteration Error
   - 에러 메시지만 보아서는 무슨 문제인지 알 수 없었던 에러라 시간을 잡아먹었다.
   - 파이썬의 Iterator들은 무한루프가 예상되면 해당 에러를 일으킨다고 한다.
   - 알고 보니 문제의 조건 중 하나인 0을 입력하면 종료되는 기능을 구현하지 않아, 0을 입력했을 때 무한 루프가 도는 것이였다.
   - 조건을 잘읽었으면 40분 정도 아꼈을 것이다.
2. Maximum recurrusion problem
   - 내 코드는 가독성을 위해 while문 대신 재귀를 이용해 재입력과 게임 재시작을 구현하였다.
   - 그러므로 게임을 재시작하거나 입력을 다시 받을 때마다 재귀가 쌓이며, 이론상 파이썬의 기본 최대 재귀 횟수인 500번의 재시작시 해당 에러가 나타난다.
   - 이 문제는 어느정도 예상하고 많이 있어본 문제라 디버깅은 쉬웠다. 최대 재귀 횟수를 15000회로 늘렸다.
   - 하지만 성능상에도 좋지않고 솔직히 코드의 가독성도 그리 좋진 않다.
3. 기타 버그
   - 출력할 문자열의 버그, 재시작 여부를 물을 때의 Wrong Input 질문 등을 구현하지 않은 등의 문제가 있었다.
   - 그다지 큰 버그도 아니였지만 모두 제대로 코드를 보고, 구현했더라면 이런일이 없었을 것이다.

## 총평

- 생각보다 모르는 것도 있었고 시간도 많이 걸렸다.
- 주 언어이고, 첫 과제라 우습게 봤는데 나의 부족함을 많이 느꼇다.