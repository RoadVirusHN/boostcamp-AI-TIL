# 2일차 과제 정리

## basic-math

```python
# -*- coding: utf8 -*-

def is_help_command(user_input):
    result = user_input.upper() == "H" or user_input.upper() == "HELP"    
    return result


def is_validated_english_sentence(user_input):
    sp = "_@#$%^&*()-+=[]\{\}\"\';:\|`~"
    allowedSp = ".,!? "
    for char in allowedSp:        
        user_input = user_input.replace(char,'')
    result = user_input
    for char in user_input:
        if char.isdigit() or char in sp:
            return False
    return result

def is_validated_morse_code(user_input):
    result = True
    allowed = "-. "
    codes = user_input.split()
    morses = get_morse_code_dict()
    for code in codes:
        for char in code:
            if char not in allowed:
                return False
        for morse in morses.values():
            if code == morse:
                break
        else:
            return False
    return result

def get_cleaned_english_sentence(raw_english_sentence):
    notAllowed = ".,!?"
    for char in notAllowed:
        raw_english_sentence = raw_english_sentence.replace(char,'')
    result = raw_english_sentence.strip('')
    return result


def decoding_character(morse_character):
    reversedMorse = {v:k for k, v in morse_code_dict.items()}
    return reversedMorse[morse_character]


def encoding_character(english_character):
    morse_code_dict = get_morse_code_dict()
    return morse_code_dict[english_character.upper()]


def decoding_sentence(morse_sentence):
    result = ''    
    for code in morse_sentence.split(' '):
        if code == '':
            result += ' '
        else:
            result += decoding_character(code.strip())
    return result

def encoding_sentence(english_sentence):
    english_sentence = get_cleaned_english_sentence(english_sentence)
    english_sentence = list(english_sentence)
    result = ''
    for char in english_sentence:
        if char==' ':
            result += ' '
        else:
            result += encoding_character(char.upper()) + " "
    tripleSpace = '   '
    while tripleSpace in result:
        result = result.replace(tripleSpace, '  ')
    return result.strip()
def main():
    print("Morse Code Program!!")
    while True:
        user_input = input('Input your message(H - Help, 0 - Exit): ')
        if user_input == '0':
            break
        elif is_help_command(user_input):
            print(get_help_message())
        elif is_validated_english_sentence(user_input):
            print(encoding_sentence(get_cleaned_english_sentence(user_input)))
        elif is_validated_morse_code(user_input):
            print(decoding_sentence(user_input))
        else:
            print('Wrong Input')
    print("Good Bye")
    print("Morse Code Program Finished!!")

if __name__ == "__main__":
    main()

```

- 제출 완료

## 총평

- 이번에는 과제가 쉽고 양이 적으며, 디버그 환경에 대한 경험이 쌓여 쉬웠다.

- 하지만 코드 리뷰에서 다른 사람들보다 길고 깔끔하지 않은 코드를 짯다는게 눈에 보였다.
- 코드가 문제없이 돌아간다면 그 이후로 리펙토링과 클린 코딩을 하지않는 버릇을 버려야한다.

- 일부 함수 성능에 대한 질문을 올렸다. 해당 답변은 내일쯤 돌아올 것 같다.