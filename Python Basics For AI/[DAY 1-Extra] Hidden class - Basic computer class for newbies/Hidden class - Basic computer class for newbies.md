[TOC]

# Hidden class - Basic computer class for newbies

- 모바일에만 익숙한 젊은 컴맹들을 위한 기초 강의

## Day 0: File System & Terminal Basic

### 1. 컴퓨터 OS

- **OS(운영체제, Operating System)** : 우리의 프로그램이 동작할 수 있는 구동 환경
- OS에 의존적인 소프트웨어와 자원을 제공하는 하드웨어를 연결하는 기반.
- 윈도우즈, MAC, Linux, 안드로이드 등이 있음.

### 2. 파일 시스템

- OS가 파일을 저장하는 트리구조 저장 체계
- **File(파일)** : 컴퓨터 등의 기기에서 의미 있는 정보를 담는 논리적인 단위.
- 파일은 파일명과 확장자로 식별됨.
- Directory(디렉토리 또는 폴더,Folder) : 파일들을 포함하는 상위 경로, 윈도우에선 폴더라고 부른다.
- root 디렉토리 : 윈도우의 경우 C드라이브
- **절대 경로**: 루트 디렉 토리부터 파일 위치까지의 경로
- **상대 경로**: 현재 있는 디렉토리부터 타깃 파일까지의 경로(../: 상위 경로, ./:현재 경로)

### 3. 터미널

- **터미널(terminal, 또는 CLI(Command Line Interface))**: mouse(GUI 환경)이 아닌 TEXT로 명령을 입력하여 프로그램을 실행하는 환경.
- Windows Terminal, Terminal 등이 존재.
- Console, CMD라고도 부름.
- 윈도우 키 + R로 CMD 실행 가능

#### 명령어x

- 모든 명령어들은 **[명령어] -h, [명령어] –help, man [명령어]**를 통하여 사용례와 설명을 볼 수 있다.

- -나 –로 시작하는 argument는 보통 **flag, option** 이라고 칭하며, 보통 필수가 아니거나 미리 정의된 설정 등을 사용하게 해준다.(man을 통해 doc을 열 경우 q를 통해 나갈 수 있다)

- $는 root user가 아님을 의미, 권한이 제한됨

- **cd [directory]**: shell 명령어, shell에서 현재 위치를 바꾸는데 사용
  - "."은 현재 위치, ".."은 상위 폴더를 의미
  - ., ..과 폴더명을 이용하여 이동한다.
  - 보통 ls와 함께 사용한다.

> cd 명령어 예제

```
RoadVirusHN@DESKTOP-1UVCFH9 MINGW64 ~
$ cd Desktop/
RoadVirusHN@DESKTOP-1UVCFH9 MINGW64 ~/Desktop
$ cd ..
RoadVirusHN@DESKTOP-1UVCFH9 MINGW64 ~
$ cd ./Desktop/
```


- **ls [option]** : 현재 폴더 위치, 또는 인수로 주어진 위치의 내부 파일, 폴더 정보를 출력해줌

  - **-l flag**를 추가하면 더욱더 상세정보를 얻을 수 있다.

  - drwxr-xr-x에서 d는 directory, rwx는 현재 사용자가 해당 폴더(파일)에 가지고 있는 권한을 의미한다.

> ls 명령어 예제

```
RoadVirusHN@DESKTOP-1UVCFH9 MINGW64 ~/AppData
$ ls
Local/  LocalLow/  Roaming/
RoadVirusHN@DESKTOP-1UVCFH9 MINGW64 ~
$ ls -l ~
total 17888
drwxr-xr-x 1 RoadVirusHN 197121        0  5월  8  2019  3d/
drwxr-xr-x 1 RoadVirusHN 197121        0  4월 16  2019  ansel/
drwxr-xr-x 1 RoadVirusHN 197121        0  8월 30  2019  AppData/
lrwxrwxrwx 1 RoadVirusHN 197121       36  8월 30  2019 'Application Data' -> /c/Users/RoadVirusHN/AppData/Roaming/
-rw-r--r-- 1 RoadVirusHN 197121       53  2월 26 23:28  useruid.ini
-rw-r--r-- 1 RoadVirusHN 197121 11534336  7월 21 07:23  NTUSER.DAT
```


-  **mv [source] [dierctory/name]**: source 파일을 directory로 옮기거나 이름을 바꾸는데 쓴다.
- **cp [source] [directory/name]**': source 파일을 directory로 복사, 또는 이름을 바꿔 복사
- **mkdir [directory/name]**: 해당 directory로 name의 폴더 생성
- **touch [directory/name.type]** : 해당 directory로 name이름의 type확장자의 폴더 생성
- **cat [filename]** : 해당 파일의 내용을 출력한다.

- 기타 등등..

  - date 명령어는 현재 시간을 표시해준다.
  - pwd 명령어는 내 현재 위치의 절대 위치를 알려준다.

  - echo는 argument로 주어진 값을 출력하는 프로그램

	#### 명령어들의 위치와 환경변수

- **$PATH**를 통해 내가 지정해놓은 명령어 프로그램들이 있는 directory를 볼 수 있다.
- 해당 directory에 command 프로그램을 놓으면 활용할 수 있으며, **환경변수(Enviroment variable)**이라고 한다.(":"는 directory 간의 구분자)
- which 명령어를 통해 해당 프로그램의 위치를 알 수 있다.