# Python DevEnv setting

## Window Setting

1. Anaconda installer 다운로드

   ![1611483500673](Python_DevEnv_setting.assets/1611483500673.png)

2. Anaconda 설치

   ![1611486147071](Python_DevEnv_setting.assets/1611486147071.png)

   - Just Me로 설치

     ![1611486476654](Python_DevEnv_setting.assets/1611486476654.png)

   - 아래만 체크한 뒤 설치

3. VScode 터미널에 source 추가

   ```bash
   conda -V # conda 설치 확인, 생략 가능
   set PATH=%PATH%;<your_path_to_anaconda_installation>\Scripts
   source <your_path_to_anaconda_installation>c/profile.d/conda.sh
   conda init <사용하는 터미널>
   ```

   - 설치 경로가 다르면 경로를 다르게 바꾸어야한다.
   - 이래도 안될경우 anaconda3 prompt에서 해당 경로에서 "code"를 쳐서 들어간 뒤 해보자.

4. conda 가상 환경 설정 후, pytorch 및 추가 모듈 설치

```bash
conda create --name <가상환경명>
conda activate <가상환경명>
conda install pytorch, matplotlib # 기타 추가 모듈
conda deactivate # 가상환경 나가기
```

## ubuntu Setting

- Window 내 WSL을 이용한 ubuntu 및 vscode 세팅

1. 윈도우 10 내의 VM과 WSL 기능 활성화
> Powershell에 입력할 명령어

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
 ![1610955445550](Python_DevEnv_setting.assets/1610955445550.png)

2. 윈도우 스토어에서 **우분투** 최신버전 설치 및 실행 


> '무료'를 눌러 설치, 이후 '실행'을 눌러 실행

![1610955527812](Python_DevEnv_setting.assets/1610955527812.png)


> '실행' 시, 화면

![1610955835070](Python_DevEnv_setting.assets/1610955835070.png)


3. 우분투 내 **apt** 업데이트 및 **curl** 설치

> sudo를 통해 root 권한으로 실행


```bash
sudo apt update
sudo apt-get install curl
```

 ![1610955993568](Python_DevEnv_setting.assets/1610955993568.png)
 ![1610956152470](Python_DevEnv_setting.assets/1610956152470.png)

4. 아나콘다 최신버전 설치

>anaconda 공식 페이지에서 linux 최신 배포판을 다운로드 (대문자 O), 원하는 다이렉토리에서 실행

```bash
sudo curl -O https://repo.anaconda.com/archive/Anaconda3-[최신버전]-Linux-x86_64.sh
```

![1610956822743](Python_DevEnv_setting.assets/1610956822743.png)


> curl로 받은 설치파일이 있는 곳에서 명령어 입력
```bash
sudo sha256sum Anaconda3-[최신버전]-Linux-x86_64.sh
sudo bash Anaconda3-[최신버전]-Linux-x86_64.sh
```

> 배포판 파일의 설치를 확인하는 명령어

![1610956859259](Python_DevEnv_setting.assets/1610956859259.png)

> 배포판 파일의 설치를 확인하는 명령어

![1610956913044](Python_DevEnv_setting.assets/1610956913044.png)

![1610957024259](Python_DevEnv_setting.assets/1610957024259.png)

> 'conda -version'으로 설치 결과를 확인

![1610958670745](Python_DevEnv_setting.assets/1610958670745.png)

6. VScode 공식 홈페이지에서 Visual Studio code 다운로드

> 아쉽게도 윈도우 wsl판 ubuntu는 Linux 내부의 VSCode Editor를 아직 지원하지 않는다고 한다.

![1610958171804](Python_DevEnv_setting.assets/1610958171804.png)

7. Remote -WSL Extension 다운로드

![1610955236911](Python_DevEnv_setting.assets/1610955236911.png)

![1610958148746](Python_DevEnv_setting.assets/1610958148746.png)

> Ctrl + ` 으로 VScode 내 terminal을 열고 code .으로 열 수 있다.

![1610963586890](Python_DevEnv_setting.assets/1610963586890.png)

## 실행 화면
![1610959173109](Python_DevEnv_setting.assets/1610959173109.png)

