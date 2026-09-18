#!/usr/bin/env python # 인터프리터 경로 지정
# coding: utf-8 # 인코딩 설정

# # GAN으로 새로운 패션아이템 생성하기 # 제목 주석
# *GAN을 이용하여 새로운 패션 아이템을 만들어봅니다* # 설명 주석
#
# 이 프로젝트는 최윤제님의 파이토치 튜토리얼 사용 허락을 받아 참고했습니다. # 출처 명시
# * [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) - MIT License # 링크 및 라이선스 정보

import os # 운영체제 상호작용을 위한 라이브러리 임포트
import torch # 파이토치 라이브러리 임포트
import torchvision # 컴퓨터 비전용 라이브러리 임포트
import torch.nn as nn # 신경망 모듈 임포트
import torch.optim as optim # 최적화 알고리즘 모듈 임포트
from torchvision import transforms, datasets # 데이터 변환 및 데이터셋 임포트
from torchvision.utils import save_image # 이미지 저장 유틸리티 임포트
import matplotlib.pyplot as plt # 시각화 라이브러리 임포트
import numpy as np # 수치 계산 라이브러리 임포트


# EPOCHS 과 BATCH_SIZE 등 학습에 필요한 하이퍼 파라미터 들을 설정해 줍니다. # 주석

# 하이퍼파라미터 # 하이퍼파라미터 설정 섹션
EPOCHS = 500 # 전체 데이터셋 학습 반복 횟수
BATCH_SIZE = 100 # 한 번에 학습할 데이터 묶음 크기
USE_CUDA = torch.cuda.is_available() # GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if USE_CUDA else "cpu") # 사용할 장치(GPU 또는 CPU) 설정
print("Using Device:", DEVICE) # 현재 사용 중인 장치 출력


# 학습에 필요한 데이터셋을 로딩합니다. # 주석

# Fashion MNIST 데이터셋 # 데이터셋 로딩 및 전처리 섹션
trainset = datasets.FashionMNIST( # Fashion MNIST 학습 데이터셋 불러오기
    './.data', # 데이터 저장 경로
    train=True, # 학습용 데이터 선택
    download=True, # 데이터가 없으면 다운로드
    transform=transforms.Compose([ # 이미지 변환 설정
       transforms.ToTensor(), # 이미지를 텐서로 변환
       transforms.Normalize((0.5,), (0.5,)) # 평균 0.5, 표준편차 0.5로 정규화
    ])
)
train_loader = torch.utils.data.DataLoader( # 데이터를 배치 단위로 제공하는 로더 설정
    dataset     = trainset, # 대상 데이터셋
    batch_size  = BATCH_SIZE, # 배치 크기
    shuffle     = True # 데이터를 무작위로 섞음
)


# 생성자는 64차원의 랜덤한 텐서를 입력받아 이에 행렬곱(Linear)과 활성화 함수(ReLU, Tanh) 연산을 실행합니다. # 주석
# 생성자의 결과값은 784차원, 즉 Fashion MNIST 속의 이미지와 같은 차원의 텐서입니다. # 주석

# 생성자 (Generator) # 생성자 모델 정의 (Sequential 방식)
G = nn.Sequential(
        nn.Linear(64, 256), # 64차원 노이즈 입력을 256개 노드로 연결
        nn.ReLU(), # 활성화 함수 ReLU
        nn.Linear(256, 256), # 256개 노드를 256개 노드로 연결
        nn.ReLU(), # 활성화 함수 ReLU
        nn.Linear(256, 784), # 256개 노드를 784(28x28)개 노드로 연결
        nn.Tanh()) # 출력값을 -1에서 1 사이로 압축


# 판별자는 784차원의 텐서를 입력받습니다. 판별자 역시 입력된 데이터에 행렬곱과 활성화 함수를 실행시키지만, # 주석
# 생성자와 달리 판별자의 결과값은 입력받은 텐서가 진짜인지 구분하는 예측값입니다. # 주석

# 판별자 (Discriminator) # 판별자 모델 정의 (Sequential 방식)
D = nn.Sequential(
        nn.Linear(784, 256), # 784(이미지) 입력을 256개 노드로 연결
        nn.LeakyReLU(0.2), # 활성화 함수 LeakyReLU (기울기 0.2)
        nn.Linear(256, 256), # 256개 노드를 256개 노드로 연결
        nn.LeakyReLU(0.2), # 활성화 함수 LeakyReLU
        nn.Linear(256, 1), # 256개 노드를 1개 노드로 연결 (진짜/가짜 판별)
        nn.Sigmoid()) # 결과값을 0에서 1 사이의 확률값으로 변환


# 생성자와 판별자 학습에 쓰일 오차 함수와 최적화 알고리즘도 정의해 줍니다. # 주석

# 모델의 가중치를 지정한 장치로 보내기 # 장치 할당
D = D.to(DEVICE) # 판별자를 GPU/CPU로 전송
G = G.to(DEVICE) # 생성자를 GPU/CPU로 전송

# 이진 크로스 엔트로피 (Binary cross entropy) 오차 함수와 # 손실 함수 및 최적화 도구 설정
# 생성자와 판별자를 최적화할 Adam 모듈
criterion = nn.BCELoss() # 이진 분류 손실 함수
d_optimizer = optim.Adam(D.parameters(), lr=0.0002) # 판별자용 Adam 최적화 도구
g_optimizer = optim.Adam(G.parameters(), lr=0.0002) # 생성자용 Adam 최적화 도구


# 모델 학습에 필요한 준비는 끝났습니다. 그럼 본격적으로 GAN을 학습시키는 loop을 만들어 보겠습니다. # 주석

total_step = len(train_loader) # 배치 개수 확인
for epoch in range(EPOCHS): # 정해진 에폭만큼 반복
    for i, (images, _) in enumerate(train_loader): # 배치 단위 이미지 로드 (레이블은 사용 안 함)
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE) # 이미지를 평탄화하여 장치로 전송

        # '진짜'와 '가짜' 레이블 생성 # 정답지 생성
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE) # 진짜는 1
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE) # 가짜는 0

        # 판별자가 진짜 이미지를 진짜로 인식하는 오차를 예산 # 진짜 이미지 판별 학습
        outputs = D(images) # 진짜 이미지를 판별자에 입력
        d_loss_real = criterion(outputs, real_labels) # 진짜 판별 결과와 정답(1) 사이의 오차
        real_score = outputs # 결과 점수 저장

        # 무작위 텐서로 가짜 이미지 생성 # 가짜 이미지 생성
        z = torch.randn(BATCH_SIZE, 64).to(DEVICE) # 64차원 노이즈 생성
        fake_images = G(z) # 생성자를 통해 가짜 이미지 생성

        # 판별자가 가짜 이미지를 가짜로 인식하는 오차를 계산 # 가짜 이미지 판별 학습
        outputs = D(fake_images) # 가짜 이미지를 판별자에 입력
        d_loss_fake = criterion(outputs, fake_labels) # 가짜 판별 결과와 정답(0) 사이의 오차
        fake_score = outputs # 결과 점수 저장

        # 진짜와 가짜 이미지를 갖고 낸 오차를 더해서 판별자의 오차 계산 # 판별자 총 오차 합산
        d_loss = d_loss_real + d_loss_fake

        # 역전파 알고리즘으로 판별자 모델의 학습을 진행 # 판별자 가중치 업데이트
        d_optimizer.zero_grad() # 판별자 기울기 초기화
        g_optimizer.zero_grad() # 생성자 기울기 초기화
        d_loss.backward() # 역전파
        d_optimizer.step() # 판별자 업데이트

        # 생성자가 판별자를 속였는지에 대한 오차를 계산 # 생성자 학습 준비
        fake_images = G(z) # 가짜 이미지 재생성
        outputs = D(fake_images) # 가짜 이미지를 판별자에 입력
        g_loss = criterion(outputs, real_labels) # 판별자가 가짜를 진짜(1)로 믿는지 오차 계산

        # 역전파 알고리즘으로 생성자 모델의 학습을 진행 # 생성자 가중치 업데이트
        d_optimizer.zero_grad() # 기울기 초기화
        g_optimizer.zero_grad() # 기울기 초기화
        g_loss.backward() # 역전파
        g_optimizer.step() # 생성자 업데이트

    # 학습 진행 알아보기 # 에폭마다 학습 결과 출력
    print('Epoch [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
          .format(epoch, EPOCHS, d_loss.item(), g_loss.item(),
                  real_score.mean().item(), fake_score.mean().item()))


# ## 학습한 모델 저장하기 # 모델 저장 섹션
#
# 학습이 끝난 생성자(`G`)와 판별자(`D`)의 가중치를 파일로 저장합니다. # 주석
# `state_dict()`만 저장하는 것이 권장되는 방식입니다 (모델 구조는 코드에 있으므로). # 주석

# 저장 경로 준비 # 디렉토리 생성 및 경로 설정
os.makedirs('./checkpoints', exist_ok=True) # checkpoints 폴더 생성 (이미 있으면 통과)
G_PATH = './checkpoints/gan_generator.pth' # 생성자 가중치 저장 경로
D_PATH = './checkpoints/gan_discriminator.pth' # 판별자 가중치 저장 경로

# state_dict 저장 # 실제 파일 저장
torch.save(G.state_dict(), G_PATH) # 생성자 가중치 저장
torch.save(D.state_dict(), D_PATH) # 판별자 가중치 저장

print(f"생성자 저장: {G_PATH}") # 저장 완료 메시지 출력
print(f"판별자 저장: {D_PATH}") # 저장 완료 메시지 출력


# 학습이 끝난 생성자의 결과물을 한번 확인해 보겠습니다. # 시각화 테스트 섹션

z = torch.randn(BATCH_SIZE, 64).to(DEVICE) # 노이즈 생성
fake_images = G(z) # 생성자로 이미지 생성
for i in range(10): # 생성된 이미지 중 10개 출력
    fake_images_img = np.reshape(fake_images.data.cpu().numpy()[i], (28, 28)) # 28x28로 형태 변경
    plt.imshow(fake_images_img, cmap='gray') # 회색조 표시
    plt.show() # 출력


# ## 저장한 모델 불러와서 결과물 표시하기 # 모델 로드 및 테스트 섹션
#
# 학습된 가중치 파일을 다시 읽어와 새 생성자 모델 `G_loaded`에 적용한 뒤, # 주석
# 무작위 잠재 벡터로 이미지를 생성해 격자 형태로 표시합니다. # 주석

# 동일한 구조의 새 생성자 정의 # 로드용 모델 인스턴스 생성
G_loaded = nn.Sequential(
    nn.Linear(64, 256), # 64 -> 256
    nn.ReLU(), # 활성화 함수
    nn.Linear(256, 256), # 256 -> 256
    nn.ReLU(), # 활성화 함수
    nn.Linear(256, 784), # 256 -> 784
    nn.Tanh()).to(DEVICE) # Tanh 활성화 및 장치 전송

# 저장된 가중치 로드 # 가중치 덮어씌우기
G_loaded.load_state_dict(torch.load(G_PATH, map_location=DEVICE)) # 파일에서 읽어오기
G_loaded.eval()  # 추론(테스트) 모드로 설정
print(f"생성자 가중치 로드 완료: {G_PATH}") # 완료 메시지

# 잠재 벡터 → 이미지 생성 # 데이터 생성
N_SAMPLES = 16 # 생성할 샘플 개수
with torch.no_grad(): # 기울기 계산 비활성화
    z = torch.randn(N_SAMPLES, 64).to(DEVICE) # 무작위 노이즈 생성
    fake_images = G_loaded(z).cpu().numpy() # 이미지 생성 및 넘파이 변환

# 4x4 격자로 표시 # 결과 시각화
fig, axes = plt.subplots(4, 4, figsize=(8, 8)) # 4x4 서브플롯 생성
for i, ax in enumerate(axes.flatten()): # 각 칸마다 반복
    img = fake_images[i].reshape(28, 28) # 28x28 재구성
    ax.imshow(img, cmap='gray') # 이미지 표시
    ax.axis('off') # 축 정보 끄기
plt.suptitle('Loaded Generator Output', fontsize=14) # 상단 제목 추가
plt.tight_layout() # 간격 자동 조정
plt.show() # 최종 출력
