#!/usr/bin/env python # 인터프리터 경로 지정
# coding: utf-8 # 인코딩 설정

# # cGAN으로 생성 제어하기 # 제목 주석

import os # 운영체제 상호작용을 위한 라이브러리 임포트
import torch # 파이토치 라이브러리 임포트
import torchvision # 컴퓨터 비전용 라이브러리 임포트
import torch.nn as nn # 신경망 모듈 임포트
import torch.optim as optim # 최적화 알고리즘 모듈 임포트
from torchvision import transforms, datasets # 데이터 변환 및 데이터셋 임포트
from torchvision.utils import save_image # 이미지 저장 유틸리티 임포트
import matplotlib.pyplot as plt # 시각화 라이브러리 임포트
import numpy as np # 수치 계산 라이브러리 임포트


# 하이퍼파라미터 # 하이퍼파라미터 설정 섹션
EPOCHS = 10 # 학습 반복 횟수 설정 (원래 300이었으나 10으로 조정됨)
BATCH_SIZE = 100 # 한 번에 학습할 데이터 묶음 크기
USE_CUDA = torch.cuda.is_available() # GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if USE_CUDA else "cpu") # 사용할 장치(GPU 또는 CPU) 설정
print("Using Device:", DEVICE) # 현재 사용 중인 장치 출력


# Fashion MNIST 데이터셋 # 데이터셋 로딩 및 전처리 섹션
trainset = datasets.FashionMNIST( # Fashion MNIST 학습 데이터셋 불러오기
    './.data', # 데이터 저장 경로
    train=True, # 학습용 데이터 선택
    download=True, # 데이터가 없으면 다운로드
    transform=transforms.Compose([ # 이미지 변환 설정 (체인 형태)
       transforms.ToTensor(), # 이미지를 텐서로 변환
       transforms.Normalize((0.5,), (0.5,)) # 평균 0.5, 표준편차 0.5로 정규화
    ])
)
train_loader = torch.utils.data.DataLoader( # 데이터를 배치 단위로 제공하는 로더 설정
    dataset     = trainset, # 대상 데이터셋
    batch_size  = BATCH_SIZE, # 배치 크기
    shuffle     = True # 매 에폭마다 데이터를 섞음
)


# 생성자 (Generator) # 생성자 모델 정의 섹션
class Generator(nn.Module): # nn.Module을 상속받아 생성자 클래스 정의
    def __init__(self): # 초기화 메서드
        super().__init__() # 부모 클래스 초기화
        
        self.embed = nn.Embedding(10, 10) # 10개의 레이블을 각각 10차원 벡터로 변환하는 임베딩 층
        
        self.model = nn.Sequential( # 신경망 층을 순차적으로 구성
            nn.Linear(110, 256), # 입력(노이즈 100 + 레이블 10)을 256개 노드로 연결
            nn.LeakyReLU(0.2, inplace=True), # 활성화 함수 LeakyReLU (음수 기울기 0.2)
            nn.Linear(256, 512), # 256개 노드를 512개 노드로 연결
            nn.LeakyReLU(0.2, inplace=True), # 활성화 함수 LeakyReLU
            nn.Linear(512, 1024), # 512개 노드를 1024개 노드로 연결
            nn.LeakyReLU(0.2, inplace=True), # 활성화 함수 LeakyReLU
            nn.Linear(1024, 784), # 1024개 노드를 784(28x28)개 노드로 연결
            nn.Tanh() # 출력값을 -1에서 1 사이로 압축하는 Tanh 활성화 함수
        )
    
    def forward(self, z, labels): # 순전파 메서드
        c = self.embed(labels) # 레이블 정보를 임베딩 벡터로 변환
        x = torch.cat([z, c], 1) # 노이즈 벡터(z)와 레이블 벡터(c)를 이어붙임
        return self.model(x) # 모델을 통과시켜 이미지 데이터 생성


# 판별자 (Discriminator) # 판별자 모델 정의 섹션
class Discriminator(nn.Module): # nn.Module을 상속받아 판별자 클래스 정의
    def __init__(self): # 초기화 메서드
        super().__init__() # 부모 클래스 초기화
        
        self.embed = nn.Embedding(10, 10) # 레이블 정보를 위한 임베딩 층
        
        self.model = nn.Sequential( # 신경망 층을 순차적으로 구성
            nn.Linear(794, 1024), # 입력(이미지 784 + 레이블 10)을 1024개 노드로 연결
            nn.LeakyReLU(0.2, inplace=True), # 활성화 함수 LeakyReLU
            nn.Dropout(0.3), # 과적합 방지를 위한 드롭아웃 (30% 확률)
            nn.Linear(1024, 512), # 1024개 노드를 512개 노드로 연결
            nn.LeakyReLU(0.2, inplace=True), # 활성화 함수 LeakyReLU
            nn.Dropout(0.3), # 드롭아웃
            nn.Linear(512, 256), # 512개 노드를 256개 노드로 연결
            nn.LeakyReLU(0.2, inplace=True), # 활성화 함수 LeakyReLU
            nn.Dropout(0.3), # 드롭아웃
            nn.Linear(256, 1), # 256개 노드를 1개 노드로 연결 (진짜/가짜 판별)
            nn.Sigmoid() # 결과값을 0(가짜)에서 1(진짜) 사이의 확률값으로 변환
        )
    
    def forward(self, x, labels): # 순전파 메서드
        c = self.embed(labels) # 레이블 정보를 임베딩 벡터로 변환
        x = torch.cat([x, c], 1) # 이미지 데이터(x)와 레이블 벡터(c)를 이어붙임
        return self.model(x) # 모델을 통과시켜 판별 결과 반환


# 모델 인스턴스를 만들고 모델의 가중치를 지정한 장치로 보내기 # 모델 초기화 및 장치 할당
D = Discriminator().to(DEVICE) # 판별자 인스턴스 생성 및 GPU/CPU로 전송
G = Generator().to(DEVICE) # 생성자 인스턴스 생성 및 GPU/CPU로 전송

# 이진 교차 엔트로피 함수와 # 손실 함수 및 최적화 도구 설정
# 생성자와 판별자를 최적화할 Adam 모듈
criterion = nn.BCELoss() # 이진 분류를 위한 손실 함수 설정
d_optimizer = optim.Adam(D.parameters(), lr =0.0002) # 판별자용 Adam 최적화 알고리즘
g_optimizer = optim.Adam(G.parameters(), lr =0.0002) # 생성자용 Adam 최적화 알고리즘


total_step = len(train_loader) # 전체 데이터셋의 배치 개수
for epoch in range(EPOCHS): # 정해진 에폭만큼 반복 학습
    for i, (images, labels) in enumerate(train_loader): # 배치 단위로 데이터 꺼내기
        images = images.reshape(BATCH_SIZE, -1).to(DEVICE) # 이미지를 1차원 벡터로 평탄화하고 장치로 전송
        
        # '진짜'와 '가짜' 레이블 생성 # 학습을 위한 정답지(레이블) 생성
        real_labels = torch.ones(BATCH_SIZE, 1).to(DEVICE) # 진짜 이미지는 1로 설정
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(DEVICE) # 가짜 이미지는 0으로 설정

        # 판별자가 진짜 이미지를 진짜로 인식하는 오차 계산 (데이터셋 레이블 입력) # 진짜 이미지 판별 학습
        labels = labels.to(DEVICE) # 레이블을 장치로 전송
        outputs = D(images, labels) # 진짜 이미지와 레이블을 판별자에 입력
        d_loss_real = criterion(outputs, real_labels) # 진짜 판별 결과와 실제 정답(1) 사이의 오차 계산
        real_score = outputs # 나중에 출력을 위해 점수 저장
    
        # 무작위 텐서와 무작위 레이블을 생성자에 입력해 가짜 이미지 생성 # 가짜 이미지 생성
        z = torch.randn(BATCH_SIZE, 100).to(DEVICE) # 무작위 노이즈 벡터 생성
        g_label = torch.randint(0, 10, (BATCH_SIZE,)).to(DEVICE) # 무작위 레이블 생성
        fake_images = G(z, g_label) # 생성자를 통해 가짜 이미지 생성
        
        # 판별자가 가짜 이미지를 가짜로 인식하는 오차 계산 # 가짜 이미지 판별 학습
        outputs = D(fake_images, g_label) # 가짜 이미지와 해당 레이블을 판별자에 입력
        d_loss_fake = criterion(outputs, fake_labels) # 가짜 판별 결과와 가짜 정답(0) 사이의 오차 계산
        fake_score = outputs # 나중에 출력을 위해 점수 저장
        
        # 진짜와 가짜 이미지를 갖고 낸 오차를 더해서 판별자의 오차 계산 # 판별자 총 오차 합산
        d_loss = d_loss_real + d_loss_fake
        
        # 역전파 알고리즘으로 판별자 모델의 학습을 진행 # 판별자 가중치 업데이트
        d_optimizer.zero_grad() # 기울기 초기화
        g_optimizer.zero_grad() # 생성자 기울기도 초기화 (안정성을 위해)
        d_loss.backward() # 역전파로 기울기 계산
        d_optimizer.step() # 판별자 가중치 업데이트
        
        # 생성자가 판별자를 속였는지에 대한 오차 계산(무작위 레이블 입력) # 생성자 학습을 위한 준비
        fake_images = G(z, g_label) # 가짜 이미지 재생성
        outputs = D(fake_images, g_label) # 가짜 이미지를 다시 판별자에 입력
        g_loss = criterion(outputs, real_labels) # 판별자가 가짜를 진짜(1)로 속았는지 오차 계산

        # 역전파 알고리즘으로 생성자 모델의 학습을 진행 # 생성자 가중치 업데이트
        d_optimizer.zero_grad() # 판별자 기울기 초기화
        g_optimizer.zero_grad() # 생성자 기울기 초기화
        g_loss.backward() # 역전파로 기울기 계산
        g_optimizer.step() # 생성자 가중치 업데이트
        
    print('이폭 [{}/{}] d_loss:{:.4f} g_loss: {:.4f} D(x):{:.2f} D(G(z)):{:.2f}' # 학습 진행 상황 출력
          .format(epoch,
                  EPOCHS,
                  d_loss.item(),
                  g_loss.item(),
                  real_score.mean().item(),
                  fake_score.mean().item()))


# 만들고 싶은 아이템 생성하고 시각화하기 # 학습 결과 테스트 및 시각화 섹션
item_number = 9 # 생성하고 싶은 아이템 번호 (예: 9번은 앵클 부츠)
z = torch.randn(1, 100).to(DEVICE) # 테스트용 무작위 노이즈 (배치 크기 1)
g_label = torch.full((1,), item_number, dtype=torch.long).to(DEVICE) # 지정한 아이템 번호 레이블 생성
sample_images = G(z, g_label) # 생성자를 이용해 이미지 생성

sample_images_img = np.reshape(sample_images.data.cpu().numpy() # 텐서를 넘파이 배열로 변환하고 형태 조정
                               [0],(28, 28))
plt.imshow(sample_images_img, cmap = 'gray') # 생성된 이미지를 회색조로 표시
plt.show() # 그래프 출력
