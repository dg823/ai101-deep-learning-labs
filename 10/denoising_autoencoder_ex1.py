#!/usr/bin/env python
# coding: utf-8

# # 오토인코더로 망가진 이미지 복원하기

import torch # 파이토치 임포트
import torchvision # 토치비전 임포트
import torch.nn.functional as F # 신경망 함수 임포트
from torch import nn, optim # 모델 및 최적화 임포트
from torchvision import transforms, datasets # 데이터 변환 및 데이터셋 임포트

import matplotlib.pyplot as plt # 시각화 라이브러리
import numpy as np # 수치 계산 라이브러리
import os # 운영체제 제어


# 하이퍼파라미터
EPOCH = 10 # 학습 반복 횟수
BATCH_SIZE = 64 # 배치 크기
USE_CUDA = torch.cuda.is_available() # GPU 확인
DEVICE = torch.device("cuda" if USE_CUDA else "cpu") # 장치 설정
print("다음 기기로 학습합니다:", DEVICE) # 사용 장치 출력


# 결과를 저장할 디렉토리 생성
if not os.path.exists('results'): # 폴더 확인
    os.makedirs('results') # 폴더 생성


# Fashion MNIST 학습 데이터셋
trainset = datasets.FashionMNIST( # 학습 데이터 불러오기
    root      = './.data/', 
    train     = True, # 학습용
    download  = True, 
    transform = transforms.ToTensor() # 텐서 변환
)

train_loader = torch.utils.data.DataLoader( # 데이터 로더
    dataset     = trainset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,
    num_workers = 0 
)


class Autoencoder(nn.Module): # 오토인코더 모델
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential( # 인코더 부분
            nn.Linear(28*28, 128), # 784 -> 128
            nn.ReLU(),
            nn.Linear(128, 64), # 128 -> 64
            nn.ReLU(),
            nn.Linear(64, 12), # 64 -> 12
            nn.ReLU(),
            nn.Linear(12, 3),   # 특징 압축 (3차원)
        )
        self.decoder = nn.Sequential( # 디코더 부분
            nn.Linear(3, 12), # 3 -> 12
            nn.ReLU(),
            nn.Linear(12, 64), # 12 -> 64
            nn.ReLU(),
            nn.Linear(64, 128), # 64 -> 128
            nn.ReLU(),
            nn.Linear(128, 28*28), # 128 -> 784
            nn.Sigmoid(),       # 0~1 출력
        )

    def forward(self, x): # 순전파
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = Autoencoder().to(DEVICE) # 모델 생성 및 이동
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005) # 최적화 설정
criterion = nn.MSELoss() # 손실 함수 설정


def add_noise(img): # 노이즈 추가 함수
    noise = torch.randn(img.size()) * 0.2 # 랜덤 노이즈 생성
    noisy_img = img + noise # 이미지에 노이즈 합치기
    return noisy_img


def train(autoencoder, train_loader): # 학습 함수
    autoencoder.train() # 학습 모드
    avg_loss = 0 # 평균 손실 초기화
    for step, (x, label) in enumerate(train_loader): # 배치 반복
        noisy_x = add_noise(x)  # 입력에 노이즈 추가
        noisy_x = noisy_x.view(-1, 28*28).to(DEVICE) # 평탄화 및 이동
        y = x.view(-1, 28*28).to(DEVICE) # 정답(깨끗한 이미지) 설정

        label = label.to(DEVICE) # 레이블 이동
        encoded, decoded = autoencoder(noisy_x) # 모델 실행

        loss = criterion(decoded, y) # 손실 계산
        optimizer.zero_grad() # 기울기 초기화
        loss.backward() # 역전파
        optimizer.step() # 업데이트
        
        avg_loss += loss.item() # 손실 누적
    return avg_loss / len(train_loader) # 평균 손실 반환


for epoch in range(1, EPOCH+1): # 학습 반복
    loss = train(autoencoder, train_loader) # 학습 실행
    print("[Epoch {}] loss:{}".format(epoch, loss)) # 상태 출력


# # 이미지 복원 시각화 하기

# 테스트 데이터셋 불러오기
testset = datasets.FashionMNIST(
    root      = './.data/', 
    train     = False, # 테스트용
    download  = True,
    transform = transforms.ToTensor()
)

# 샘플 데이터 준비
sample_data = testset.data[0].view(-1, 28*28) # 첫 번째 이미지
sample_data = sample_data.type(torch.FloatTensor)/255. # 정규화

# 복원 테스트
original_x = sample_data[0] # 원본
noisy_x = add_noise(original_x).to(DEVICE) # 노이즈 추가
_, recovered_x = autoencoder(noisy_x.view(-1, 28*28)) # 모델 복원


f, a = plt.subplots(1, 3, figsize=(15, 15)) # 서브플롯 생성

# 넘파이 변환 (시각화용)
original_img = np.reshape(original_x.to("cpu").data.numpy(), (28, 28)) # 원본 변환
noisy_img = np.reshape(noisy_x.to("cpu").data.numpy(), (28, 28)) # 노이즈 변환
recovered_img = np.reshape(recovered_x.to("cpu").data.numpy(), (28, 28)) # 복원 변환

# 원본 출력
a[0].set_title('Original')
a[0].imshow(original_img, cmap='gray')

# 노이즈 출력
a[1].set_title('Noisy')
a[1].imshow(noisy_img, cmap='gray')

# 복원 출력
a[2].set_title('Recovered')
a[2].imshow(recovered_img, cmap='gray')

# 결과 저장
plt.savefig('results/final_result.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n최종 결과 이미지가 'results/final_result.png'에 저장되었습니다.")
