#!/usr/bin/env python
# coding: utf-8

# # 오토인코더로 이미지의 특징을 추출하기

import torch # 파이토치 임포트
import torchvision # 토치비전 임포트
import torch.nn.functional as F # 신경망 함수 임포트
from torch import nn, optim # 모델 및 최적화 임포트
from torchvision import transforms, datasets # 데이터 변환 및 데이터셋 임포트

import matplotlib.pyplot as plt # 시각화 라이브러리
from mpl_toolkits.mplot3d import Axes3D # 3D 시각화 도구
from matplotlib import cm # 컬러맵 설정
import numpy as np # 수치 계산 라이브러리
import os # 운영체제 제어


# 하이퍼파라미터
EPOCH = 10 # 학습 반복 횟수
BATCH_SIZE = 64 # 한 번에 학습할 데이터 수
USE_CUDA = torch.cuda.is_available() # GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if USE_CUDA else "cpu") # 학습 장치 설정
print("Using Device:", DEVICE) # 사용 중인 장치 출력


# Fashion MNIST 데이터셋
trainset = datasets.FashionMNIST( # 패션 데이터셋 불러오기
    root      = './.data/', # 데이터 저장 경로
    train     = True, # 학습용 데이터 선택
    download  = True, # 데이터 없으면 다운로드
    transform = transforms.ToTensor() # 텐서로 변환
)
train_loader = torch.utils.data.DataLoader( # 데이터 로더 설정
    dataset     = trainset, # 데이터셋 지정
    batch_size  = BATCH_SIZE, # 배치 크기 설정
    shuffle     = True, # 데이터 섞기
    num_workers = 0  # 병렬 처리 수
)


class Autoencoder(nn.Module): # 오토인코더 클래스 정의
    def __init__(self): # 초기화 함수
        super(Autoencoder, self).__init__() # 부모 클래스 초기화

        self.encoder = nn.Sequential( # 인코더 정의
            nn.Linear(28*28, 128), # 입력층 (784 -> 128)
            nn.ReLU(), # 활성화 함수
            nn.Linear(128, 64), # 은닉층 (128 -> 64)
            nn.ReLU(), # 활성화 함수
            nn.Linear(64, 12), # 은닉층 (64 -> 12)
            nn.ReLU(), # 활성화 함수
            nn.Linear(12, 3),   # 잠재 변수층 (12 -> 3)
        )
        self.decoder = nn.Sequential( # 디코더 정의
            nn.Linear(3, 12), # 입력층 (3 -> 12)
            nn.ReLU(), # 활성화 함수
            nn.Linear(12, 64), # 은닉층 (12 -> 64)
            nn.ReLU(), # 활성화 함수
            nn.Linear(64, 128), # 은닉층 (64 -> 128)
            nn.ReLU(), # 활성화 함수
            nn.Linear(128, 28*28), # 출력층 (128 -> 784)
            nn.Sigmoid(),       # 0~1 사이 값 출력
        )

    def forward(self, x): # 순전파 함수
        encoded = self.encoder(x) # 인코딩 과정
        decoded = self.decoder(encoded) # 디코딩 과정
        return encoded, decoded # 압축 결과와 복원 결과 반환


def train(autoencoder, train_loader): # 학습 함수 정의
    autoencoder.train() # 학습 모드 설정
    for step, (x, label) in enumerate(train_loader): # 배치 반복
        x = x.view(-1, 28*28).to(DEVICE) # 입력을 평탄화하여 장치로 이동
        y = x.view(-1, 28*28).to(DEVICE) # 정답(입력과 동일) 설정
        label = label.to(DEVICE) # 레이블 이동

        encoded, decoded = autoencoder(x) # 모델 실행

        loss = criterion(decoded, y) # 손실 계산 (오차 측정)
        optimizer.zero_grad() # 기울기 초기화
        loss.backward() # 역전파 (기울기 계산)
        optimizer.step() # 가중치 업데이트


if __name__ == '__main__': # 메인 실행부
    autoencoder = Autoencoder().to(DEVICE) # 모델 생성 및 장치 이동
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005) # 최적화 알고리즘 설정
    criterion = nn.MSELoss() # 손실 함수 (평균 제곱 오차) 설정

    # 원본 이미지를 시각화 하기 (첫번째 열)
    view_data = trainset.data[:5].view(-1, 28*28) # 데이터 5개 선택 및 평탄화
    view_data = view_data.type(torch.FloatTensor)/255. # 정규화 (0~1)

    # 결과를 저장할 디렉토리 생성
    if not os.path.exists('results'): # 폴더 없으면 생성
        os.makedirs('results')

    for epoch in range(1, EPOCH+1): # 에포크 반복
        train(autoencoder, train_loader) # 모델 학습 실행

        # 디코더에서 나온 이미지를 시각화 하기 (두번째 열)
        test_x = view_data.to(DEVICE) # 테스트 데이터 이동
        _, decoded_data = autoencoder(test_x) # 복원 수행

        # 원본과 디코딩 결과 비교해보기
        f, a = plt.subplots(2, 5, figsize=(5, 2)) # 서브플롯 생성
        print("[Epoch {}]".format(epoch)) # 현재 에포크 출력
        for i in range(5): # 원본 출력 반복
            img = np.reshape(view_data.data.numpy()[i],(28, 28)) # 이미지 형태로 변환
            a[0][i].imshow(img, cmap='gray') # 흑백 이미지 표시
            a[0][i].set_xticks(()); a[0][i].set_yticks(()) # 축 눈금 제거

        for i in range(5): # 복원 이미지 출력 반복
            img = np.reshape(decoded_data.to("cpu").data.numpy()[i], (28, 28)) # 이미지 형태로 변환
            a[1][i].imshow(img, cmap='gray') # 흑백 이미지 표시
            a[1][i].set_xticks(()); a[1][i].set_yticks(()) # 축 눈금 제거
        
        # 학습 과정 이미지 저장
        plt.savefig(f'results/epoch_{epoch:02d}.png', dpi=300, bbox_inches='tight') # 이미지 파일 저장
        plt.show() # 화면 출력

    # # 잠재변수 들여다보기

    # 잠재변수를 3D 플롯으로 시각화
    view_data = trainset.data[:200].view(-1, 28*28) # 데이터 200개 선택
    view_data = view_data.type(torch.FloatTensor)/255. # 정규화
    test_x = view_data.to(DEVICE) # 장치 이동
    encoded_data, _ = autoencoder(test_x) # 특징(잠재변수) 추출
    encoded_data = encoded_data.to("cpu") # 결과를 CPU로 이동

    CLASSES = { # 클래스 이름 정의
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    # 새로운 figure 생성
    plt.clf()  # 이전 plot 초기화
    fig = plt.figure(figsize=(12, 10)) # 그림 크기 설정
    ax = fig.add_subplot(111, projection='3d')  # 3D 축 생성

    X = encoded_data.data[:, 0].numpy() # 잠재 변수 첫 번째 차원
    Y = encoded_data.data[:, 1].numpy() # 잠재 변수 두 번째 차원
    Z = encoded_data.data[:, 2].numpy() # 잠재 변수 세 번째 차원

    labels = trainset.targets[:200].numpy() # 실제 레이블 정보

    # 산점도 그리기
    scatter = ax.scatter(X, Y, Z, c=labels, cmap='rainbow', s=100, alpha=0.8) # 3D 점 표시
    
    # 텍스트 레이블 추가 (일부만 표시)
    for i, (x, y, z, s) in enumerate(zip(X, Y, Z, labels)): # 레이블 표시 반복
        if i % 20 == 0:  # 20개마다 표시
            name = CLASSES[s] # 클래스 명칭 가져오기
            color = cm.rainbow(int(255*s/9)) # 색상 지정
            ax.text(x, y, z, name, color='black', fontsize=10, # 텍스트 추가
                    bbox=dict(facecolor=color, alpha=0.5, edgecolor='black', pad=2))

    # 축 레이블 추가
    ax.set_xlabel('Latent 1', fontsize=12) # X축 이름
    ax.set_ylabel('Latent 2', fontsize=12) # Y축 이름
    ax.set_zlabel('Latent 3', fontsize=12) # Z축 이름
    
    # 축 범위 설정
    margin = 0.1  # 여백
    x_range = X.max() - X.min() # X범위 계산
    y_range = Y.max() - Y.min() # Y범위 계산
    z_range = Z.max() - Z.min() # Z범위 계산
    
    ax.set_xlim(X.min() - margin * x_range, X.max() + margin * x_range) # X축 범위 설정
    ax.set_ylim(Y.min() - margin * y_range, Y.max() + margin * y_range) # Y축 범위 설정
    ax.set_zlim(Z.min() - margin * z_range, Z.max() + margin * z_range) # Z축 범위 설정
    
    # 컬러바 추가
    cbar = plt.colorbar(scatter) # 범례 색상바 추가
    cbar.set_label('Class', fontsize=12) # 컬러바 이름
    
    # 제목 추가
    plt.title('3D Visualization of Latent Space', fontsize=14, pad=20) # 전체 제목
    
    # 그리드 추가
    ax.grid(True) # 격자 표시
    
    # 시점 설정
    ax.view_init(elev=20, azim=45) # 3D 뷰 각도 조절
    
    # 레이아웃 조정
    plt.tight_layout() # 간격 최적화
    
    # 저장 및 표시
    plt.savefig('results/latent_space_3d.png', dpi=300, bbox_inches='tight') # 최종 시각화 저장
    plt.show() # 화면 출력

    print("\n모든 이미지가 'results' 디렉토리에 저장되었습니다.")
    print("- 학습 과정 이미지: epoch_01.png ~ epoch_10.png")
    print("- 3D 시각화 이미지: latent_space_3d.png")
