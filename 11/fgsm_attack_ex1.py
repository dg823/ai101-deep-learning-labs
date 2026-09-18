#!/usr/bin/env python
# coding: utf-8

# # 8.1 FGSM 공격
# 정상 이미지에 노이즈를 더해 머신러닝 모델을 헷갈리게 하는 이미지가
# 바로 적대적 예제(Adversarial Example) 입니다.
# 이 프로젝트에선 Fast Gradient Sign Method, 즉 줄여서 FGSM이라는 방식으로
# 적대적 예제를 생성해 미리 학습이 완료된 딥러닝 모델을 공격해보도록 하겠습니다.

import torch  # PyTorch 라이브러리 임포트
import torch.nn.functional as F  # PyTorch의 기능적 함수 모듈 임포트
import torchvision.models as models  # 사전 학습된 모델들을 제공하는 모듈 임포트
import torchvision.transforms as transforms  # 이미지 변환 도구 임포트

from PIL import Image  # 이미지 처리를 위한 PIL 라이브러리 임포트
import json  # JSON 파일 처리를 위한 모듈 임포트


import matplotlib.pyplot as plt  # 시각화를 위한 matplotlib 임포트


# ## 학습된 모델 불러오기

model = models.resnet101(pretrained=True)  # ImageNet으로 사전 학습된 ResNet101 모델 로드
model.eval()  # 모델을 평가 모드로 설정
print(model)  # 모델의 레이어 구조 출력


# ## 데이터셋 불러오기

# ImageNet 클래스 인덱스와 이름이 매핑된 JSON 파일 로드
CLASSES = json.load(open('./imagenet_samples/imagenet_classes.json'))
# 인덱스 순서대로 클래스 이름을 리스트로 변환
idx2class = [CLASSES[str(i)] for i in range(1000)]


# ## 이미지 불러오기

# 공격할 대상 이미지 파일 열기
img = Image.open('imagenet_samples/corgie.jpg')
#img = Image.open('imagenet_samples/stoplight.jpg')


# 이미지를 모델 입력 크기에 맞게 전처리하는 변환 정의
img_transforms = transforms.Compose([
    transforms.Resize((224, 224), Image.BICUBIC),  # 224x224 크기로 조정
    transforms.ToTensor(),  # 텐서로 변환
])

img_tensor = img_transforms(img)  # 이미지 변환 적용
img_tensor = img_tensor.unsqueeze(0)  # 배치 차원 추가 (1, 3, 224, 224)

print("이미지 텐서 모양:", img_tensor.size())  # 변환된 이미지의 크기 출력


# 시각화를 위해 텐서를 넘파이 행렬로 변환
original_img_view = img_tensor.squeeze(0).detach()  # 배치 차원 제거 및 그래프 분리
# 차원 순서 변경 (C, H, W) -> (H, W, C) 후 넘파이 변환
original_img_view = original_img_view.transpose(0,2).transpose(0,1).numpy()

# 원본 이미지 시각화
plt.imshow(original_img_view)


# ## 공격 전 성능 확인하기
output = model(img_tensor)  # 모델에 이미지 통과
probabilities = F.softmax(output, dim=1)  # 확률 분포 계산
prediction = output.max(1, keepdim=False)[1]  # 가장 높은 확률의 인덱스(예측값) 추출

prediction_idx = prediction.item()  # 예측 인덱스를 정수로 변환
prediction_name = idx2class[prediction_idx]  # 클래스 이름 찾기
prediction_prob = probabilities[0][prediction_idx].item()  # 예측 확률 값 가져오기

# 예측 결과 출력
print("예측된 레이블 번호:", prediction_idx)
print("레이블 이름:", prediction_name)
print("예측 확률: {:.2%}".format(prediction_prob))


# ## FGSM 공격 함수 정의

def fgsm_attack(image, epsilon, gradient):
    # 입력 이미지의 기울기(Gradient) 값의 부호(Sign)를 구함
    sign_gradient = gradient.sign()
    # 이미지의 각 픽셀 값을 기울기 부호 방향으로 epsilon만큼 조절하여 노이즈 추가
    perturbed_image = image + epsilon * sign_gradient
    # 픽셀 값이 [0, 1] 범위를 벗어나지 않도록 클리핑 수행
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image  # 생성된 적대적 예제 반환


# ## 적대적 예제 생성

# 이미지 텐서에 대해 기울기를 계산하도록 설정
img_tensor.requires_grad_(True)

# 이미지를 다시 모델에 통과시킴
output = model(img_tensor)

# 실제 레이블과의 오차 계산 (263번은 웰시코기 클래스 번호)
loss = F.nll_loss(output, torch.tensor([263])) 

# 모델의 기울기 초기화 후 오차에 대한 역전파 수행
model.zero_grad()
loss.backward()

# 입력 이미지 텐서에 쌓인 기울기값 추출
gradient = img_tensor.grad.data

# FGSM 공격 함수를 호출하여 적대적 예제(왜곡된 이미지) 생성
epsilon = 0.03  # 노이즈 강도 설정
perturbed_data = fgsm_attack(img_tensor, epsilon, gradient)

# 생성된 적대적 예제를 모델에 다시 통과시킴
output = model(perturbed_data)


# ## 적대적 예제 성능 확인

perturbed_probabilities = F.softmax(output, dim=1)  # 공격 후 확률 분포 계산
perturbed_prediction = output.max(1, keepdim=True)[1]  # 공격 후 예측 인덱스 추출

perturbed_prediction_idx = perturbed_prediction.item()  # 공격 후 예측 인덱스 정수화
perturbed_prediction_name = idx2class[perturbed_prediction_idx]  # 공격 후 클래스 이름
perturbed_prediction_prob = perturbed_probabilities[0][perturbed_prediction_idx].item()  # 공격 후 예측 확률

# 공격 후 예측 결과 출력
print("예측된 레이블 번호:", perturbed_prediction_idx)
print("레이블 이름:", perturbed_prediction_name)
print("예측 확률: {:.2%}".format(perturbed_prediction_prob))


# 시각화를 위해 적대적 예제를 넘파이 행렬로 변환
perturbed_data_view = perturbed_data.squeeze(0).detach()  # 배치 차원 제거 및 분리
perturbed_data_view = perturbed_data_view.transpose(0,2).transpose(0,1).numpy()  # 차원 변경 및 넘파이화

# 적대적 예제 시각화
plt.imshow(perturbed_data_view)


# ## 원본과 적대적 예제 비교

f, a = plt.subplots(1, 2, figsize=(10, 10))  # 1행 2열로 시각화 창 생성

# 왼쪽: 원본 이미지 및 예측 결과
a[0].set_title(prediction_name)
a[0].imshow(original_img_view)

# 오른쪽: 적대적 예제 및 공격 후 변화된 예측 결과
a[1].set_title(perturbed_prediction_name)
a[1].imshow(perturbed_data_view)

plt.show()  # 결과 출력
