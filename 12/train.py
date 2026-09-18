import argparse # 명령행 인자 파싱을 위한 라이브러리 임포트

import torch # 파이토치 라이브러리 임포트
import torch.nn as nn # 신경망 모듈 임포트
import torch.optim as optim # 최적화 알고리즘 모듈 임포트

from mnist_classifier.trainer import Trainer # 사용자 정의 Trainer 클래스 임포트

from mnist_classifier.utils import load_mnist # MNIST 데이터 로드 함수 임포트
from mnist_classifier.utils import split_data # 데이터 분할 함수 임포트
from mnist_classifier.utils import get_model # 모델 생성 함수 임포트


def define_argparser(): # 명령행 인자 설정을 위한 함수 정의
    p = argparse.ArgumentParser() # 인자 파서 객체 생성

    p.add_argument("--model_fn", required=True) # 모델 가중치를 저장할 파일 이름 (필수 인자)
    p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1) # 사용할 GPU ID (없으면 -1)

    p.add_argument("--train_ratio", type=float, default=.8) # 학습 데이터 비율 (기본값 0.8)

    p.add_argument("--batch_size", type=int, default=256) # 배치 크기 (기본값 256)
    p.add_argument("--n_epochs", type=int, default=10) # 학습 에폭 수 (기본값 10)

    p.add_argument("--model", default="fc", choices=["fc", "cnn"]) # 모델 타입 선택 (fc: Fully Connected, cnn: Convolutional)

    p.add_argument("--n_layers", type=int, default=5) # 모델의 레이어 수 (기본값 5)
    p.add_argument("--use_dropout", action="store_true") # 드롭아웃 사용 여부 (인자가 있으면 True)
    p.add_argument("--dropout_p", type=float, default=.3) # 드롭아웃 확률 (기본값 0.3)

    p.add_argument("--verbose", type=int, default=1) # 로그 출력 상세도 (기본값 1)

    config = p.parse_args() # 입력된 인자들을 파싱하여 config 객체에 저장

    return config # 설정 객체 반환


def main(config): # 메인 학습 프로세스 함수 정의
    # Set device based on user defined configuration. # 설정에 따라 장치(CPU/GPU) 할당
    device = torch.device("cpu") if config.gpu_id < 0 else torch.device("cuda:%d" % config.gpu_id) # GPU ID가 -1이면 CPU, 아니면 해당 GPU 사용

    x, y = load_mnist(is_train=True, flatten=(config.model == "fc")) # MNIST 데이터 로드 (fc 모델이면 평탄화)
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio) # 데이터를 학습용과 검증용으로 분할하여 장치로 전송

    print("Train:", x[0].shape, y[0].shape) # 학습 데이터 크기 출력
    print("Valid:", x[1].shape, y[1].shape) # 검증 데이터 크기 출력

    input_size = int(x[0].shape[-1]) # 입력 차원 크기 계산
    output_size = int(max(y[0])) + 1 # 출력 클래스 개수 계산 (레이블 최대값 + 1)

    model = get_model( # 모델 인스턴스 생성
        input_size, # 입력 크기
        output_size, # 출력 크기
        config, # 설정 객체
        device, # 장치 정보
    ).to(device) # 모델을 해당 장치로 전송
    optimizer = optim.Adam(model.parameters()) # Adam 최적화 알고리즘 설정
    crit = nn.NLLLoss() # Negative Log Likelihood 손실 함수 설정

    if config.verbose >= 1: # 상세 출력 모드인 경우
        print(model) # 모델 구조 출력
        print(optimizer) # 최적화 설정 출력
        print(crit) # 손실 함수 출력

    trainer = Trainer(model, optimizer, crit) # 학습을 수행할 Trainer 객체 생성

    trainer.train( # 학습 시작
        train_data=(x[0], y[0]), # 학습 데이터
        valid_data=(x[1], y[1]), # 검증 데이터
        config=config # 설정 정보 전달
    )

    # Save best model weights. # 최적의 모델 가중치 저장
    torch.save({ # 가중치와 설정을 딕셔너리 형태로 저장
        "model": trainer.model.state_dict(), # 모델 상태 사전
        "opt": optimizer.state_dict(), # 최적화 도구 상태 사전
        "config": config, # 사용된 설정 정보
    }, config.model_fn) # 지정된 파일 경로에 저장


if __name__ == "__main__": # 스크립트가 직접 실행될 경우
    config = define_argparser() # 인자 파싱 수행
    main(config) # 메인 함수 실행
