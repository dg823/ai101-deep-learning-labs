import argparse  # 명령행 인자 처리를 위한 모듈 임포트

import torch  # PyTorch 라이브러리 임포트
import torch.nn as nn  # 신경망 관련 모듈 임포트
import torch.optim as optim  # 최적화 관련 모듈 임포트

from model import ImageClassifier  # 사용자 정의 모델 임포트
from trainer import Trainer  # 사용자 정의 학습 관리 클래스 임포트

from utils import load_mnist  # 데이터 로드 함수 임포트
from utils import split_data  # 데이터 분리 함수 임포트
from utils import get_hidden_sizes  # 은닉층 크기 계산 함수 임포트


def define_argparser():  # 실행 인자를 정의하는 함수
    p = argparse.ArgumentParser()  # 인자 파서 생성

    p.add_argument('--model_fn', required=True)  # 저장할 모델 파일 경로 (필수)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)  # 사용할 GPU ID

    p.add_argument('--train_ratio', type=float, default=.8)  # 학습 데이터 비율

    p.add_argument('--batch_size', type=int, default=256)  # 배치 크기
    p.add_argument('--n_epochs', type=int, default=20)  # 학습 에포크 수

    p.add_argument('--n_layers', type=int, default=5)  # 모델의 레이어 수
    p.add_argument('--use_dropout', action='store_true')  # 드롭아웃 사용 여부 (플래그)
    p.add_argument('--dropout_p', type=float, default=.3)  # 드롭아웃 확률

    p.add_argument('--verbose', type=int, default=1)  # 로그 출력 상세도

    config = p.parse_args()  # 입력된 인자들을 파싱하여 저장

    return config  # 설정 객체 반환


def main(config):  # 메인 학습 실행 함수
    # 설정된 GPU ID에 따라 장치(CPU/GPU) 설정
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    x, y = load_mnist(is_train=True, flatten=True)  # MNIST 학습 데이터 로드
    x, y = split_data(x.to(device), y.to(device), train_ratio=config.train_ratio)  # 학습/검증 데이터 분리 및 장치 이동

    print("Train:", x[0].shape, y[0].shape)  # 학습 데이터 크기 출력
    print("Valid:", x[1].shape, y[1].shape)  # 검증 데이터 크기 출력

    input_size = int(x[0].shape[-1])  # 입력 데이터의 피처 수 (28*28=784)
    output_size = int(max(y[0])) + 1  # 출력 클래스 수 (0~9까지 총 10개)

    # 모델 객체 생성
    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size,
                                      output_size,
                                      config.n_layers),
        use_batch_norm=not config.use_dropout,  # 드롭아웃 미사용 시 배치 정규화 사용
        dropout_p=config.dropout_p,
    ).to(device)  # 설정된 장치로 모델 이동
    
    optimizer = optim.Adam(model.parameters())  # Adam 최적화기 설정
    crit = nn.NLLLoss()  # 로그 소프트맥스 결과에 대한 음의 로그 가능도 손실 함수

    if config.verbose >= 1:  # 로그 상세도가 1 이상이면 모델 구조 및 설정 출력
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)  # Trainer 객체 생성

    # 모델 학습 시작
    trainer.train(
        train_data=(x[0], y[0]),
        valid_data=(x[1], y[1]),
        config=config
    )

    # 가장 성능이 좋았던 모델의 상태와 설정을 파일로 저장
    torch.save({
        'model': trainer.model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, config.model_fn)


if __name__ == '__main__':  # 스크립트가 직접 실행될 때
    config = define_argparser()  # 인자 설정 로드
    main(config)  # 메인 함수 호출
