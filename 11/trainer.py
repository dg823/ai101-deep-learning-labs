from copy import deepcopy  # 모델 가중치의 깊은 복사를 위한 모듈 임포트
import numpy as np  # 수치 계산을 위한 numpy 임포트

import torch  # PyTorch 라이브러리 임포트

class Trainer():  # 학습 과정을 관리하는 Trainer 클래스 정의

    def __init__(self, model, optimizer, crit):  # 초기화 메서드: 모델, 최적화기, 손실 함수 설정
        self.model = model  # 학습할 모델
        self.optimizer = optimizer  # 가중치 업데이트를 위한 최적화기
        self.crit = crit  # 손실(Error) 계산을 위한 손실 함수

        super().__init__()  # 부모 클래스 초기화

    def _batchify(self, x, y, batch_size, random_split=True):  # 데이터를 배치 단위로 나누는 메서드
        if random_split:  # 무작위 분할이 설정된 경우
            indices = torch.randperm(x.size(0), device=x.device)  # 무작위 인덱스 생성
            x = torch.index_select(x, dim=0, index=indices)  # 데이터 섞기
            y = torch.index_select(y, dim=0, index=indices)  # 레이블 섞기

        x = x.split(batch_size, dim=0)  # 데이터를 배치 크기만큼 나눔
        y = y.split(batch_size, dim=0)  # 레이블을 배치 크기만큼 나눔

        return x, y  # 배치 리스트 반환

    def _train(self, x, y, config):  # 단일 에포크에 대한 학습 수행 메서드
        # 모델을 학습 모드로 전환 (BatchNorm, Dropout 활성화)
        self.model.train()
        
        x, y = self._batchify(x, y, config.batch_size)  # 데이터를 배치로 나눔
        total_loss = 0  # 총 손실값 초기화

        for i, (x_i, y_i) in enumerate(zip(x, y)):  # 각 배치에 대해 반복
            y_hat_i = self.model(x_i)  # 모델을 통해 예측 수행
            loss_i = self.crit(y_hat_i, y_i.squeeze())  # 실제값과 비교하여 손실 계산

            # 모델의 기울기(Gradient) 초기화
            self.optimizer.zero_grad()
            loss_i.backward()  # 역전파를 통해 기울기 계산

            self.optimizer.step()  # 가중치 업데이트

            if config.verbose >= 2:  # 상세 로그 출력 설정 시
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # 메모리 누수 방지를 위해 텐서에서 값만 추출하여 누적
            total_loss += float(loss_i)

        return total_loss / len(x)  # 평균 학습 손실 반환

    def _validate(self, x, y, config):  # 검증 데이터에 대한 성능 평가 메서드
        # 모델을 평가 모드로 전환 (BatchNorm, Dropout 비활성화)
        self.model.eval()

        # 기울기 계산을 비활성화하여 메모리와 연산 효율 높임
        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)  # 검증 시에는 섞지 않고 배치 분할
            total_loss = 0  # 총 검증 손실 초기화

            for i, (x_i, y_i) in enumerate(zip(x, y)):  # 각 배치에 대해 반복
                y_hat_i = self.model(x_i)  # 예측 수행
                loss_i = self.crit(y_hat_i, y_i.squeeze())  # 손실 계산

                if config.verbose >= 2:  # 상세 로그 출력 설정 시
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)  # 손실 누적

            return total_loss / len(x)  # 평균 검증 손실 반환

    def train(self, train_data, valid_data, config):  # 전체 학습 과정을 수행하는 메서드
        lowest_loss = np.inf  # 최소 손실값을 무한대로 초기화
        best_model = None  # 최고 성능 모델의 가중치를 저장할 변수

        for epoch_index in range(config.n_epochs):  # 설정된 에포크 수만큼 반복
            train_loss = self._train(train_data[0], train_data[1], config)  # 학습 수행
            valid_loss = self._validate(valid_data[0], valid_data[1], config)  # 검증 수행

            # 검증 손실이 이전 최저값보다 작거나 같으면 가중치 스냅샷 저장
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())  # 현재 모델 가중치 복사

            # 에포크별 로그 출력
            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # 학습 종료 후 가장 성능이 좋았던 시점의 가중치로 복원
        self.model.load_state_dict(best_model)
