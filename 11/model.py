import torch  # PyTorch 라이브러리 임포트
import torch.nn as nn  # PyTorch의 신경망 모듈 임포트


class Block(nn.Module):  # 신경망의 기본 단위가 되는 블록 클래스 정의
    
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):  # 초기화 메서드: 입력/출력 크기, 배치 정규화 사용 여부, 드롭아웃 확률 설정
        self.input_size = input_size  # 입력 크기 저장
        self.output_size = output_size  # 출력 크기 저장
        self.use_batch_norm = use_batch_norm  # 배치 정규화 사용 여부 저장
        self.dropout_p = dropout_p  # 드롭아웃 확률 저장
        
        super().__init__()  # 부모 클래스(nn.Module)의 초기화 메서드 호출
        
        def get_regularizer(use_batch_norm, size):  # 정규화 레이어를 선택하는 내부 함수 정의
            # 배치 정규화를 사용하면 BatchNorm1d를, 아니면 Dropout을 반환
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        
        self.block = nn.Sequential(  # 순차적인 레이어 층 구성
            nn.Linear(input_size, output_size),  # 선형 결합 레이어
            nn.LeakyReLU(),  # 활성화 함수로 LeakyReLU 사용
            get_regularizer(use_batch_norm, output_size),  # 선택된 정규화 레이어 추가
        )
        
    def forward(self, x):  # 순전파 메서드 정의
        # |x| = (batch_size, input_size)  # 입력 x의 크기 설명
        y = self.block(x)  # 블록 내부의 레이어들을 통과시킴
        # |y| = (batch_size, output_size)  # 출력 y의 크기 설명
        
        return y  # 최종 출력 반환

    
class ImageClassifier(nn.Module):  # 전체 이미지 분류 모델 클래스 정의

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500, 400, 300, 200, 100],
                 use_batch_norm=True,
                 dropout_p=.3):  # 초기화 메서드: 입력/출력 크기, 은닉층 리스트, 정규화 옵션 등 설정
        
        super().__init__()  # 부모 클래스의 초기화 메서드 호출

        assert len(hidden_sizes) > 0, "You need to specify hidden layers"  # 은닉층이 최소 하나 이상인지 확인

        last_hidden_size = input_size  # 이전 층의 크기를 입력 크기로 시작
        blocks = []  # 블록들을 담을 리스트 초기화
        for hidden_size in hidden_sizes:  # 설정된 은닉층 크기들을 순회하며
            blocks += [Block(  # 블록 객체 생성 및 리스트 추가
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size  # 다음 층의 입력 크기를 현재 층의 출력 크기로 업데이트
        
        self.layers = nn.Sequential(  # 전체 레이어 구성
            *blocks,  # 생성된 블록들을 언패킹하여 추가
            nn.Linear(last_hidden_size, output_size),  # 마지막 출력층 추가
            nn.LogSoftmax(dim=-1),  # 로그 소프트맥스를 통해 확률 분포(로그값) 반환
        )
        
    def forward(self, x):  # 순전파 메서드 정의
        # |x| = (batch_size, input_size)  # 입력 x의 크기        
        y = self.layers(x)  # 전체 레이어 층을 통과시킴
        # |y| = (batch_size, output_size)  # 출력 y의 크기
        
        return y  # 최종 결과 반환
