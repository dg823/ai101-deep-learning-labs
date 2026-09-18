import torch  # PyTorch 라이브러리 임포트


def load_mnist(is_train=True, flatten=True):  # MNIST 데이터셋을 로드하는 함수 정의
    from torchvision import datasets, transforms  # torchvision에서 데이터셋과 변환 도구 임포트

    dataset = datasets.MNIST(  # MNIST 데이터셋 객체 생성
        '../data', train=is_train, download=True,  # 데이터 경로, 학습용 여부, 다운로드 설정
        transform=transforms.Compose([  # 데이터 변환 파이프라인 설정
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (0~1 사이로 정규화됨)
        ]),
    )

    x = dataset.data.float() / 255.  # 이미지 데이터를 float 타입으로 변환하고 255로 나누어 0~1 사이로 정규화
    y = dataset.targets  # 레이블 데이터 가져오기

    if flatten:  # flatten 옵션이 True인 경우
        x = x.view(x.size(0), -1)  # 28x28 이미지를 784 차원의 1차원 벡터로 변환

    return x, y  # 입력 데이터 x와 레이블 y 반환


def split_data(x, y, train_ratio=.8):  # 데이터를 학습용과 검증용으로 분리하는 함수 정의
    train_cnt = int(x.size(0) * train_ratio)  # 학습용 데이터 개수 계산
    valid_cnt = x.size(0) - train_cnt  # 검증용 데이터 개수 계산

    # 데이터를 섞기 위해 인덱스를 무작위로 생성
    indices = torch.randperm(x.size(0))  # 0부터 데이터 개수-1까지의 숫자를 무작위 순서로 생성
    x = torch.index_select(  # 인덱스에 따라 데이터를 재배치
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)  # 학습용과 검증용 크기로 데이터 분할
    y = torch.index_select(  # 인덱스에 따라 레이블을 재배치
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)  # 학습용과 검증용 크기로 레이블 분할

    return x, y  # 분할된 데이터와 레이블 세트 반환


def get_hidden_sizes(input_size, output_size, n_layers):  # 은닉층 크기 리스트를 계산하는 함수 정의
    step_size = int((input_size - output_size) / n_layers)  # 층별로 줄어들 크기 간격 계산

    hidden_sizes = []  # 은닉층 크기를 담을 리스트 초기화
    current_size = input_size  # 현재 크기를 입력 크기로 설정
    for i in range(n_layers - 1):  # 레이어 수만큼 반복 (마지막 출력층 제외)
        hidden_sizes += [current_size - step_size]  # 일정한 간격으로 크기를 줄여 리스트에 추가
        current_size = hidden_sizes[-1]  # 현재 크기를 방금 추가한 크기로 업데이트

    return hidden_sizes  # 은닉층 크기 리스트 반환
