#!/usr/bin/env python
# coding: utf-8

# # 프로젝트 1. 영화 리뷰 감정 분석
# **RNN 을 이용해 IMDB 데이터를 가지고 텍스트 감정분석을 해 봅시다.**
# 이번 책에서 처음으로 접하는 텍스트 형태의 데이터셋인 IMDB 데이터셋은 50,000건의 영화 리뷰로 이루어져 있습니다.
# 각 리뷰는 다수의 영어 문장들로 이루어져 있으며, 평점이 7점 이상의 긍정적인 영화 리뷰는 2로, 평점이 4점 이하인 부정적인 영화 리뷰는 1로 레이블링 되어 있습니다. 영화 리뷰 텍스트를 RNN 에 입력시켜 영화평의 전체 내용을 압축하고, 이렇게 압축된 리뷰가 긍정적인지 부정적인지 판단해주는 간단한 분류 모델을 만드는 것이 이번 프로젝트의 목표입니다.

import os  # 운영체제 관련 유틸리티 임포트
import torch  # 파이토치 임포트
import torch.nn as nn  # 신경망 모듈 임포트
import torch.nn.functional as F  # 신경망 함수 모듈 임포트
from torchtext.datasets import IMDB  # IMDB 데이터셋 임포트
from torchtext.vocab import build_vocab_from_iterator  # 반복자로부터 어휘집 구축 함수 임포트
from torchtext.data.utils import get_tokenizer  # 토크나이저 획득 함수 임포트
from tqdm import tqdm  # 진행률 표시줄 임포트

# 하이퍼파라미터
BATCH_SIZE = 64  # 한 번에 처리할 데이터 묶음 크기
lr = 0.001  # 학습률
EPOCHS = 10  # 총 학습 횟수
USE_CUDA = torch.cuda.is_available()  # GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")  # 학습 기기 설정
print("다음 기기로 학습합니다:", DEVICE)  # 설정된 기기 출력

# 데이터 로딩하기
print("데이터 로딩중...")  # 로딩 시작 메시지
tokenizer = get_tokenizer('basic_english')  # 기본 영어 토크나이저 설정

def yield_tokens(data_iter):  # 토큰 생성을 위한 제너레이터 함수
    for _, text in data_iter:  # 데이터 반복자로부터 텍스트 추출
        yield tokenizer(text)  # 토큰화된 결과 반환

train_iter = IMDB(split='train')  # 학습 데이터 반복자 생성
test_iter = IMDB(split='test')  # 테스트 데이터 반복자 생성

# 단어 사전 구축 (특수 토큰 포함)
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
vocab.set_default_index(vocab["<unk>"])  # 모르는 단어는 <unk>로 처리

def text_pipeline(x):  # 텍스트 처리 파이프라인
    return vocab(tokenizer(x))  # 토큰화 후 정수 인덱스로 변환

def label_pipeline(x):  # 레이블 처리 파이프라인
    return int(x) - 1  # 레이블 값을 0과 1로 조정

def process_batch(batch, batch_size):  # 배치를 텐서로 변환하는 함수
    label_list, text_list, lengths = [], [], []  # 데이터 저장 리스트
    for i, (_label, _text) in enumerate(batch):  # 배치 데이터 반복
        if i >= batch_size:  # 설정된 배치 크기까지만 처리
            break
        label_list.append(label_pipeline(_label))  # 레이블 변환 및 추가
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)  # 텍스트 변환
        if len(processed_text) == 0:  # 빈 텍스트 처리
            processed_text = torch.tensor([vocab["<pad>"]], dtype=torch.int64)
        text_list.append(processed_text)  # 텍스트 추가
        lengths.append(len(processed_text))  # 길이 추가
    
    if not text_list:  # 빈 배치 처리
        return None, None, None
    
    max_length = max(lengths)  # 현재 배치 내 최대 길이 확인
    padded_text_list = []  # 패딩 처리된 리스트
    for text in text_list:  # 각 텍스트에 대해
        if len(text) < max_length:  # 최대 길이보다 짧으면
            padding = torch.full((max_length - len(text),), vocab["<pad>"], dtype=torch.int64)  # 패딩 생성
            padded_text = torch.cat([text, padding])  # 패딩 결합
        else:
            padded_text = text  # 그대로 유지
        padded_text_list.append(padded_text)  # 리스트 추가
    
    text_tensor = torch.stack(padded_text_list)  # 텐서 결합
    label_tensor = torch.tensor(label_list, dtype=torch.int64)  # 레이블 텐서 생성
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)  # 길이 텐서 생성
    
    return label_tensor.to(DEVICE), text_tensor.to(DEVICE), lengths_tensor.to(DEVICE)  # 기기로 전송 후 반환

def train(model, optimizer, train_iter, batch_size):  # 학습 함수
    model.train()  # 모델을 학습 모드로 설정
    total_loss = 0  # 총 손실 초기화
    batch = []  # 현재 배치 데이터
    progress_bar = tqdm(desc='Training')  # 진행 표시줄 생성
    
    for i, data in enumerate(train_iter):  # 학습 데이터 반복
        batch.append(data)  # 데이터를 배치에 추가
        if len(batch) == batch_size:  # 배치 크기가 되면
            label, text, lengths = process_batch(batch, batch_size)  # 배치 처리
            if label is not None:
                optimizer.zero_grad()  # 기울기 초기화
                logit = model(text, lengths)  # 모델 예측
                loss = F.cross_entropy(logit, label)  # 손실 계산
                loss.backward()  # 역전파
                optimizer.step()  # 가중치 업데이트
                total_loss += loss.item()  # 손실 누적
                progress_bar.update(1)  # 진행률 갱신
                progress_bar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})  # 손실 출력
            batch = []  # 배치 초기화
    
    # 마지막 남은 배치 처리
    if batch:
        label, text, lengths = process_batch(batch, batch_size)
        if label is not None:
            optimizer.zero_grad()
            logit = model(text, lengths)
            loss = F.cross_entropy(logit, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{total_loss/(i+1):.4f}'})

def evaluate(model, test_iter, batch_size):  # 평가 함수
    model.eval()  # 모델을 평가 모드로 설정
    corrects, total_loss = 0, 0  # 정확도 및 손실 초기화
    total_samples = 0  # 총 샘플 수
    batch = []  # 배치 데이터
    progress_bar = tqdm(desc='Evaluating')  # 진행 표시줄 생성
    
    with torch.no_grad():  # 기울기 계산 비활성화
        for i, data in enumerate(test_iter):  # 테스트 데이터 반복
            batch.append(data)
            if len(batch) == batch_size:
                label, text, lengths = process_batch(batch, batch_size)
                if label is not None:
                    logit = model(text, lengths)  # 예측
                    loss = F.cross_entropy(logit, label, reduction='sum')  # 손실 합계 계산
                    total_loss += loss.item()  # 손실 누적
                    corrects += (logit.max(1)[1] == label).sum().item()  # 맞춘 개수 누적
                    total_samples += len(label)  # 샘플 수 누적
                    progress_bar.update(1)
                    progress_bar.set_postfix({'accuracy': f'{100.0 * corrects / total_samples:.2f}%'})
                batch = []
        
        # 마지막 남은 배치 처리
        if batch:
            label, text, lengths = process_batch(batch, batch_size)
            if label is not None:
                logit = model(text, lengths)
                loss = F.cross_entropy(logit, label, reduction='sum')
                total_loss += loss.item()
                corrects += (logit.max(1)[1] == label).sum().item()
                total_samples += len(label)
                progress_bar.update(1)
                progress_bar.set_postfix({'accuracy': f'{100.0 * corrects / total_samples:.2f}%'})
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0  # 평균 손실
    avg_accuracy = 100.0 * corrects / total_samples if total_samples > 0 else 0  # 평균 정확도
    return avg_loss, avg_accuracy  # 결과 반환

vocab_size = len(vocab)  # 사전 크기
n_classes = 2  # 클래스 개수 (긍정/부정)

print("[단어수]: %d [클래스] %d" % (vocab_size, n_classes))  # 정보 출력


class BasicGRU(nn.Module):  # 기본 GRU 모델 클래스 정의
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicGRU, self).__init__()
        print("Building Basic GRU model...")
        self.n_layers = n_layers  # 계층 수
        self.embed = nn.Embedding(n_vocab, embed_dim)  # 임베딩 계층
        self.hidden_dim = hidden_dim  # 은닉 차원
        self.dropout = nn.Dropout(dropout_p)  # 드롭아웃
        self.gru = nn.GRU(embed_dim, self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True)  # GRU 계층
        self.out = nn.Linear(self.hidden_dim, n_classes)  # 출력 계층

    def forward(self, text, lengths):  # 순전파
        embedded = self.embed(text)  # 임베딩 수행
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)  # 패딩 압축 (lengths는 CPU에 있어야 함)
        packed_output, hidden = self.gru(packed)  # GRU 실행
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)  # 패딩 복원
        hidden = self.dropout(hidden[-1])  # 마지막 은닉 상태에 드롭아웃 적용
        return self.out(hidden)  # 최종 예측값 반환


train_iter = IMDB(split='train')  # 재정의 (반복자 소모 방지)
test_iter = IMDB(split='test')

model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)  # 모델 인스턴스 생성 및 이동
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 최적화 도구 설정

best_val_loss = None  # 최고 검증 손실 초기화
for e in range(1, EPOCHS+1):  # 학습 루프
    print(f'\nEpoch {e}/{EPOCHS}')
    train(model, optimizer, train_iter, BATCH_SIZE)  # 학습 수행
    val_loss, val_accuracy = evaluate(model, test_iter, BATCH_SIZE)  # 검증 수행
    print(f"[Epoch {e}] Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%")
    
    if not best_val_loss or val_loss < best_val_loss:  # 모델 성능 개선 시 저장
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

# 최적 모델 로드 후 최종 평가
model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter, BATCH_SIZE)
print(f'\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')
