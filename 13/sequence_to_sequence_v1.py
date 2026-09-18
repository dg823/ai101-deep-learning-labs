#!/usr/bin/env python
# coding: utf-8

# # Seq2Seq 기계 번역
# 이번 프로젝트에선 임의로 Seq2Seq 모델을 아주 간단화 시켰습니다.
# 한 언어로 된 문장을 다른 언어로 된 문장으로 번역하는 덩치가 큰 모델이 아닌
# 영어 알파벳 문자열("hello")을 스페인어 알파벳 문자열("hola")로 번역하는 Mini Seq2Seq 모델을 같이 구현해 보겠습니다.

import torch  # 파이토치 임포트
import torch.nn as nn  # 신경망 모듈 임포트
import torch.nn.functional as F  # 신경망 함수 모듈 임포트
import random  # 무작위 추출을 위한 random 임포트
import matplotlib.pyplot as plt  # 그래프 출력을 위한 matplotlib 임포트

# 학습 데이터 정의
training_pairs = [  # 영어-스페인어 학습 쌍 리스트
    ("I go to bed.", "Me voy a la cama."),
    ("I want to sleep.", "Quiero dormir."),
    ("I play a game.", "Juego un juego.")
]

vocab_size = 256  # 총 아스키 코드 개수 (단어 사전 크기)

class Seq2Seq(nn.Module):  # Seq2Seq 모델 클래스 정의
    def __init__(self, vocab_size, hidden_size):  # 초기화 메서드
        super(Seq2Seq, self).__init__()  # 부모 클래스 초기화
        self.n_layers = 2  # GRU 계층 개수 설정
        self.hidden_size = hidden_size  # 은닉 상태 크기 설정
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 임베딩 계층 정의
        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers, dropout=0.2)  # 인코더 GRU 정의
        self.decoder = nn.GRU(hidden_size, hidden_size, num_layers=self.n_layers, dropout=0.2)  # 디코더 GRU 정의
        self.attention = nn.Linear(hidden_size * 2, hidden_size)  # 어텐션 계산을 위한 선형 계층 (현재 미사용 가능성 있음)
        self.project = nn.Linear(hidden_size * 2, vocab_size)  # 출력 프로젝션 계층 정의
        
        # 가중치 초기화
        self._init_weights()  # 가중치 초기화 메서드 호출
    
    def _init_weights(self):  # 가중치 초기화 로직
        for name, param in self.named_parameters():  # 모델의 모든 파라미터 반복
            if 'weight' in name:  # 가중치인 경우
                nn.init.xavier_uniform_(param)  # Xavier 균등 초기화
            elif 'bias' in name:  # 편향인 경우
                nn.init.zeros_(param)  # 0으로 초기화

    def forward(self, inputs, targets):  # 순전파 메서드
        batch_size = 1  # 배치 크기 1 설정
        seq_len = inputs.size(0)  # 입력 시퀀스 길이
        
        # 인코더
        initial_state = self._init_state(batch_size)  # 인코더 초기 상태 생성
        embedding = self.embedding(inputs).unsqueeze(1)  # 입력 임베딩 및 차원 추가
        encoder_output, encoder_state = self.encoder(embedding, initial_state)  # 인코더 GRU 실행
        
        # 디코더
        decoder_state = encoder_state  # 디코더 초기 상태를 인코더의 마지막 상태로 설정
        decoder_input = torch.LongTensor([ord('V')]).to(inputs.device)  # 디코더 시작 기호(V) 설정
        
        outputs = []  # 결과 저장 리스트
        for i in range(targets.size(0)):  # 타겟 길이만큼 반복
            decoder_input = self.embedding(decoder_input).unsqueeze(1)  # 디코더 입력 임베딩
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)  # 디코더 GRU 실행
            
            # 어텐션 계산
            decoder_output_reshaped = decoder_output.squeeze(1)  # 디코더 출력 형태 변경
            encoder_output_reshaped = encoder_output.squeeze(1)  # 인코더 출력 형태 변경
            
            attention_scores = torch.matmul(decoder_output_reshaped, encoder_output_reshaped.transpose(0, 1))  # 어텐션 점수 계산
            attention_weights = F.softmax(attention_scores, dim=-1)  # 어텐션 가중치 (소프트맥스)
            
            context = torch.matmul(attention_weights, encoder_output_reshaped)  # 문맥 벡터 계산
            context = context.unsqueeze(1)  # 문맥 벡터 차원 추가
            
            combined = torch.cat([decoder_output, context], dim=-1)  # 디코더 출력과 문맥 벡터 결합
            projection = self.project(combined)  # 출력 프로젝션
            
            outputs.append(projection)  # 결과 추가
            decoder_input = torch.LongTensor([targets[i]]).to(inputs.device)  # 다음 입력을 현재 타겟으로 설정 (Teacher Forcing)
        
        outputs = torch.stack(outputs).squeeze()  # 리스트를 텐서로 결합 및 차원 압축
        return outputs  # 최종 출력 반환
    
    def translate(self, input_text, max_length=20):  # 번역 메서드
        input_seq = torch.LongTensor(list(map(ord, input_text)))  # 입력 텍스트를 아스키 시퀀스로 변환
        print(f"입력 시퀀스: {input_seq}")  # 입력 출력
        
        # 인코더
        initial_state = self._init_state(1)  # 인코더 초기 상태
        embedding = self.embedding(input_seq).unsqueeze(1)  # 입력 임베딩
        encoder_output, encoder_state = self.encoder(embedding, initial_state)  # 인코더 실행
        
        # 디코더
        decoder_state = encoder_state  # 초기 디코더 상태
        decoder_input = torch.LongTensor([ord('V')])  # 시작 기호
        
        translated_text = []  # 번역 결과 리스트
        last_char = None  # 이전 문자 기록
        repeat_count = 0  # 반복 횟수 카운트
        
        for i in range(max_length):  # 최대 길이만큼 반복
            decoder_input = self.embedding(decoder_input).unsqueeze(1)  # 디코더 입력 임베딩
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)  # 디코더 실행
            
            # 어텐션 계산
            decoder_output_reshaped = decoder_output.squeeze(1)  # 디코더 출력 변환
            encoder_output_reshaped = encoder_output.squeeze(1)  # 인코더 출력 변환
            
            attention_scores = torch.matmul(decoder_output_reshaped, encoder_output_reshaped.transpose(0, 1))  # 점수 계산
            attention_weights = F.softmax(attention_scores, dim=-1)  # 가중치 계산
            
            context = torch.matmul(attention_weights, encoder_output_reshaped)  # 문맥 벡터
            context = context.unsqueeze(1)  # 차원 추가
            
            combined = torch.cat([decoder_output, context], dim=-1)  # 결합
            projection = self.project(combined)  # 프로젝션
            
            # 소프트맥스 적용 및 예측
            probs = F.softmax(projection.squeeze(0), dim=-1)  # 확률 계산
            values, indices = torch.topk(probs, k=1)  # 상위 1개 추출
            next_char = indices.item()  # 예측된 문자 코드
            
            print(f"Step {i}:")  # 현재 단계 출력
            print(f"  Projection shape: {projection.size()}")  # 형태 출력
            print(f"  Values: {values.detach().tolist()}")  # 확률값 출력
            print(f"  Indices: {indices.detach().tolist()}")  # 인덱스 출력
            print(f"  Predicted char: {chr(next_char)} (ASCII: {next_char})")  # 예측 문자 출력
            
            # 종료 조건 체크
            if next_char < 32 or next_char > 126:  # 출력 가능한 ASCII 범위를 벗어나면 종료
                break
                
            if next_char == last_char:  # 같은 문자가 반복되는지 확인
                repeat_count += 1
                if repeat_count > 2:  # 3번 이상 반복되면 루프 종료
                    break
            else:
                repeat_count = 0  # 반복이 끊기면 카운트 리셋
                last_char = next_char
            
            # 문장 종료 조건 체크
            if len(translated_text) > 0 and next_char == ord('.'):  # 마침표가 나오면 종료
                break
                
            translated_text.append(chr(next_char))  # 결과 리스트에 문자 추가
            decoder_input = torch.LongTensor([next_char])  # 다음 단계의 입력으로 사용
        
        result = ''.join(translated_text)  # 문자 리스트를 문자열로 합침
        print(f"최종 번역 결과: {result}")  # 결과 출력
        return result  # 결과 반환
    
    def _init_state(self, batch_size=1):  # 초기 상태 생성 메서드
        weight = next(self.parameters()).data  # 모델 파라미터 참조
        return weight.new(self.n_layers, batch_size, self.hidden_size).zero_()  # 0으로 채워진 초기 상태 반환

# 모델 초기화
seq2seq = Seq2Seq(vocab_size, 64)  # Seq2Seq 인스턴스 생성 (은닉층 64)
criterion = nn.CrossEntropyLoss()  # 교차 엔트로피 손실 함수 정의
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)  # Adam 최적화 도구
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # 학습률 스케줄러 설정

# 학습
log = []  # 손실 기록 리스트
for epoch in range(1000):  # 1000 에포크 반복
    total_loss = 0  # 에포크 총 손실
    for x_text, y_text in training_pairs:  # 학습 데이터 쌍 반복
        x = torch.LongTensor(list(map(ord, x_text)))  # 입력을 아스키 텐서로 변환
        y = torch.LongTensor(list(map(ord, y_text)))  # 타겟을 아스키 텐서로 변환
        
        prediction = seq2seq(x, y)  # 모델 예측 수행
        loss = criterion(prediction, y)  # 손실 계산
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파
        torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), 1.0)  # 기울기 폭주 방지를 위한 클리핑
        optimizer.step()  # 가중치 업데이트
        total_loss += loss.item()  # 손실 합산
    
    avg_loss = total_loss / len(training_pairs)  # 평균 손실 계산
    scheduler.step(avg_loss)  # 스케줄러 단계 수행
    log.append(avg_loss)  # 로그에 추가
    
    if epoch % 100 == 0:  # 100 에포크마다 진행 상황 출력
        print(f"\nEpoch: {epoch}, Average Loss: {avg_loss}")  # 손실 출력
        # 각 학습 예제에 대한 예측 출력
        for x_text, y_text in training_pairs:  # 중간 결과 확인
            x = torch.LongTensor(list(map(ord, x_text)))
            y = torch.LongTensor(list(map(ord, y_text)))
            prediction = seq2seq(x, y)
            _, top1 = prediction.data.topk(1, 1)
            print(f"Input: {x_text}")
            print(f"Prediction: {[chr(c) for c in top1.squeeze().tolist()]}")

# 손실 그래프 출력
plt.plot(log)  # 손실 로그 그래프 그리기
plt.ylabel('cross entropy loss')  # y축 이름
plt.show()  # 그래프 표시

# 번역 테스트
print("\n번역 테스트:")  # 테스트 시작 문구
test_texts = ["I go to bed", "I want to sleep", "I play a game"]  # 테스트 입력 리스트
for text in test_texts:  # 각 테스트 문장에 대해
    print(f"\n입력: {text}")  # 입력 출력
    translated_text = seq2seq.translate(text)  # 번역 수행
    print(f"번역: {translated_text}")  # 번역 결과 출력
