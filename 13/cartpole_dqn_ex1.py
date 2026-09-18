#!/usr/bin/env python
# coding: utf-8

# # 카트폴 게임 마스터하기

import gym  # OpenAI Gym 라이브러리 임포트
import random  # 무작위 추출을 위한 random 임포트
import math  # 수학 함수를 위한 math 임포트
import torch  # 파이토치 라이브러리 임포트
import torch.nn as nn  # 신경망 모듈 임포트
import torch.optim as optim  # 최적화 알고리즘 임포트
import torch.nn.functional as F  # 신경망 함수 모듈 임포트
from collections import deque  # 큐 구현을 위한 deque 임포트
import matplotlib.pyplot as plt  # 그래프 출력을 위한 matplotlib 임포트
import numpy as np  # 수치 계산을 위한 numpy 임포트

# Fix for numpy bool8 issue
np.bool8 = np.bool_  # numpy 하위 호환성을 위한 별칭 설정

# ### 하이퍼파라미터
# 하이퍼파라미터
EPISODES = 50    # 총 에피소드 반복 횟수
EPS_START = 0.9  # 무작위 행동 확률의 시작값
EPS_END = 0.05   # 무작위 행동 확률의 최종값
EPS_DECAY = 200  # 무작위 행동 확률을 감소시키는 기준값
GAMMA = 0.8      # 할인 계수 (미래 보상의 가치 반영)
LR = 0.001       # 신경망 학습률
BATCH_SIZE = 64  # 한 번에 학습할 배치 크기


# ## DQN 에이전트

class DQNAgent:  # DQN 에이전트 클래스 정의
    def __init__(self):  # 초기화 메서드
        self.model = nn.Sequential(  # 신경망 모델 정의
            nn.Linear(4, 256),  # 입력 4(상태), 은닉 256 선형 계층
            nn.ReLU(),  # 활성화 함수 ReLU
            nn.Linear(256, 2)  # 은닉 256, 출력 2(행동) 선형 계층
        )
        self.optimizer = optim.Adam(self.model.parameters(), LR)  # Adam 최적화 도구 설정
        self.steps_done = 0  # 총 진행된 스텝 수 초기화
        self.memory = deque(maxlen=10000)  # 경험 재생을 위한 메모리(큐) 설정

    def memorize(self, state, action, reward, next_state):  # 경험 저장 메서드
        self.memory.append((state,  # 현재 상태 저장
                            action,  # 선택한 행동 저장
                            torch.FloatTensor([reward]),  # 보상을 텐서로 저장
                            torch.FloatTensor([next_state])))  # 다음 상태를 텐서로 저장
    
    def act(self, state):  # 행동 선택 메서드
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)  # 엡실론 임계값 계산
        self.steps_done += 1  # 스텝 수 증가
        if random.random() > eps_threshold:  # 확률적으로 최적 행동 선택 (Exploitation)
            return self.model(state).data.max(1)[1].view(1, 1)  # 모델의 결과값 중 최대값의 인덱스 반환
        else:  # 확률적으로 무작위 행동 선택 (Exploration)
            return torch.LongTensor([[random.randrange(2)]])  # 0 또는 1 중 무작위 반환
    
    def learn(self):  # 모델 학습 메서드
        if len(self.memory) < BATCH_SIZE:  # 메모리에 충분한 데이터가 쌓일 때까지 대기
            return
        batch = random.sample(self.memory, BATCH_SIZE)  # 메모리에서 무작위 배치 추출
        states, actions, rewards, next_states = zip(*batch)  # 데이터 분리

        states = torch.cat(states)  # 상태 텐서 결합
        actions = torch.cat(actions)  # 행동 텐서 결합
        rewards = torch.cat(rewards)  # 보상 텐서 결합
        next_states = torch.cat(next_states)  # 다음 상태 텐서 결합

        current_q = self.model(states).gather(1, actions)  # 현재 상태에서의 Q값 계산
        max_next_q = self.model(next_states).detach().max(1)[0]  # 다음 상태에서의 최대 Q값 계산
        expected_q = rewards + (GAMMA * max_next_q)  # 기대 Q값 계산 (벨만 방정식)
        
        loss = F.mse_loss(current_q.squeeze(), expected_q)  # 현재 Q값과 기대 Q값의 오차 계산
        self.optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파 수행
        self.optimizer.step()  # 가중치 업데이트


# ## 학습 준비하기
# `gym`을 이용하여 `CartPole-v1`환경을 준비하고 앞서 만들어둔 DQNAgent를 agent로 인스턴스화 합니다.
# 자, 이제 `agent` 객체를 이용하여 `CartPole-v1` 환경과 상호작용을 통해 게임을 배우도록 하겠습니다.

env = gym.make('CartPole-v1', render_mode='human')  # 카트폴 환경 생성 (렌더링 모드 설정)
agent = DQNAgent()  # DQN 에이전트 인스턴스 생성
score_history = []  # 점수 기록을 위한 리스트 초기화


# ## 학습 시작

for e in range(1, EPISODES+1):  # 설정한 에피소드만큼 반복
    state, _ = env.reset()  # 환경 초기화 및 초기 상태 획득
    steps = 0  # 스텝 점수 초기화
    while True:  # 에피소드 종료 시까지 반복
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)  # 현재 상태를 텐서로 변환
        action = agent.act(state)  # 에이전트가 행동 선택
        next_state, reward, terminated, truncated, info = env.step(action.item())  # 행동 실행 및 결과 획득
        done = terminated or truncated  # 종료 조건 확인

        # 게임이 끝났을 경우 마이너스 보상주기
        if done:  # 게임이 비정상 종료 시
            reward = -1  # 패널티 부여

        agent.memorize(state, action, reward, next_state)  # 경험 저장
        agent.learn()  # 학습 수행

        state = next_state  # 상태 업데이트
        steps += 1  # 스텝 점수 증가

        if done:  # 에피소드가 끝나면
            print("에피소드:{0} 점수: {1}".format(e, steps))  # 결과 출력
            score_history.append(steps)  # 점수 저장
            break  # 반복 종료


plt.plot(score_history)  # 점수 추이 그래프 그리기
plt.ylabel('score')  # y축 레이블 설정
plt.show()  # 그래프 출력
