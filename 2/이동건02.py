import numpy as np

def sigma(x):
    ret = 0  # 합계 변수 초기화
    for x_i in x:  # 리스트 x의 각 원소 x_i를 순회
        ret += x_i  # 현재 원소를 합계에 더함
    return ret  # 최종 합계 반환

def argmax(target_func, X):
    max_val = -np.inf  # 최대값을 음의 무한대로 초기화
    ret = None  # 결과 변수 초기화
    for x in X:  # 입력 리스트 X의 각 원소를 순회
        val = target_func(x)  # 현재 원소에 대한 함수값 계산
        if val > max_val:  # 현재 값이 기존 최대값보다 크면
            max_val = val  # 최대값 갱신
            ret = x  # 입력값 저장
    return ret  # 최대값 입력값 반환

def pi(n):
    ret = 0  # 합계 변수 초기화
    for i in range(n):  # 0부터 n-1까지 반복
        term = ((-1) ** i) / (2 * i + 1)  # 라이프니츠 급수 각 항 계산
        ret += term  # 항 더함
    return 4 * ret  # 최종 합계에 4를 곱하여 pi 근사값 반환

x_data = [i + 1 for i in range(10)]  # 1부터 10까지 리스트 생성
print(sigma(x_data))  # Sigma 결과 출력


X_data = [i + 1 for i in range(10)]  # 1부터 10까지 리스트 생성
def f(x):
    return -np.log(x)  # 테스트용 함수 f(x) = -log(x) 정의

print(argmax(f, X_data))  # Argmax 결과 출력

print(pi(1000)) # Pi 근사값 출력 (n=1000)
