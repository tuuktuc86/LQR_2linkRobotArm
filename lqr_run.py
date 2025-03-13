import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg
import pandas as pd

# 시스템 파라미터 설정
params = {
    'm1': 1.0,  # 첫 번째 링크의 질량
    'm2': 1.0,  # 두 번째 링크의 질량
    'l1': 1.0,  # 첫 번째 링크 길이
    'l2': 1.0,  # 두 번째 링크 길이
    'lc1': 0.5,  # 첫 번째 링크의 무게 중심
    'lc2': 0.5,  # 두 번째 링크의 무게 중심
    'g': 9.81,   # 중력 가속도
    'dt': 0.01,  # 시뮬레이션 시간 간격
    'sim_time': 10  # 총 시뮬레이션 시간
}

# 목표점 리스트 (순차적으로 이동할 목표들)
# targets = [(1.5, 1.0), (1.0, 1.5), (0.5, 1.0), (1.0, 0.5)]
# 원의 중심과 반지름 설정
center_x, center_y = 1.0, 1.0
radius = 0.5

# 원 위의 점 20개 생성 (각도를 균등하게 배분)
# 🔹 반시계 방향 회전을 위해 시작 각도를 π/2 (90도)로 설정
# 🔹 180도(π)에서 시작하여 반시계 방향으로 진행 (360도(2π) 이후까지 진행)
angles = np.linspace(np.pi, 3 * np.pi, 30, endpoint=False)

# 🔹 (x, y) 좌표 계산
targets = [(center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)) for theta in angles]

distance_threshold = 0.02  # 목표 전환 거리
error_integral = np.array([0.0, 0.0])

# 역기구학(Inverse Kinematics)
def inverse_kinematics(x, y):
    l1, l2 = params['l1'], params['l2']
    d = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if np.abs(d) > 1:
        raise ValueError("목표점이 로봇의 가용 작업공간을 벗어났습니다.")
    theta2 = np.arccos(d)
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    return np.array([theta1, theta2])

# 순방향 기구학
def forward_kinematics(q):
    l1, l2 = params['l1'], params['l2']
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    x2 = x1 + l2 * np.cos(q[0] + q[1])
    y2 = y1 + l2 * np.sin(q[0] + q[1])
    return x1, y1, x2, y2

# LQR 제어기
def lqr_control(q, q_ref):
    global error_integral  # 적분 항 유지
    A = np.zeros((2, 2))
    B = np.eye(2)
    Q = np.diag([3000, 3000])  # 위치 오차 최소화
    R = np.diag([0.01, 0.01])
    
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    # 추가 P 제어기
    K_p = np.array([[10, 0], [0, 10]])

    # 적분 항 (Integral Control)
    K_I = np.array([[1, 0], [0, 1]])  # 적분 이득 추가
    error_integral += (q_ref - q) * params['dt']  # 오차 적분
    # ✅ **적분 항 제한 (Anti-Windup)**
    error_integral = np.clip(error_integral, -0.5, 0.5)  

    # # ✅ **목표 근처에서는 적분 항 감소**
    # if np.linalg.norm(q_ref - q) < 0.02:
    #     error_integral *= 0.9 
    #u = -K @ (q - q_ref) - K_p @ (q - q_ref) - K_I @ error_integral
    u = -K @ (q - q_ref) - K_p @ (q - q_ref) 
    # # 최소 토크 하한 설정
    # u_min = 0.05
    # u = np.where(np.abs(u) < u_min, np.sign(u) * u_min, u)
    
    return u

# **💡 복구된 동역학 모델**
# def arm_dynamics(q, u):
#     m1, m2, l1, l2, lc1, lc2, g = (
#         params['m1'], params['m2'], params['l1'], params['l2'],
#         params['lc1'], params['lc2'], params['g']
#     )
    
#     M11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(q[1]))
#     M12 = m2 * (lc2**2 + l1 * lc2 * np.cos(q[1]))
#     M21 = M12
#     M22 = m2 * lc2**2
#     M = np.array([[M11, M12], [M21, M22]])
    
#     C1 = -m2 * l1 * lc2 * np.sin(q[1])
#     C = np.array([[C1 * q[1], C1 * (q[0] + q[1])],
#                   [-C1 * q[0], 0]])
    
#     G = np.array([
#         m1 * lc1 * g * np.cos(q[0]) + m2 * g * (lc2 * np.cos(q[0] + q[1]) + l1 * np.cos(q[0])),
#         m2 * lc2 * g * np.cos(q[0] + q[1])
#     ])
    
#     ddq = np.linalg.inv(M) @ (u - C @ np.zeros(2) - G)
#     return ddq

def arm_dynamics(q, dq, u):
    m1, m2, l1, l2, lc1, lc2, g = (
        params['m1'], params['m2'], params['l1'], params['l2'],
        params['lc1'], params['lc2'], params['g']
    )
    M11 = m1 * lc1 ** 2 + l1 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q[1])) + l2
    M22 = m2 * lc2 ** 2 + l2
    M12 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
    M21 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
    M = np.array([[M11, M12], [M21, M22]])
    h = m2 * l1 * lc2 * np.sin(q[1])
    g1 = m1 * lc1 * g * np.cos(q[0]) + m2 * g * \
        (lc2 * np.cos(q[0] + q[1]) + l1 * np.cos(q[0]))
    g2 = m2 * lc2 * g * np.cos(q[0] + q[1])
    G = np.array([g1, g2])
    C = np.array([[-h * dq[1], -h * dq[0] - h * dq[1]], [h * dq[0], 0]])
    ddq = np.linalg.inv(M).dot(u - C.dot(dq) - G)

    return ddq

# 시뮬레이션 설정
dt = params['dt']
iter = int(params['sim_time'] / dt)

# 초기 상태
q = np.array([np.pi / 2, 0])
dq = np.array([0.0, 0.0])
target_index = 0
q_ref = inverse_kinematics(*targets[target_index])
data_records = []
q_rec, x_end_effector, y_end_effector, target_indices = [], [], [], []

for k in range(iter):
    u = lqr_control(q, q_ref)
    ddq = arm_dynamics(q, dq, u)  # **💡 arm_dynamics 복구!**
    q_ddot1_max = 2000
    q_ddot2_max = 2000
    # ddq[0] = np.clip(ddq[0], -30, 30)
    # ddq[1] = np.clip(ddq[1], -10, 10)
    #ddq = np.clip(ddq, -10.0, 10.0)  # 가속도 제한

    dq += dt * ddq
    #dq *= 0.98  # 감쇠 적용 (속도를 서서히 줄임)
    #dq = np.clip(dq, -5.0, 5.0)  # 속도 제한
    q += dt * dq  

    q_rec.append(q.copy())
    
    x1, y1, x2, y2 = forward_kinematics(q)
    x_end_effector.append(x2)
    y_end_effector.append(y2)
    target_indices.append(target_index)
    # 데이터 저장
    data_records.append([
        k * dt, q[0], q[1], dq[0], dq[1], ddq[0], ddq[1], x2, y2, u[0], u[1], target_index
    ])  

    # 목표 변경 체크
    distance = np.linalg.norm([x2 - targets[target_index][0], y2 - targets[target_index][1]])
    if distance < distance_threshold and target_index < len(targets) - 1:
        error_integral *= 0.5
        target_index += 1
        q_ref = inverse_kinematics(*targets[target_index])

# 애니메이션 설정
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("LQR + PI Control for Multiple Targets")
ax.grid(True, linestyle="--")

robot_arm, = ax.plot([], [], 'ko-', linewidth=4)
path, = ax.plot([], [], 'r.', markersize=2)
target_markers = ax.scatter(*zip(*targets), color=['g'] * len(targets), s=80)
target_text = ax.text(0.05, 0.05, f"Target: {target_indices[0] + 1}", transform=ax.transAxes, fontsize=12)

def update(frame):
    q_frame = q_rec[frame]
    x1, y1, x2, y2 = forward_kinematics(q_frame)
    robot_arm.set_data([0, x1, x2], [0, y1, y2])
    path.set_data(x_end_effector[:frame], y_end_effector[:frame])

    # 현재 목표 인덱스
    current_target = target_indices[frame]

    # 목표 색상 업데이트
    target_colors = ['g'] * len(targets)
    target_colors[current_target] = 'b'
    target_markers.set_color(target_colors)

    # 타겟 번호 업데이트
    target_text.set_text(f"Target: {current_target + 1}")

    return robot_arm, path, target_text, target_markers
df = pd.DataFrame(data_records, columns=["time_step", "q1", "q2", "dq1", "dq2", "ddq1", "ddq2", "x", "y", "u1", "u2", "target_index"])
df.to_csv("simulation_data.csv", index=False)
print("✅ 데이터 저장 완료: simulation_data.csv")
ani = animation.FuncAnimation(fig, update, frames=len(q_rec), interval=30, blit=True)
plt.show()
