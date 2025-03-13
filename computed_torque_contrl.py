import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.linalg
import pandas as pd

# 시스템 파라미터 설정
params = {
    'm1': 1.0,     # 첫 번째 링크의 질량
    'm2': 1.0,     # 두 번째 링크의 질량
    'l1': 1.0,     # 첫 번째 링크 길이
    'l2': 1.0,     # 두 번째 링크 길이
    'lc1': 0.5,    # 첫 번째 링크 무게중심 위치
    'lc2': 0.5,    # 두 번째 링크 무게중심 위치
    'g': 9.81,     # 중력 가속도
    'dt': 0.01,    # 시뮬레이션 시간 간격
    'sim_time': 10 # 총 시뮬레이션 시간
}

# --- (1) 목표점 설정 (원 위를 따라가도록) ---
center_x, center_y = 1.0, 1.0
radius = 0.5
angles = np.linspace(np.pi, 3*np.pi, 30, endpoint=False)  # 반시계방향
targets = [(center_x + radius * np.cos(theta), 
            center_y + radius * np.sin(theta))
           for theta in angles]

distance_threshold = 0.02  # 목표 전환 거리
error_integral = np.array([0.0, 0.0])  # (예: 필요하다면 적분제어에 쓸 수 있음)

# --- (2) 순방향/역기구학 함수 ---
def inverse_kinematics(x, y):
    """ 단순 2관절 역기구학 (엘보 솔루션) """
    l1, l2 = params['l1'], params['l2']
    d = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    # 작업공간 범위 초과 시 예외처리
    if np.abs(d) > 1:
        raise ValueError("목표점이 로봇의 가용 작업공간을 벗어났습니다.")
    theta2 = np.arccos(d)
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2),
                                          l1 + l2 * np.cos(theta2))
    return np.array([theta1, theta2])

def forward_kinematics(q):
    """ 2관절 로봇암의 (관절 각도 -> x,y 좌표) """
    l1, l2 = params['l1'], params['l2']
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    x2 = x1 + l2 * np.cos(q[0] + q[1])
    y2 = y1 + l2 * np.sin(q[0] + q[1])
    return x1, y1, x2, y2

# --- (3) 동역학 (M, C, G) 계산 ---
def arm_dynamics(q, dq, u):
    """ ddq = M^-1 (u - C dq - G) """
    m1, m2 = params['m1'], params['m2']
    l1, l2 = params['l1'], params['l2']
    lc1, lc2 = params['lc1'], params['lc2']
    g = params['g']

    # --- M(q) ---
    M11 = m1 * lc1**2 + l1 + m2*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(q[1])) + l2
    M22 = m2 * lc2**2 + l2
    M12 = m2 * l1*lc2*np.cos(q[1]) + m2*lc2**2 + l2
    M21 = M12
    M = np.array([[M11, M12],
                  [M21, M22]])

    # --- C(q, dq) ---
    h = m2 * l1 * lc2 * np.sin(q[1])
    C = np.array([[-h*dq[1], -h*(dq[0] + dq[1])],
                  [ h*dq[0],   0             ]])

    # --- G(q) ---
    g1 = m1 * lc1*g*np.cos(q[0]) + m2*g*(lc2*np.cos(q[0] + q[1]) + l1*np.cos(q[0]))
    g2 = m2 * lc2*g*np.cos(q[0] + q[1])
    G = np.array([g1, g2])

    ddq = np.linalg.inv(M) @ (u - C @ dq - G)
    return ddq

# --- (4) [핵심 수정] 컴퓨티드 토크(피드백 선형화) + PD ---
def computed_torque_control(q, dq, q_ref):
    """
    - 비선형 항 M, C, G를 먼저 보상하여 '가상 더블 인테그레이터'로 만들고,
    - v = -Kp (q - q_ref) - Kd(dq - 0) 형태의 PD 제어를 적용.
    - u = M(q)*v + C(q,dq)*dq + G(q)
    """
    # (a) Kp, Kd 게인 설정 (상황에 따라 조정)
    Kp = np.diag([50.0, 50.0])  # 위치 오차 게인
    Kd = np.diag([10.0, 10.0])  # 속도 감쇄 게인

    # (b) v = 원하는 가속도 (PD로 간단히 구성)
    #     여기서는 dq_ref=0 가정 (정지 목표)
    e = (q - q_ref)             # 위치 오차
    edot = dq                   # 속도 자체가 오차
    v = -Kp @ e - Kd @ edot

    # (c) M, C, G 계산
    m1, m2 = params['m1'], params['m2']
    l1, l2 = params['l1'], params['l2']
    lc1, lc2 = params['lc1'], params['lc2']
    g = params['g']

    # --- M(q) ---
    M11 = m1 * lc1**2 + l1 + m2*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(q[1])) + l2
    M22 = m2 * lc2**2 + l2
    M12 = m2 * l1*lc2*np.cos(q[1]) + m2*lc2**2 + l2
    M21 = M12
    M = np.array([[M11, M12],
                  [M21, M22]])

    # --- C(q, dq) ---
    h = m2 * l1 * lc2 * np.sin(q[1])
    C = np.array([[-h*dq[1], -h*(dq[0]+dq[1])],
                  [ h*dq[0],   0             ]])

    # --- G(q) ---
    g1 = m1 * lc1*g*np.cos(q[0]) + m2*g*(lc2*np.cos(q[0] + q[1]) + l1*np.cos(q[0]))
    g2 = m2 * lc2*g*np.cos(q[0] + q[1])
    G = np.array([g1, g2])

    # (d) 컴퓨티드 토크
    u = M @ v + C @ dq + G
    return u

# 시뮬레이션 설정
dt = params['dt']
iter_num = int(params['sim_time'] / dt)

# 초기 상태
q = np.array([np.pi/2, 0.0])  # q1, q2
dq = np.array([0.0, 0.0])     # dq1, dq2
target_index = 0
q_ref = inverse_kinematics(*targets[target_index])

# 기록용
data_records = []
q_rec = []
x_end_effector = []
y_end_effector = []
target_indices = []

for k in range(iter_num):
    # --- (5) 제어 입력 계산 (컴퓨티드 토크) ---
    u = computed_torque_control(q, dq, q_ref)

    # --- (6) 동역학 시뮬레이션 ---
    ddq = arm_dynamics(q, dq, u)
    dq += dt * ddq
    q += dt * dq

    # 기록
    q_rec.append(q.copy())
    x1, y1, x2, y2 = forward_kinematics(q)
    x_end_effector.append(x2)
    y_end_effector.append(y2)
    target_indices.append(target_index)

    data_records.append([
        k*dt, q[0], q[1], dq[0], dq[1], ddq[0], ddq[1], x2, y2, u[0], u[1], target_index
    ])

    # --- (7) 목표 변경 체크 ---
    distance = np.linalg.norm([x2 - targets[target_index][0],
                               y2 - targets[target_index][1]])
    if distance < distance_threshold and target_index < len(targets) - 1:
        target_index += 1
        q_ref = inverse_kinematics(*targets[target_index])

# (8) 애니메이션
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Computed Torque (Feedback Linearization) + PD Control")
ax.grid(True, linestyle="--")

robot_arm, = ax.plot([], [], 'ko-', linewidth=4)
path, = ax.plot([], [], 'r.', markersize=2)
target_markers = ax.scatter(*zip(*targets), s=80, marker='X')
target_text = ax.text(0.05, 0.05, f"Target: {target_indices[0]+1}",
                      transform=ax.transAxes, fontsize=12)

def update(frame):
    q_frame = q_rec[frame]
    x1, y1, x2, y2 = forward_kinematics(q_frame)
    robot_arm.set_data([0, x1, x2], [0, y1, y2])
    path.set_data(x_end_effector[:frame], y_end_effector[:frame])

    current_target = target_indices[frame]
    # 타깃 색깔 업데이트(간단히 모두 검정으로)
    colors = ['g']*len(targets)
    colors[current_target] = 'b'
    target_markers.set_color(colors)
    target_text.set_text(f"Target: {current_target+1}")
    return robot_arm, path, target_markers, target_text

ani = animation.FuncAnimation(fig, update, frames=len(q_rec), interval=30, blit=True)

# (9) 데이터 저장
df = pd.DataFrame(data_records,
                  columns=["time_step","q1","q2","dq1","dq2","ddq1","ddq2",
                           "x","y","u1","u2","target_index"])
df.to_csv("computed_torque_data.csv", index=False)
print("✅ 데이터 저장 완료: computed_torque_data.csv")

plt.show()
