import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sys_params import SYS_PARAMS
from utils import *
import pandas as pd

params = SYS_PARAMS()
sim_time = 2
dt = params['Ts']
iter = sim_time/dt

q = np.array([1.15330155041141, -1.25860400808722])
dq = np.array([0.0, 0.0])

#그림그리기 위한 용도
ref_path = np.genfromtxt('reference_forest.csv', delimiter=',', skip_header=1)


trajName = 'circle'
isDesturbance = 0

rq_rec = np.zeros((int(iter)+1, 2))
rx_rec = np.zeros((int(iter)+1, 2))
ry_rec = np.zeros((int(iter)+1, 2))
x_rec = np.zeros((int(iter)+1, 2))
y_rec = np.zeros((int(iter)+1, 2))
q_rec = np.zeros((int(iter)+1, 2))
dq_rec = np.zeros((int(iter)+1, 2))
ddq_rec = np.zeros((int(iter)+1, 2))
u_rec = np.zeros((int(iter)+1, 2))
t_rec = np.zeros(int(iter)+1)

for k in range(1, int(iter) + 1):
    t = k * dt

    Theta = t
    r, XE, YE = Inverse_Kinemetic(Theta)

    if t > 1:
        dr = np.array([0.01, 0.01])
        ddr = np.array([0.001, 0.001])
    else:
        dr = np.array([0, 0])
        ddr = np.array([0, 0])

    v = Controller(q, dq, r, dr, ddr)
    u = Feedback_linearization(q, dq, v)
    # print("****")
    # print(q.shape) q, dqshape(2,)
    # print(dq.shape)
    #print(f"k = {k}")
    ddq = Arm_Dynamic(q, dq, u)
    dq += dt * ddq
    q += dt * dq
    
    x1, y1, x2, y2 = Forward_Kinemetic(q)
    # print(f"k = {k} u = {u} x = {x2:.5f} y = {y2:.5f} q = {q} dq = {dq} ddq = {ddq}")
    

    if k == 1:
        continue
    rq_rec[k, :] = r
    rx_rec[k, :] = XE
    ry_rec[k, :] = YE
    x_rec[k, :] = [x1, x2]
    y_rec[k, :] = [y1, y2]
    q_rec[k, :] = q
    dq_rec[k, :] = dq
    ddq_rec[k, :] = ddq
    u_rec[k, :] = u
    t_rec[k] = t


############################
Joint_1 = [0, 0]
Joint_2 = [1, 0]
Joint_3 = [2, 0]
fig, ax = plt.subplots()
ax.axis('equal')
ax.set_xlim(-1, 2.5)
ax.set_ylim(-1, 2.5)
ax.grid(True)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Robot Movement')

Robot_arm_1, = ax.plot([Joint_1[0], Joint_2[0]], [
                       Joint_1[1], Joint_2[1]], 'k', linewidth=4)
Robot_arm_2, = ax.plot([Joint_2[0], Joint_3[0]], [
                       Joint_2[1], Joint_3[1]], 'k', linewidth=4)
Robot_path, = ax.plot([Joint_3[0]-0.01, Joint_3[0]],
                      [Joint_3[1]-0.01, Joint_3[1]], 'r.', linewidth=0.5)
#그림그리기 위한 용도
Target_path, = ax.plot(ref_path[:, 0], ref_path[:, 1], '--b')
#Target_path, = ax.plot(rx_rec[3:, 0], ry_rec[3:, 0], '--b')

path_x, path_y = [], []

# # # save reference trajectory
# a1 = np.array(rx_rec[3:, 0]).reshape(-1, 1)
# a2 = np.array(ry_rec[3:, 0]).reshape(-1, 1)
# a3 = np.array(rq_rec[3:, 0]).reshape(-1, 1)
# a4 = np.array(rq_rec[3:, 1]).reshape(-1, 1)
# a5 = np.array(q_rec[3:, 0]).reshape(-1, 1)
# a6 = np.array(q_rec[3:, 1]).reshape(-1, 1)
# a7 = np.array(dq_rec[3:, 0]).reshape(-1, 1)
# a8 = np.array(dq_rec[3:, 1]).reshape(-1, 1)
# a9 = np.array(ddq_rec[3:, 0]).reshape(-1, 1)
# a10 = np.array(ddq_rec[3:, 1]).reshape(-1, 1)
# a11 = np.array(u_rec[3:, 0]).reshape(-1, 1)
# a12 = np.array(u_rec[3:, 1]).reshape(-1, 1)
# # # print(a1.shape)
# # # print(a2.shape)
# # # print(a3.shape)
# # # print(a4.shape)
# b = np.concatenate((a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12), axis = 1)
# # a7 = np.concatenate((a6, a4), axis = 1) #1998, 1 and x, y, q1, q2의 추종값
# print(b.shape)
# df = pd.DataFrame(b)
# df.to_csv(f'origin_reference_{sim_time}_{dt}.csv')
#중간에 1.4,0.8, 2.0.0 여러개 있어서 하나씩만 빼고 다 지움
# index가 안들어 있어서 x, y 따로 설정해서 저장하긴 함. 근데 없어도 무방. 그리고 columns 번호 삭제 해야됨

def update(frame):
    Robot_X1 = x_rec[frame, 0]
    Robot_Y1 = y_rec[frame, 0]
    Robot_X2 = x_rec[frame, 1]
    Robot_Y2 = y_rec[frame, 1]

    Robot_arm_1.set_data([Joint_1[0], Robot_X1], [Joint_1[1], Robot_Y1])
    Robot_arm_2.set_data([Robot_X1, Robot_X2], [Robot_Y1, Robot_Y2])
    Robot_path.set_data(Robot_X2, Robot_Y2)

    path_x.append(Robot_X2)
    path_y.append(Robot_Y2)
    Robot_path.set_data(path_x, path_y)

    return Robot_arm_1, Robot_arm_2, Robot_path


ani = animation.FuncAnimation(fig, update, frames=range(
    0, int(iter)+1, 10), blit=True, interval=20, repeat=True)
plt.show()

# ############################
# plt.figure(1)

# plt.subplot(2, 2, 1)
# plt.plot(t_rec, 180/np.pi*q_rec[:, 0], 'k', t_rec,
#          180/np.pi*rq_rec[:, 0], '--b', linewidth=1.2)
# plt.title('Theta 1 Input & Output')
# plt.xlabel('Time(s)')
# plt.ylabel('Theta (Deg)')
# plt.axis([0, 10, -10, 160])
# plt.legend(['Theta 1 Output', 'Theta 1 Input'])
# plt.grid(True)

# plt.subplot(2, 2, 2)
# plt.plot(t_rec, 180/np.pi*q_rec[:, 1], 'k', t_rec,
#          180/np.pi*rq_rec[:, 1], '--b', linewidth=1.2)
# plt.title('Theta 2 Input & Output')
# plt.xlabel('Time(s)')
# plt.ylabel('Theta (Deg)')
# plt.axis([0, 10, -160, 10])
# plt.legend(['Theta 2 Output', 'Theta 2 Input'])
# plt.grid(True)

# plt.subplot(2, 2, 3)
# plt.plot(t_rec, x_rec[:, 1], 'k', t_rec, rx_rec[:, 0], '--b', linewidth=1.2)
# plt.title('X(end point) Input & Output')
# plt.xlabel('Time(s)')
# plt.ylabel('X (m)')
# plt.axis([0, 10, -1, 4])
# plt.legend(['X output', 'X input'])
# plt.grid(True)

# plt.subplot(2, 2, 4)
# plt.plot(t_rec, y_rec[:, 1], 'k', t_rec, ry_rec[:, 0], '--b', linewidth=1.2)
# plt.title('Y(end point) Input & Output')
# plt.xlabel('Time(s)')
# plt.ylabel('Y (m)')
# plt.axis([0, 10, -2, 4])
# plt.legend(['Y output', 'Y input'])
# plt.grid(True)

# ############################
# plt.figure(2)

# plt.subplot(2, 1, 1)
# plt.plot(t_rec, u_rec[:, 0], 'k', linewidth=1.2)
# plt.title('u(1)')
# plt.grid(True)

# plt.subplot(2, 1, 2)
# plt.plot(t_rec, u_rec[:, 1], 'k', linewidth=1.2)
# plt.title('u(2)')
# plt.grid(True)

# plt.show()
