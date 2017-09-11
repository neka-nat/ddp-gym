import gym
import env
from autograd import grad, jacobian
import autograd.numpy as np

env = gym.make('CartPoleContinuous-v0').env
obs = env.reset()

# x(i+1) = f(x(i), u)
next_state = lambda x, u: env._state_eq(x, u)
# l(x, u)
running_cost = lambda x, u: 0.5 * np.sum(np.square(u))
# lf(x)
final_cost = lambda x: 0.5 * (np.square(1.0 - np.cos(x[2])) + np.square(x[1]) + np.square(x[3]))

lf_x = grad(final_cost)
lf_xx = jacobian(lf_x)
l_x = grad(running_cost, 0)
l_u = grad(running_cost, 1)
l_xx = jacobian(l_x, 0)
l_uu = jacobian(l_u, 1)
l_ux = jacobian(l_u, 0)
f_x = jacobian(next_state, 0)
f_u = jacobian(next_state, 1)
f_xx = jacobian(f_x, 0)
f_uu = jacobian(f_u, 1)
f_ux = jacobian(f_u, 0)

def forward(x_seq, u_seq, k_seq, kk_seq):
    x_seq_hat = np.array(x_seq)
    u_seq_hat = np.array(u_seq)
    for t in range(len(u_seq)):
        control = k_seq[t] + np.matmul(kk_seq[t], (x_seq_hat[t] - x_seq[t]))
        u_seq_hat[t] = np.clip(u_seq[t] + control, -env.max_force, env.max_force)
        x_seq_hat[t + 1] = next_state(x_seq_hat[t], u_seq_hat[t])
    return x_seq_hat, u_seq_hat

pred_time = 30
u_seq = [np.zeros(1) for _ in range(pred_time)]
x_seq = [obs.copy()]
for t in range(pred_time):
    x_seq.append(next_state(x_seq[-1], u_seq[t]))

v = [0.0 for _ in range(pred_time + 1)]
v_x = [np.zeros(4) for _ in range(pred_time + 1)]
v_xx = [np.zeros((4, 4)) for _ in range(pred_time + 1)]

while True:
    env.render()
    v[-1] = final_cost(x_seq[-1])
    v_x[-1] = lf_x(x_seq[-1])
    v_xx[-1] = lf_xx(x_seq[-1])
    k_seq = []
    kk_seq = []
    for t in range(pred_time - 1, -1, -1):
        f_x_t = f_x(x_seq[t], u_seq[t])
        f_u_t = f_u(x_seq[t], u_seq[t])
        q_x = l_x(x_seq[t], u_seq[t]) + np.matmul(f_x_t.T, v_x[t + 1])
        q_u = l_u(x_seq[t], u_seq[t]) + np.matmul(f_u_t.T, v_x[t + 1])
        q_xx = l_xx(x_seq[t], u_seq[t]) + \
          np.matmul(np.matmul(f_x_t.T, v_xx[t + 1]), f_x_t) + \
          np.dot(v_x[t + 1], np.squeeze(f_xx(x_seq[t], u_seq[t])))
        tmp = np.matmul(f_u_t.T, v_xx[t + 1])
        q_uu = l_uu(x_seq[t], u_seq[t]) + np.matmul(tmp, f_u_t) + \
          np.dot(v_x[t + 1], np.squeeze(f_uu(x_seq[t], u_seq[t])))
        q_ux = l_ux(x_seq[t], u_seq[t]) + np.matmul(tmp, f_x_t) + \
          np.dot(v_x[t + 1], np.squeeze(f_ux(x_seq[t], u_seq[t])))
        inv_q_uu = np.linalg.inv(q_uu)
        k = -np.matmul(inv_q_uu, q_u)
        kk = -np.matmul(inv_q_uu, q_ux)
        dv = 0.5 * np.matmul(q_u, k)
        v[t] += dv
        v_x[t] = q_x + np.matmul(q_ux.T, k).T
        v_xx[t] = q_xx + np.matmul(q_ux.T, kk)
        k_seq.append(k)
        kk_seq.append(kk)
    k_seq.reverse()
    kk_seq.reverse()
    x_seq, u_seq = forward(x_seq, u_seq, k_seq, kk_seq)

    print(u_seq.T)
    obs, _, _, _ = env.step(u_seq[0])
    x_seq[0] = obs.copy()
