import os
import socket
import time
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import csv

HISTORY_LEN = 4
FEATURE_DIM = 7
ACTION_SPACE = 9
target_fps = 15
target_temp = 80
experiment_time = 2000

log_file = open('state_log.csv','w',newline='',encoding='utf-8')
log_writer = csv.writer(log_file)
log_writer.writerow(['t','state','action','next_state','reward','fps','avg_q','loss'])

class DQNAgent:
    def __init__(self, history_len, feature_dim, action_size):
        self.history_len = history_len
        self.feature_dim = feature_dim
        self.action_size = action_size
        self.load_model = False
        
        self.clk_action_list = []
        for i in range(3):
            for j in range(3):
                clk_action = (4*i + 3, 4*j + 3)
                self.clk_action_list.append(clk_action)

        self.learning_rate = 0.005
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 16
        self.train_start = 1000

        self.memory = deque(maxlen=1000)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.q_max = 0
        self.avg_q_max = 0
        self.currentLoss = 0

        if self.load_model:
                  self.model = load_model("results/attention technqiue fix temp 80/timesteps=300/model_video_fps30_full_1.h5")

    def build_model(self):
        model = Sequential()
        model.add(LSTM(4, input_shape=(self.history_len, self.feature_dim), return_sequences=False))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, stacked_state):
        stacked_state = np.expand_dims(stacked_state, axis=0)
        print(self.epsilon)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(stacked_state, verbose=0)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        
        print('length of self memory', len(self.memory))

        states = np.zeros((self.batch_size, self.history_len, self.feature_dim))
        next_states = np.zeros((self.batch_size, self.history_len, self.feature_dim))
        actions, rewards, dones = [], [], []

        for i, (s, a, r, s_next, d) in enumerate(mini_batch):
            states[i] = s
            next_states[i] = s_next
            actions.append(a)
            rewards.append(r)
            dones.append(d)

        target = self.model.predict(states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])

        hist = self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.currentLoss = hist.history['loss'][0]


def get_reward(fps, power, target_fps, c_t, target_temp,
               alpha_fps=0.1, alpha_power=1, alpha_temp=0.5):
    fps_ratio = fps / float(target_fps)
    
    f_fps = min(fps_ratio, 2.0)
    f_power = 1.0 / power if power > 0 else 0.0
    temp_diff = c_t - target_temp
    f_temp = 0.1 * (-temp_diff) if temp_diff <= 0 else -1.0 * temp_diff
    return alpha_fps * f_fps + alpha_power * f_power + alpha_temp * f_temp


if __name__ == "__main__":
    agent = DQNAgent(HISTORY_LEN, FEATURE_DIM, ACTION_SPACE)
    state_deque = deque(maxlen=HISTORY_LEN)

    dummy_state = (11, 11, 0, 0, 50.0, 50.0, 0.0)
    for _ in range(HISTORY_LEN):
        state_deque.append(dummy_state)

    stacked_state = np.array(state_deque)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", 8703))
    server_socket.listen(5)

    print("Waiting for connection...")
    client_socket, address = server_socket.accept()
    prev_stacked_state = None
    t = 0

    while t < experiment_time:
        msg = client_socket.recv(1024).decode()
        if not msg:
            print("No message received. Exiting...")
            break

        values = list(map(float, msg.split(',')))
        if len(values) != HISTORY_LEN * FEATURE_DIM:
            print(f"Incorrect input size. Expected {HISTORY_LEN * FEATURE_DIM} floats.")
            break

        # Convert the flat 28-element list into a (4,7) stacked state
        current_stacked_state = np.array(values).reshape((HISTORY_LEN, FEATURE_DIM))

        # Extract latest observation from the current stacked state
        latest = current_stacked_state[-1]
        c_c, g_c, c_p, g_p, c_t, g_t, fps = latest
        power = c_p + g_p

        print(f'time step: {t} with fps: {fps}')
        reward = get_reward(fps, power, target_fps, c_t, target_temp)

        if prev_stacked_state is not None:
            action = agent.get_action(prev_stacked_state)
            agent.q_max += np.amax(agent.model.predict(np.expand_dims(prev_stacked_state, axis=0), verbose=0))
            agent.avg_q_max = agent.q_max / (t)

            # print('This is the current state', prev_stacked_state)
            # print('This is the next stacked state', current_stacked_state)

            done = True  # You can define proper logic for episode ends
            agent.append_sample(prev_stacked_state, action, reward, current_stacked_state, done)

            log_writer.writerow([t, prev_stacked_state, action, current_stacked_state, reward, fps, agent.avg_q_max, agent.currentLoss])

            if len(agent.memory) >= agent.train_start:
                agent.train_model()
        prev_stacked_state = current_stacked_state
        t += 1
        
        # print('CT',c_t)
        # print('TARGET TEMP ', target_temp)
  
        if c_t >= target_temp:
            c_c = int(4 * random.randint(0, int(c_c / 3) - 1) + 3)
            g_c = int(4 * random.randint(0, int(g_c / 3) - 1) + 3)
            action = 3 * int(c_c / 4) + int(g_c / 4)
        elif target_temp - c_t >= 10:
            if fps < target_fps:
                if np.random.rand() <= 0.3:
                    c_c = 11
                    g_c = 11
            else:
                c_c = agent.clk_action_list[action][0]
                g_c = agent.clk_action_list[action][1]
        else:
            c_c = agent.clk_action_list[action][0]
            g_c = agent.clk_action_list[action][1]
        
        print(c_c,g_c)

        clk_msg = f"{int(c_c)},{int(g_c)}"
        client_socket.send(clk_msg.encode())

        if t % 100 == 0:
            agent.update_target_model()
        if t == 299:
            agent.model.save("model_video_fps30_full.h5")
            print("[Model saved]")

    server_socket.close()
    print("Experiment completed.")
