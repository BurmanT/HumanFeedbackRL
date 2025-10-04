import gym 
import pygame 
import datetime as dt 
import time 
import os
import uuid
from itertools import count
from pathlib import Path
from csv import DictWriter

import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import imageio  # Add this import at the top

"""
SGDFunctionApproximator code from: https://github.com/benibienz/TAMER/tree/master
"""

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

class SGDFunctionApproximator:
    """ SGD function approximator with RBF preprocessing. """
    def __init__(self, env):
        
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        )
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(env.action_space.n):
            state, _ = env.reset()  # Unpack state and info
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(state)], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """ Returns the featurized representation for a state. """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

class Tamer:
    def __init__(self, env_name='MountainCar-v0', episodes_train=3, max_steps_per_episode=100, render=False,
                 discount_factor=1, epsilon=0, min_eps=0, tame=False, ts_len=0.2, output_dir=LOGS_DIR, model_file_to_load=None):
        
        self.env = gym.make(env_name, max_episode_steps=max_steps_per_episode, 
                            render_mode="human" if render else "rgb_array")
        self.episodes_train = episodes_train
        self.max_steps_per_episode = max_steps_per_episode
        self.render = render
        self.action_dict = {0: 'Left', 1: 'No Push', 2: 'Right'}
        #self.H = SGDFunctionApproximator(self.env)  # init H function
        self.ts_len = ts_len  # length of post-action feedback window
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.model_file_to_load = model_file_to_load
        # hyperparameters 
        self.discount_factor = discount_factor
        self.min_eps = min_eps
        self.epsilon = epsilon if not tame else 0 # for epsilon-greedy policy
        self.tame = tame  # if false, vanilla Q learning
        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                self.H = SGDFunctionApproximator(self.env)  # init H function
            else:  # optionally run as standard Q Learning
                self.Q = SGDFunctionApproximator(self.env)  # init Q function
        os.makedirs(self.output_dir, exist_ok=True)
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')
        # Initialize pygame for feedback 
        self.rewards_history = []
        pygame.init()
        pygame.display.set_caption("Human Feedback: W=Positive, A=Negative")
    
    def get_human_feedback(self):
        feedback = 0
        # waiting = True
        #print("Human Feedback: W=Positive, A=Negative")
        #while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    feedback = 1
                    print("Human Feedback: W, Positive")
                    break 
                elif event.key == pygame.K_a:
                    feedback = -1
                    print("Human Feedback: A, Negative")
                    break
                    #waiting = False
        return feedback
    
    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)
    



    def train(self):
        state, _ = self.env.reset()
        for i in range(self.episodes_train):
            print(f'Episode: {i + 1}  Timestep:', end='')
            tot_reward = 0
            state, _ = self.env.reset()
            ep_start_time = dt.datetime.now().time()
            #print("reset episode state is ", state)
            with open(self.reward_log_path, 'a+', newline='') as write_obj:
                dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
                dict_writer.writeheader()
            
                done = False
                step = 0
                while not done and step < self.max_steps_per_episode:
                    # render MountainCar in its own window 
                    if self.render:
                        self.env.render()

                    # Take action
                    action = self.act(state)
                    #print("Action taken is chosen ", action)
                    #print("Timestep {step} Action Taken:", self.action_dict[action])
                
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    #print(state, reward, terminated, truncated, info)
                    #key=input("stop here")
                    #tot_reward += reward
                    
                    if not self.tame:
                        if done and next_state[0] >= 0.5:
                            td_target = reward
                        else:
                            td_target = reward + self.discount_factor * np.max(
                                self.Q.predict(next_state)
                            )
                        self.Q.update(state, action, td_target)
                    else:
                        now = time.time()
                        while time.time() < now + self.ts_len:
                            time.sleep(0.01)
                            human_reward = self.get_human_feedback()
                            feedback_ts = dt.datetime.now().time()
                            if human_reward != 0:
                                dict_writer.writerow(
                                    {
                                        'Episode': i + 1,
                                        'Ep start ts': ep_start_time,
                                        'Feedback ts': feedback_ts,
                                        'Human Reward': human_reward,
                                        'Environment Reward': reward
                                    }
                                )
                                self.H.update(state, action, human_reward)
                                break
                    tot_reward += reward
                    step += 1
                    state = next_state
            # Decay epsilon
            if self.epsilon > self.min_eps:
                self.epsilon -= self.epsilon_step
            self.rewards_history.append(tot_reward)         
            print(f"Episode {i+1} finished after {step} steps with total reward {tot_reward}")
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.rewards_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.show()
    
    def play(self, n_episodes=1, render=True):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        self.epsilon = 0
        ep_rewards = []
        frames = []
        for i in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            tot_reward = 0
            while not done:
                action = self.act(state)
                # next_state, reward, done, info = self.env.step(action)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                tot_reward += reward
                if render:
                    self.env.render()
                frame = self.env.render()  # if render_mode="rgb_array" was set at env creation
                frames.append(frame)
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        imageio.mimsave("agent_play.gif", frames, fps=30)
        print(f"Saved GIF to {'agent_play.gif'}")
        #self.env.close()
        return ep_rewards
    
    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        return avg_reward

if __name__ == '__main__':

    tame = True  # set to false for vanilla Q learning

    if(tame):
        # get feedback for 5 episodes 
        episodes_train = 5 
        render = True
    else:
        # train vanilla q learning for 500 episodes
        episodes_train = 500
        render = False

    env_name = 'MountainCar-v0'
    max_steps_per_episode = 200
    # hyperparameters
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0 # lowest epsilon value 
    
    # tame = True # TO DO: False  # set to false for vanilla Q learning
    # render = True # TO DO: False # set to false for vanilla Q learning
    # episodes_train = 5 # TO DO: 500 # set to more episodes for vanilla Q learning
   
    tame = Tamer(env_name=env_name, episodes_train=episodes_train, max_steps_per_episode=max_steps_per_episode, render=render)
    tame.train()
    tame.play(n_episodes=1, render=True)
    #tame.evaluate(n_episodes=30)
    tame.plot()
    tame.env.close()  # <-- Only close once, here
    pygame.quit()