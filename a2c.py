import argparse
import gym
import os
import sys
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

import matplotlib.pyplot as plt
plt.switch_backend('agg')

#import ipdb

# if gpu is to be used
use_cuda = torch.cuda.is_available()
#use_cuda = False
print("use_cuda : ", use_cuda)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class Actor(nn.Module):
    def __init__(self, state_size, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x),dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(16, 8)
        #self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class A2C(object):
    def __init__(self, env, args):
        super(A2C, self).__init__()
        self.env = env
        self.actor = Actor(env.observation_space.shape[0], env.action_space.n)
        self.critic = Critic(env.observation_space.shape[0])
        if use_cuda:
            self.actor.cuda()
            self.critic.cuda()
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        self.N_steps = args.N_steps
        self.num_episodes = args.num_episodes
        self.test_episodes = args.test_episodes
        #self.num_steps = args.num_steps
        self.gamma = args.gamma
        self.expt_name = args.expt_name
        self.save_path = args.save_path
        self.test_freq = args.test_freq
        self.save_freq = args.save_freq
        self.train_rewards = []
        self.test_rewards = []
        self.train_steps = []
        self.test_steps = []
        self.losses_actor = []
        self.losses_critic = []

    def select_action(self, state):
        state = Variable(Tensor(state))
        log_probs = self.actor(state)
        value = self.critic(state)
        action = Categorical(log_probs.exp()).sample()
        return action.data.cpu().numpy()[0], log_probs[action], value

    def play_episode(self, e):
        state = self.env.reset()
        steps = 0
        rewards = []
        log_probs = []
        values = []
        # for i in range(self.num_steps):
        while True:
            action, log_prob, value = self.select_action(state)
            state, reward, is_terminal, _ = self.env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            steps +=1
            if is_terminal:
                break
        return steps, rewards, torch.cat(log_probs), torch.cat(values)

    def optimize(self, rewards, log_probs, values):
        T = len(rewards)
        N = self.N_steps
        R = np.zeros(T, dtype=np.float32)
        loss_actor = 0
        loss_critic = 0
        for t in reversed(range(T)):
            V_end = 0 if (t+N >= T) else values[t+N].data
            R[t] = (self.gamma**N * V_end) + sum([self.gamma**k * rewards[t+k]*1e-2 for k in range(min(N, T-t))])
        R = Variable(Tensor(R), requires_grad=False)
        # compute losses using the advantage function;
        # Note: `values` is detached while computing loss for actor
        loss_actor = ((R - values.detach()) * -log_probs).mean()
        loss_critic = ((R - values)**2).mean()
        # loss = loss_actor + loss_critic

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss_actor.backward()
        loss_critic.backward()
        # nn.utils.clip_grad_norm(self.actor.parameters(), grad_norm_limit)
        # nn.utils.clip_grad_norm(self.critic.parameters(), grad_norm_limit)
        self.optimizer_actor.step()
        self.optimizer_critic.step()
        # self.losses.append(loss.detach().cpu().numpy())
        # ipdb.set_trace()
        self.losses_actor.append(loss_actor.data.cpu().numpy()[0])
        self.losses_critic.append(loss_critic.data.cpu().numpy()[0])

    def train(self, num_episodes):
        print("Going to be training for a total of {} episodes".format(num_episodes))
        state = Variable(torch.Tensor(self.env.reset()))
        for e in range(num_episodes):
            steps, rewards, log_probs, values = self.play_episode(e)
            self.train_rewards.append(sum(rewards))
            self.train_steps.append(steps)
            self.optimize(rewards, log_probs,values)

            if (e+1) % 100 == 0:
                print("Episode: {}, reward: {}, steps: {}".format(e+1, sum(rewards), steps))

            # Freeze the current policy and test over 100 episodes
            if (e+1) % self.test_freq == 0:
                print("-"*10 + " testing now " + "-"*10)
                self.test(self.test_episodes, e)

            # Save the current policy model
            if (e+1) % (self.save_freq) == 0:
                torch.save(self.actor.state_dict(),  os.path.join(self.save_path, "train_actor_ep_{}.pkl".format(e+1)))
                torch.save(self.critic.state_dict(), os.path.join(self.save_path, "train_critic_ep_{}.pkl".format(e+1)))

        # plot once when done training
        self.plot_rewards(save=True)

    def test(self, num_episodes, e_train):
        state = Variable(torch.Tensor(self.env.reset()))
        testing_rewards = []
        testing_steps = []
        for e in range(num_episodes):
            steps, rewards, log_probs,values = self.play_episode(e)
            self.test_rewards.append(sum(rewards))
            self.test_steps.append(steps)
            testing_rewards.append(sum(rewards))
            testing_steps.append(steps)
        print("Mean reward achieved : {} ".format(np.mean(testing_rewards)))
        print("-"*50)
        if np.mean(testing_rewards) >= 200:
            print("-"*10 + " Solved! " + "-"*10)
            print("Mean reward achieved : {} in {} steps".format(np.mean(testing_rewards), np.mean(testing_steps)))
            print("-"*50)
            # if (e_train+1) % 5000 == 0: self.plot_rewards(save=True)
            # else: self.plot_rewards(save=False)
        if (e_train+1) % 5000 == 0: self.plot_rewards(save=True)
        #else: self.plot_rewards(save=False)

    def plot_rewards(self, save=False):
        train_rewards = [self.train_rewards[i:i+self.test_freq] for i in range(0,len(self.train_rewards),self.test_freq)]
        test_rewards = [self.test_rewards[i:i+self.test_episodes] for i in range(0,len(self.test_rewards),self.test_episodes)]
        train_losses_actor = [self.losses_actor[i:i+self.test_freq] for i in range(0,len(self.losses_actor),self.test_freq)]
        train_losses_critic = [self.losses_critic[i:i+self.test_freq] for i in range(0,len(self.losses_critic),self.test_freq)]
        train_losses = [self.losses_critic[i:i+self.test_freq]+self.losses_actor[i:i+self.test_freq] for i in range(0,len(self.losses_critic),self.test_freq)]

        # rewards
        train_rewards_mean = [np.mean(i) for i in train_rewards]
        test_rewards_mean = [np.mean(i) for i in test_rewards]
        train_rewards_std = [np.std(i) for i in train_rewards]
        test_rewards_std = [np.std(i) for i in test_rewards]
        train_nepisodes = [self.test_freq * (i+1) for i in range(len(train_rewards_mean))]

        # steps
        train_steps = [self.train_steps[i:i+self.test_freq] for i in range(0,len(self.train_steps),self.test_freq)]
        test_steps = [self.test_steps[i:i+self.test_episodes] for i in range(0,len(self.test_steps),self.test_episodes)]
        train_steps_mean = [np.mean(i) for i in train_steps]
        test_steps_mean = [np.mean(i) for i in test_steps]
        train_steps_std = [np.mean(i) for i in train_steps]
        test_steps_std = [np.mean(i) for i in test_steps]

        # loss
        train_losses_actor_mean = [np.mean(i) for i in train_losses_actor]
        train_losses_actor_std = [np.std(i) for i in train_losses_actor]
        train_losses_critic_mean = [np.mean(i) for i in train_losses_critic]
        train_losses_critic_std = [np.std(i) for i in train_losses_critic]
        train_losses_mean = [np.mean(i) for i in train_losses]
        train_losses_std = [np.std(i) for i in train_losses]


        # training : reward over time
        plt.figure(1)
        plt.clf()
        plt.title("Training : Avg. Reward over {} episodes".format(self.test_episodes))
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg Reward")
        plt.errorbar(train_nepisodes, train_rewards_mean, yerr=train_rewards_std, color="indigo", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "train_rewards_{}.png".format(len(self.train_rewards)))
        else:
            plt.show()
            # pause so that the plots are updated
            plt.pause(0.001)

        # testing : reward over time
        plt.figure(2)
        plt.clf()
        plt.title("Testing : Avg. Reward over {} episodes".format(self.test_episodes))
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg Reward")
        try:
            plt.errorbar(train_nepisodes, test_rewards_mean, yerr=test_rewards_std, color="indigo", uplims=True, lolims=True)
        except:
            ipdb.set_trace()
        if save :
            plt.savefig(self.expt_name + "test_rewards_{}.png".format(len(self.test_rewards)))
        else:
            plt.show()

        # training : avg number of steps per episode
        plt.figure(3)
        plt.clf()
        plt.title("Training : Avg. number of steps taken per episode")
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg number of steps")
        plt.errorbar(train_nepisodes, train_steps_mean, yerr=train_steps_std, color="navy", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "train_steps_{}.png".format(len(self.train_steps)))
        else:
            plt.show()
            # pause so that the plots are updated
            plt.pause(0.001)

        # testing : avg number of steps per episode
        plt.figure(4)
        plt.clf()
        plt.title("Testing : Avg. number of steps taken per episode")
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg number of steps")
        plt.errorbar(train_nepisodes, test_steps_mean, yerr=test_steps_std, color="navy", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "test_steps_{}.png".format(len(self.test_steps)))
        else:
            plt.show()

        # training : avg actor loss over time
        plt.figure(5)
        plt.clf()
        plt.title("Avg. Actor Training Loss over {} episodes".format(train_nepisodes[-1]))
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg. Loss")
        # plt.plot(train_losses_mean, color="crimson")
        plt.errorbar(train_nepisodes, train_losses_mean, yerr=train_losses_std, color="tomato", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "train_loss_actor_{}.png".format(len(self.test_rewards)))
        else:
            plt.show()

        # training : avg critic loss over time
        plt.figure(5)
        plt.clf()
        plt.title("Avg. Critic Training Loss over {} episodes".format(train_nepisodes[-1]))
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg. Loss")
        # plt.plot(train_losses_mean, color="crimson")
        plt.errorbar(train_nepisodes, train_losses_mean, yerr=train_losses_std, color="tomato", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "train_loss_critic_{}.png".format(len(self.test_rewards)))
        else:
            plt.show()

        # training : combined avg loss over time
        plt.figure(5)
        plt.clf()
        plt.title("Avg. AC Combined Training Loss over {} episodes".format(train_nepisodes[-1]))
        plt.xlabel("Number of training episodes")
        plt.ylabel("Avg. Loss")
        # plt.plot(train_losses_mean, color="crimson")
        plt.errorbar(train_nepisodes, train_losses_mean, yerr=train_losses_std, color="crimson", uplims=True, lolims=True)
        if save :
            plt.savefig(self.expt_name + "train_loss_{}.png".format(len(self.test_rewards)))
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Using A2C for solving LunarLander")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor (default = 0.99)")
    parser.add_argument("--lr_actor", type=float, default=5e-4, help="learning rate actor (default = 1e-2)")
    parser.add_argument("--lr_critic", type=float, default=5e-4, help="learning rate critic (default = 1e-2)")
    parser.add_argument("--num_episodes", type=int, default=50000, help="number of episodes")
    parser.add_argument("--test_episodes", type=int, default=100, help="number of episodes to test on")
    #parser.add_argument("--num_steps", type=int, default=50, help="number of steps to run per episode")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--save_freq", type=int, default=1e4, help="checkpoint frequency for saving models")
    parser.add_argument("--test_freq", type=int, default=500, help="frequency for testing policy")
    parser.add_argument("--save_path", type=str, default="models/a2c_50_5/", help="path for saving models")
    parser.add_argument("--expt_name", type=str, default="plots/a2c_50_5/", help="expt name for saving results")
    parser.add_argument("--N_steps", type=int, default=100, help="N-step for the trace")

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.expt_name):
        os.mkdir(args.expt_name)

    # create the environment
    env = gym.make("LunarLander-v2")
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    # plt.ion()

    # A2C agent
    agent = A2C(env, args)
    agent.train(args.num_episodes)
    # agent.test()

    env.close()

if __name__ == "__main__":
    main()
