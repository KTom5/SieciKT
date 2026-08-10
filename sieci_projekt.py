import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import networkx as nx
import matplotlib.pyplot as plt
import time


def adj_from_obs(n, observation):
    W = np.zeros([n, n], dtype=np.int8)
    W[np.triu_indices(n, k=1)] = observation

    return np.maximum(W, W.T)

def reset(obs_new, autlen):
    obs_new[:, 0, :] = 0
    obs_new[:, 0, autlen] = 1

def step(position, next_actions, obs_new, act_new, batch_size, autlen, rng, act_rndness):
    rnd_size = round(act_rndness*batch_size)
    next_actions[rng.integers(0, batch_size, size=rnd_size)] = rng.choice(2, size=rnd_size)

    act_new[:, position] = next_actions
    obs_new[:, position+1, :] = obs_new[:, position, :]
    obs_new[:, position+1, position] = next_actions
    obs_new[:, position+1, autlen+position] = 0

    if (position<autlen-1):
        obs_new[:, position+1, autlen+position+1] = 1

def new_generation(model, obs_new, act_new, rew_new, batch_size, autlen, rng, act_rndness, n, compute_reward):

    reset(obs_new, autlen)

    for position in range(autlen):
        obs_t = torch.FloatTensor(obs_new[:, position, :])
        probs_t = nn.Softmax(dim=1)(model(obs_t))
        next_probs = probs_t.data.numpy()

        next_actions = np.array([rng.choice(2, p=probs) for probs in next_probs])

        step(position, next_actions, obs_new, act_new, batch_size, autlen, rng, act_rndness)

    for graph in range(batch_size):
        rew_new[graph] = compute_reward(n, adj_from_obs(n, obs_new[graph, autlen, 0:autlen]))


def train(compute_reward,
          n=20,
          batch_size=200,
          num_generations=1000,
          percent_learn=90,
          percent_survive=97.5,
          neurons=[72,12],
          learning_rate=0.003,
          act_rndness_init=0.005,
          act_rndness_wait=10,
          act_rndness_mult=1.1,
          act_rndness_max=0.025,
          verbose=True,
          output_best_graph_rate=25
          ):

    autlen = (n*(n-1))//2
    obslen = 2*autlen

    model = nn.Sequential(
        nn.Linear(obslen, neurons[0]),
        nn.ReLU(),
        nn.Dropout(0.2)
    )

    for i in range(1, len(neurons)):
        model = model.append(nn.Linear(neurons[i - 1], neurons[i]))
        model = model.append(nn.ReLU())
        model = model.append(nn.Dropout(0.2))

    model = model.append(nn.Linear(neurons[-1], 2))

    objective = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    rng = np.random.default_rng()

    obs_new = np.zeros([batch_size, autlen + 1, obslen], dtype=np.bool_)
    act_new = np.zeros([batch_size, autlen], dtype=np.bool_)
    rew_new = np.zeros([batch_size], dtype=np.float64)

    obs_survive = None
    act_survive = None
    rew_survive = None

    act_rndness = act_rndness_init
    old_max_reward = None
    old_max_gen = None

    writer = SummaryWriter()

    for gen in range(num_generations):
        tic = time.perf_counter()

        new_generation(model, obs_new, act_new, rew_new, batch_size, autlen, rng, act_rndness, n, compute_reward)

        if gen==0:
            obs_full = obs_new
            act_full = act_new
            rew_full = rew_new
        else:
            obs_full = np.concatenate((obs_new, obs_survive), axis=0)
            act_full = np.concatenate((act_new, act_survive), axis=0)
            rew_full = np.concatenate((rew_new, rew_survive), axis=0)

        cutoff_percent = percent_learn
        while cutoff_percent<99.9:
            lrn_reward = np.percentile(rew_full, cutoff_percent)
            ind_learn = np.where(rew_full>=lrn_reward)[0]
            if ind_learn.size <= batch_size//2:
                break
            else:
                cutoff_percent = (100+cutoff_percent)/2

        ind_learn = ind_learn[:batch_size//2]

        num_learn_pairs = ind_learn.size * autlen
        obs_learn = obs_full[ind_learn, 0:autlen, :].reshape(num_learn_pairs, obslen)
        act_learn = act_full[ind_learn, :].reshape(num_learn_pairs)

        optimizer.zero_grad()
        act_learn_scores = model(torch.FloatTensor(obs_learn))
        loss = objective(act_learn_scores, torch.LongTensor(act_learn))

        loss.backward()
        optimizer.step()

        cutoff_percent = percent_survive
        while cutoff_percent<99.9:
            srv_reward = np.percentile(rew_full, cutoff_percent)
            ind_survive = np.where(rew_full>=srv_reward)[0]
            if ind_survive.size <= batch_size:
                break
            else:
                cutoff_percent = (100+cutoff_percent)/2

        ind_survive = ind_survive[:batch_size]
        num_survive = ind_survive.size

        obs_survive = obs_full[ind_survive, :, :].copy().reshape(num_survive, autlen+1, obslen)
        act_survive = act_full[ind_survive, :].copy().reshape(num_survive, autlen)
        rew_survive = rew_full[ind_survive].copy().reshape(num_survive)

        max_reward = np.max(rew_full)

        if gen==0:
            old_max_reward = max_reward
            old_max_gen = 0
        elif max_reward > old_max_reward + 0.0001:
            act_rndness = act_rndness_init
            old_max_reward = max_reward
            old_max_gen = gen
        elif gen - old_max_gen >= act_rndness_wait:
            act_rndness = min(act_rndness * act_rndness_mult, act_rndness_max)
            old_max_gen = gen

        toc = time.perf_counter()

        if verbose:
            print(f'gen={gen}, rew_max={max_reward:.8f}, rew_surv={srv_reward:.8f}, rew_learn={lrn_reward:.8f}, time={toc-tic:.4f}, act_rndness={act_rndness:.4f}')
        writer.add_scalar('rew_max', max_reward, gen)
        writer.add_scalar('rew_surv', srv_reward, gen)
        writer.add_scalar('rew_learn', lrn_reward, gen)
        writer.flush()

        if gen % output_best_graph_rate == 0:
            if gen>0:
                plt.close('all')

            ind_maximum = np.argmax(rew_full)
            max_A = adj_from_obs(n, obs_full[ind_maximum, autlen, 0:autlen])
            gnx = nx.from_numpy_array(max_A)

            plt.figure(num=1, figsize=(4,4), dpi=300)

            pos=nx.kamada_kawai_layout(gnx)
            nx.draw_kamada_kawai(gnx, node_size=80, with_labels=True)

            plt.ioff()
            plt.show()

            writer.add_figure('best graph', plt.figure(num=1), gen)
            # fig_name = f'fig-{gen}.png'
            # plt.savefig(fig_name, transparent=True)

    plt.close('all')
    writer.close()

    ind_maximum = np.argmax(rew_full)
    max_A = adj_from_obs(n, obs_full[ind_maximum, autlen, 0:autlen])
    return max_reward, max_A

def eigen_vector(v, A):
    for i in range(0, 1000):
        v = A@v
        if sum(v) == 0:
            return v
        v = v/sum(v)
    return v

def reward_function(n, A):
    v = np.zeros(n)
    v[0] = 1
    eigen = eigen_vector(v, np.eye(n)+A)
    not_connected=int(np.any(eigen==0))
    return float(np.max(eigen))-not_connected

for n in range(20, 21):
  r, A = train(compute_reward=reward_function,
              n=n,
              num_generations=10000,
              #output_best_graph_rate=5
                )