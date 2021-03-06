import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from torch.utils.tensorboard import SummaryWriter

from env.grid_env import GridEnv
from agent.agent_policy_gradient import Agent
import pygame
from env.config import *
from utils.pygame_painter import Painter

params = {
    'name': 'policy_gradient',
    # field
    'w': FIELD.w,
    'h': FIELD.h,
    'start_direction': MOVEMENTS[np.random.randint(0, len(ACTION))],
    'start_pos': np.random.randint(0, min(FIELD.h, FIELD.w), size=(2,)),

    'field_data': FIELD.data,
    # model params
    'update_every': 10,
    'eps_start': 0.15,  # Default/starting value of eps
    'eps_decay': 0.99999,  # Epsilon decay rate
    'eps_min': 0.15,  # Minimum epsilon
    'gamma': 0.98,
    'buffer_size': 200000,
    'batch_size': 128,
    'action_size': len(ACTION),

    'is_double': False,
    'is_priority_buffer': True,

    # grid params
    'max_step': 200,

    # train params
    'visualise': False,
    'is_normalize': True,
    'num_episodes': 5000000,
    'scale': 15,

    # folder params

    # output
    'output_folder': "output_policy_gradient",
    'log_folder': 'log',
    'model_folder': 'model',
    'memory_config_dir': "memory_config"

}

params['log_folder'] = os.path.join(params['output_folder'], params['log_folder'])
params['model_folder'] = os.path.join(params['output_folder'], params['model_folder'])
if not os.path.exists(params['log_folder']):
    os.makedirs(params['log_folder'])
if not os.path.exists(params['model_folder']):
    os.makedirs(params['model_folder'])
painter = Painter(params) if params['visualise'] else None
grid_env = GridEnv(params, painter)

model_path = os.path.join(params['output_folder'], "model", "Agent_dqn_state_dict_1600.mdl")
agent_pg = Agent(params, painter)

writer = SummaryWriter(log_dir=params['log_folder'])

all_mean_rewards = []
all_mean_losses = []
time_step = 0
for i_episode in range(params['num_episodes']):
    observed_map, robot_pose = grid_env.reset()
    done = False
    rewards = []
    while not done:
        action = agent_pg.act(observed_map, robot_pose)
        observed_map_next, robot_pose_next, reward, done = grid_env.step(action)
        agent_pg.step(state=[observed_map, robot_pose], action=action, reward=reward,
                      next_state=[observed_map_next, robot_pose_next], done=done)
        # 转到下一个状态
        observed_map = observed_map_next.copy()
        robot_pose = robot_pose_next.copy()

        time_step += 1

        if params['visualise']:
            painter.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        rewards.append(reward)

        if done:
            loss = agent_pg.learn()
            all_mean_losses.append(loss)
            writer.add_scalar('train/loss_per_episode', loss, time_step)

            if (i_episode + 1) % 200 == 0:
                # plt.cla()
                model_save_path = os.path.join(params['model_folder'], "Agent_dqn_state_dict_%d.mdl" % (i_episode + 1))
                agent_pg.store_model(model_save_path)

            all_mean_rewards.append(np.mean(rewards))
            print()
            print(
                "i episode:{}; mean reward:{}; num_found_free_cell:{}/{};num_found_targets:{}/{}; num_found_occupied:{}/{}"
                    .format(i_episode, np.mean(rewards), grid_env.count_found_free, grid_env.count_free,
                            grid_env.count_found_target, grid_env.count_target,
                            grid_env.count_found_occupied, grid_env.count_occupied))
            writer.add_scalar('train/losses_smoothed', np.mean(all_mean_losses[max(0, i_episode - 200):]), i_episode)
            writer.add_scalar('train/rewards_smoothed', np.mean(all_mean_rewards[max(0, i_episode - 200):]), i_episode)
            writer.add_scalar("train/reward_per_episode", np.mean(rewards), i_episode)
            writer.add_scalar('train/num_found_free_cell', grid_env.count_found_free, i_episode)
            writer.add_scalar('train/num_found_targets', grid_env.count_found_target, i_episode)
            writer.add_scalar('train/num_found_total_cell', grid_env.count_found_free + grid_env.count_found_target,
                              i_episode)

print('Complete')
