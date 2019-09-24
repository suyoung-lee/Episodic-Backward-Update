#! /usr/bin/env python
#__author__ = Nathan Sprague: NatureDQN Lasagne/Theano-based implementation
#__author__ = Shibi He: Optimality Tightening implementation
__author__ = 'suyounglee' #: Episodic Backward Update

"""
DQN agents
"""
import time
import os
import logging
import numpy as np
import cPickle

import ale_data_set
import sys
sys.setrecursionlimit(10000)

recording_size = 0

class OptimalityTightening(object):
    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size, exp_pref, update_frequency,
                 replay_start_size, rng, transitions_sequence_length, transition_range, penalty_method,
                 weight_min, weight_max, weight_decay_length, beta, two_train=False, late2=True, close2=True, verbose=False,
                 double=False, save_pkl=True):
        self.double_dqn = double
        self.network = q_network
        self.num_actions = q_network.num_actions
        self.epsilon_start = epsilon_start
        self.update_frequency = update_frequency
        self.beta = beta

        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.exp_dir = exp_pref + '_' + str(weight_max) + '_' + str(weight_min)
        if late2:
            self.exp_dir += '_l2'
        if close2:
            self.exp_dir += '_close2'
        else:
            self.exp_dir += '_len' + str(transitions_sequence_length) + '_r' + str(transition_range)
        if two_train:
            self.exp_dir += '_TTR'

        self.replay_start_size = replay_start_size
        self.rng = rng
        self.transition_len = transitions_sequence_length
        self.two_train = two_train
        self.verbose = verbose
        if verbose > 0:
            print "Using verbose", verbose
            self.exp_dir += '_vb' + str(verbose)

        self.phi_length = self.network.num_frames
        self.image_width = self.network.input_width
        self.image_height = self.network.input_height
        self.penalty_method = penalty_method
        self.batch_size = self.network.batch_size
        self.discount = self.network.discount
        self.transition_range = transition_range
        self.late2 = late2
        self.close2 = close2
        self.same_update = False
        self.save_pkl = save_pkl

        self.start_index = 0
        self.terminal_index = None

        self.weight_max = weight_max
        self.weight_min = weight_min
        self.weight = self.weight_max
        self.weight_decay_length = weight_decay_length
        self.weight_decay = (self.weight_max - self.weight_min) / self.weight_decay_length

        self.batchnum = 0
        self.epi_len = 0
        self.batch_count = 0
        self.epi_state = None
        self.epi_actions = None
        self.epi_rewards  = None
        self.epi_terminals = None
        self.Q_tilde = None
        self.y_  = None

        try:
            os.stat(self.exp_dir)
        except OSError:
            os.makedirs(self.exp_dir)

        self.data_set = ale_data_set.DataSet(width=self.image_width,
                                             height=self.image_height,
                                             rng=rng,
                                             max_steps=self.replay_memory_size,
                                             phi_length=self.phi_length,
                                             discount=self.discount,
                                             batch_size=self.batch_size,
                                             transitions_len=self.transition_len)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=self.image_width,
                                                  height=self.image_height,
                                                  rng=rng,
                                                  max_steps=self.phi_length * 2,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self._open_results_file()
        self._open_learning_file()
        self._open_recording_file()

        self.step_counter = 0
        self.episode_reward = 0
        self.start_time = None
        self.loss_averages = None
        self.total_reward = 0

        self.episode_counter = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_img = None
        self.last_action = None

        # Exponential moving average of runtime performance.
        self.steps_sec_ema = 0.
        self.program_start_time = None
        self.last_count_time = None
        self.epoch_time = None
        self.total_time = None

    def time_count_start(self):
        self.last_count_time = self.program_start_time = time.time()

    def _open_results_file(self):
        logging.info("OPENING " + self.exp_dir + '/results.csv')
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write('epoch,num_episodes,total_reward,reward_per_epoch,mean_q, epoch time, total time\n')
        self.results_file.flush()

    def _open_learning_file(self):
        self.learning_file = open(self.exp_dir + '/learning.csv', 'w', 0)
        self.learning_file.write('mean_loss,epsilon\n')
        self.learning_file.flush()

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{},{},{}\n".format(epoch, num_episodes,
                                              self.total_reward, self.total_reward / float(num_episodes),
                                              holdout_sum, self.epoch_time, self.total_time)
        print('Epoch %d average eval score: %f' % (epoch, self.total_reward / float(num_episodes)))
        self.last_count_time = time.time()
        self.results_file.write(out)
        self.results_file.flush()

    def _update_learning_file(self):
        out = "{},{}\n".format(np.mean(self.loss_averages),
                               self.epsilon)
        self.learning_file.write(out)
        self.learning_file.flush()

    def _open_recording_file(self):
        self.recording_tot = 0
        self.recording_file = open(self.exp_dir + '/recording.csv', 'w', 0)
        self.recording_file.write('nn_output, q_return, history_return, loss')
        self.recording_file.write('\n')
        self.recording_file.flush()

    def _update_recording_file(self, nn_output, q_return, history_return, loss):
        if self.recording_tot > recording_size:
            return
        self.recording_tot += 1
        out = "{},{},{},{}".format(nn_output, q_return, history_return, loss)
        self.recording_file.write(out)
        self.recording_file.write('\n')
        self.recording_file.flush()

    def start_episode(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        return_action = self.rng.randint(0, self.num_actions)

        self.last_action = return_action

        self.last_img = observation

        return return_action

    def _choose_action(self, data_set, epsilon, cur_img, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_img, self.last_action, reward, False, start_index=self.start_index)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(cur_img)
            action = self.network.choose_action(phi, epsilon)
        else:
            action = self.rng.randint(0, self.num_actions)

        return action

    def _do_training(self):
        # if all minibatches of previous episode is updated,
        # sample a new episode to create a new temporary target Q-table, Q_tilde
        if self.batchnum == self.batch_count:
            self.Q_tilde = np.array([]) # temporary target Q-table of next states S'
            self.epi_state, self.epi_actions, self.epi_rewards, self.batchnum, self.epi_terminals = self.data_set.random_episode() #sample a new episode
            self.epi_len = self.batchnum * self.batch_size			

            for i in range(self.batchnum):
                self.Q_tilde = np.append(self.Q_tilde, self.network.q_target_vals(self.epi_state[self.batch_size * i:self.batch_size * (i+1)]))
            self.Q_tilde = np.reshape(self.Q_tilde, (self.batchnum * self.batch_size, self.num_actions))
            self.Q_tilde = np.roll(self.Q_tilde, self.num_actions) # Q(S2) becomes the first column
            for i in range(self.epi_len):
                if self.epi_terminals[i]:
                    self.Q_tilde[i,:] = [0]*self.num_actions

            self.y_ = np.zeros(self.epi_len,dtype=np.float32) #target value

            for i in range(0, self.epi_len):
                if i < self.epi_len - 1:
                    # The last minibatch stores some redundant transitions of the second episode to fill a minibatch,
                    # so a terminal most likely occurs before self.epi_len
                    if self.epi_terminals[i]:
                        self.y_[i] = self.epi_rewards[i]
                        self.Q_tilde[i+1,self.epi_actions[i]] = self.beta * self.y_ [i] + (1-self.beta)*self.Q_tilde[i+1,self.epi_actions[i]]
                    elif self.epi_terminals[i+1]:
                        self.y_[i] = self.epi_rewards[i] + self.discount * np.max(self.Q_tilde[i])
                        self.Q_tilde[i+1,:] = 0
                    else:
                        self.y_[i] = self.epi_rewards[i] + self.discount * np.max(self.Q_tilde[i])
                        self.Q_tilde[i+1, self.epi_actions[i]] = self.beta * self.y_ [i] + (1-self.beta) * self.Q_tilde[i+1,self.epi_actions[i]]
                if i == self.epi_len - 1: #Most likely to be a transition of a redundant episode
                    if self.epi_terminals[i]:
                        self.y_[i] = self.epi_rewards[i]
                    else:
                        self.y_[i] = self.epi_rewards[i] + self.discount * np.max(self.Q_tilde[i])

            self.batch_count = 1

            return self.network.train(self.epi_state[0:self.batch_size],\
                                      np.reshape(self.epi_actions[0:self.batch_size],(self.batch_size,1)),\
                                      np.reshape(self.y_[0:self.batch_size],(self.batch_size,1)))

        # if an episode is still being updated, use the next minibatch of the already generated target value.
        else:
            self.batch_count += 1
            return self.network.train(self.epi_state[(self.batch_count-1)*self.batch_size:self.batch_count*self.batch_size],\
                                      np.reshape(self.epi_actions[(self.batch_count-1)*self.batch_size:self.batch_count*self.batch_size],(self.batch_size,1)),\
                                      np.reshape(self.y_[(self.batch_count-1)*self.batch_size:self.batch_count*self.batch_size],(self.batch_size,1)))

    def step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           An integer action.

        """

        self.step_counter += 1
        self.episode_reward += reward

        # TESTING---------------------------
        if self.testing:
            action = self._choose_action(self.test_data_set, 0.05,
                                         observation, np.clip(reward, -1, 1))

        # NOT TESTING---------------------------
        else:
            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)
                self.weight = max(self.weight_min,
                                  self.weight - self.weight_decay)

                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))

                if self.step_counter % self.update_frequency == 0:
                    loss = self._do_training()
                    self.batch_counter += 1
                    self.loss_averages.append(loss)

            else:  # Still gathering initial random data...
                action = self._choose_action(self.data_set, self.epsilon,
                                             observation,
                                             np.clip(reward, -1, 1))
        self.last_action = action
        self.last_img = observation

        return action

    def end_episode(self, reward, terminal=True):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.
           terminal    - Whether the episode ended intrinsically
                         (ie we didn't run out of steps)
        Returns:
            None
        """

        self.episode_reward += reward
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            # If we run out of time, only count the last episode if
            # it was the only episode.
            if terminal or self.episode_counter == 0:
                self.episode_counter += 1
                self.total_reward += self.episode_reward
        else:
            # Store the latest sample.
            self.data_set.add_sample(self.last_img,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True, start_index=self.start_index)
            """update"""
            if terminal:
                q_return = 0.
            else:
                phi = self.data_set.phi(self.last_img)
                q_return = np.mean(self.network.q_vals(phi))
            # last_q_return = -1.0
            self.start_index = self.data_set.top
            self.terminal_index = index = (self.start_index-1) % self.data_set.max_steps
            while True:
                q_return = q_return * self.network.discount + self.data_set.rewards[index]
                self.data_set.return_value[index] = q_return
                self.data_set.terminal_index[index] = self.terminal_index
                index = (index-1) % self.data_set.max_steps
                if self.data_set.terminal[index] or index == self.data_set.bottom:
                    break

            rho = 0.98
            self.steps_sec_ema *= rho
            self.steps_sec_ema += (1. - rho) * (self.step_counter/total_time)

            """logging.info("steps/second: {:.2f}, avg: {:.2f}".format(
                self.step_counter/total_time, self.steps_sec_ema))"""

            if self.batch_counter > 0:
                self._update_learning_file()
                """logging.info("average loss: {:.4f}".format(\
                                np.mean(self.loss_averages)))"""

    def finish_epoch(self, epoch):
        if self.save_pkl:
            net_file = open(self.exp_dir + '/network_file_' + str(epoch) + '.pkl', 'w')
            cPickle.dump(self.network, net_file, -1)
            net_file.close()
        this_time = time.time()
        self.total_time = this_time-self.program_start_time
        self.epoch_time = this_time-self.last_count_time

    def start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0

    def finish_testing(self, epoch):
        self.testing = False
        # self.beta = self.beta - 0.025
        holdout_size = 3200

        if self.holdout_data is None and len(self.data_set) > holdout_size:
            imgs = self.data_set.random_imgs(holdout_size)
            self.holdout_data = imgs[:, :self.phi_length]

        holdout_sum = 0
        if self.holdout_data is not None:
            for i in range(holdout_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i]))

        self._update_results_file(epoch, self.episode_counter,
                                  holdout_sum / holdout_size)
        self.total_reward = 0
        self.episode_counter = 0

