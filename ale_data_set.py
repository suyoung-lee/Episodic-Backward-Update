#! /usr/bin/env python
#__author__ = Nathan Sprague: NatureDQN Lasagne/Theano-based implementation
#__author__ = Shibi He: Optimality Tightening implementation
__author__ = 'suyounglee' #: Episodic Backward Update

import numpy as np
import time
import theano

floatX = theano.config.floatX


class DataSet(object):
    def __init__(self, width, height, rng, max_steps=1000000, phi_length=4, discount=0.99, batch_size=32,
                 transitions_len=4):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.phi_length = phi_length
        self.rng = rng
        self.discount = discount
        self.discount_table = np.power(self.discount, np.arange(30))
        self.batch_size = batch_size

        self.imgs = np.zeros((max_steps, height, width), dtype='uint8')
        self.actions = np.zeros(max_steps, dtype='int32')
        self.rewards = np.zeros(max_steps, dtype=floatX)
        self.return_value = np.zeros(max_steps, dtype=floatX)
        self.terminal = np.zeros(max_steps, dtype='bool')
        self.terminal_index = np.zeros(max_steps, dtype='int32')
        self.start_index = np.zeros(max_steps, dtype='int32')

        self.bottom = 0
        self.top = 0
        self.size = 0

        self.center_imgs = np.zeros((batch_size,
                                     self.phi_length,
                                     self.height,
                                     self.width),
                                    dtype='uint8')
        self.forward_imgs = np.zeros((batch_size,
                                      transitions_len,
                                      self.phi_length,
                                      self.height,
                                      self.width),
                                     dtype='uint8')
        self.backward_imgs = np.zeros((batch_size,
                                       transitions_len,
                                       self.phi_length,
                                       self.height,
                                       self.width),
                                      dtype='uint8')
        self.center_positions = np.zeros((batch_size, 1), dtype='int32')
        self.forward_positions = np.zeros((batch_size, transitions_len), dtype='int32')
        self.backward_positions = np.zeros((batch_size, transitions_len), dtype='int32')

        self.center_actions = np.zeros((batch_size, 1), dtype='int32')
        self.backward_actions = np.zeros((batch_size, transitions_len), dtype='int32')

        self.center_terminals = np.zeros((batch_size, 1), dtype='bool')
        self.center_rewards = np.zeros((batch_size, 1), dtype=floatX)

        self.center_return_values = np.zeros((batch_size, 1), dtype=floatX)
        self.forward_return_values = np.zeros((batch_size, transitions_len), dtype=floatX)
        self.backward_return_values = np.zeros((batch_size, transitions_len), dtype=floatX)

        self.forward_discounts = np.zeros((batch_size, transitions_len), dtype=floatX)
        self.backward_discounts = np.zeros((batch_size, transitions_len), dtype=floatX)

    def add_sample(self, img, action, reward, terminal, return_value=0.0, start_index=-1):

        self.imgs[self.top] = img
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal
        self.return_value[self.top] = return_value
        self.start_index[self.top] = start_index
        self.terminal_index[self.top] = -1

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        return self.size

    def last_phi(self):
        """Return the most recent phi (sequence of image frames)."""
        indexes = np.arange(self.top - self.phi_length, self.top)
        return self.imgs.take(indexes, axis=0, mode='wrap')

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.height, self.width), dtype='uint8')
        phi[0:self.phi_length - 1] = self.imgs.take(indexes,
                                                    axis=0,
                                                    mode='wrap')
        phi[-1] = img
        return phi

    # randomly sample an episode and split into slices of the size equal to the batchsize
    def random_episode(self):
        terminal_array = np.where(self.terminal==True)[0]

        batchnum = 0
        while batchnum == 0:
            # exclude some early and final episodes from sampling due to indexing issues,
            # sample two episodes (ind1 for main, and ind2 for the remaining steps to make multiple of 32)
            ind = self.rng.choice(range(5,len(terminal_array)-3), 2)
            ind1 = ind[0]
            ind2 = ind[1]

            indice_array = range(terminal_array[ind1],terminal_array[ind1-1]+3,-1)		
            epi_len = len(indice_array)		
            batchnum = int(np.ceil(epi_len/float(self.batch_size)))

        remainindex = int(batchnum * self.batch_size - epi_len)

        # Normally an episode does not have steps=multiple of 32.
        # Fill last minibatch with redundant steps from another episode
        indice_array= np.append(indice_array, range(terminal_array[ind2], terminal_array[ind2]-remainindex, -1))
        indice_array = indice_array.astype(int)

        epi_len = len(indice_array)
        rewards = [self.rewards[i] for i in indice_array]
        actions = [self.actions[i] for i in indice_array]
        terminals = [self.terminal[i] for i in indice_array]

        state1 = np.zeros((epi_len, self.phi_length, self.height, self.width), dtype='uint8')

        count = 0
        for i in indice_array:
            state1[count,:,:,:] = np.reshape(self.imgs[i-3:i+1],(1,self.phi_length,self.height, self.width))
            count += 1

        return state1, actions, rewards, batchnum, terminals

    def random_imgs(self, size):
        imgs = np.zeros((size,
                         self.phi_length + 1,
                         self.height,
                         self.width),
                        dtype='uint8')

        count = 0
        while count < size:
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)
            all_indices = np.arange(index, index + self.phi_length + 1)
            end_index = index + self.phi_length - 1
            if np.any(self.terminal.take(all_indices[0:-2], mode='wrap')):
                continue
            imgs[count] = self.imgs.take(all_indices, axis=0, mode='wrap')
            count += 1
        return imgs
