# Episodic-Backward-Update
Lasagne/Theano-based implementation of ["Sample-Efficient Deep Reinforcement Learning via Episodic Backward Update"](https://arxiv.org/abs/1805.12375), NeurIPS 2019.

Episodic Backward Update (EBU) with a constant diffusion factor for the ATARI environment will be uploaded.

# Dependencies
* Numpy
* Scipy
* Pillow
* Matplotlib
* Lasagne
* ALE

Our implementation is based on Shibi He's [implementation](https://github.com/ShibiHe/Q-Optimality-Tightening) of [Optimality Tightening](https://arxiv.org/abs/1611.01606) which is based on Nathan Sprague's implementation of [deep Q RL](https://github.com/spragunr/deep_q_rl).

Please refer to [https://github.com/spragunr/deep_q_rl](https://github.com/spragunr/deep_q_rl) for installing the dependencies.

# Major changes from the [deep Q RL](https://github.com/spragunr/deep_q_rl) implementation
* ale_agents.py / _do_training : generate temporary target Q table and update
* ale_data_set.py / random_episode : sample an episode instead of a minibatch of transitions
* ale_experiment.py / run, run_epoch, run_episode : fixed to apply the Nature DQN setting so that each episode is played at most 4500 steps (18,000 frames or 5 minutes).
* launcher.py contains hyperparameter beta for the diffusion factor

# Running
You can train an EBU agent with a constant diffusion factor 0.5 in breakout using random seed 12 on gpu0 as follows.
`THEANO_FLAGS='device=gpu0, allow_gc=False' python code/run_EBU.py -r 'breakout' --Seed 12 --beta 0.5`

By default, it returns the test scores at every 62,500 steps for 40 times (62,500 steps x 4 frames/step x 40 = 10M frames).

You may modify the STEPS_PER_EPOCH and EPOCHS parameter in run_EBU.py to change the total number of training steps and the frequency of evaluation.
