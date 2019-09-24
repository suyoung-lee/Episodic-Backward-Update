# Episodic-Backward-Update
Lasagne/Theano-based implementation of ["Sample-Efficient Deep Reinforcement Learning via Episodic Backward Update"](https://arxiv.org/abs/1805.12375), NeurIPS 2019.

EBU with a constant diffusion factor for the ATARI environment will be uploaded.

# Dependencies
* Numpy
* Scipy
* Pillow
* Matplotlib
* Lasagne
* ALE

Our implementation is based on Shibi He's [implementation](https://github.com/ShibiHe/Q-Optimality-Tightening) of [Optimality Tightening](https://arxiv.org/abs/1611.01606) which is based on Nathan Sprague's implementation of [deep Q RL](https://github.com/spragunr/deep_q_rl).

Please refer to [https://github.com/spragunr/deep_q_rl](https://github.com/spragunr/deep_q_rl) for installing the dependencies.

# Running
You can train an EBU agent with a constant diffusion factor 0.5 in breakout using random seed 12 on gpu0 as follows.
`THEANO_FLAGS='device=gpu0, allow_gc=False' python code/run_EBU.py -r 'breakout' --Seed 12 --beta 0.5`
By default, it returns the test scores at every 62500 steps for 40 times (62500 steps x 4 frames/step x 40 = 10M frames)
You may modify the STEPS_PER_EPOCH and EPOCHS parameter in run_EBU.py to change the number of steps and frequency of evaluation.
