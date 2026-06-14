# DQN Atari Pong — Gotchas

## CNN on T4: ~50 steps/sec (very slow)

Atari pixel-based DQN processes 84×84×4 frames through a CNN. On T4, this achieves ~50 env steps per second (including training updates every 4 frames). In 8 minutes, you get ~24,000 steps — less than half of the 50K epsilon decay schedule. Use Kaggle P100 (~120 steps/sec) for pixel-based Atari training.

## System deps required for ALE

`libcairo2-dev` and `libpango1.0-dev` must be installed via `apt-get` before `ale-py` can render frames. The launch.py handles this, but for local runs you need them pre-installed.

## Frame preprocessing must use headless OpenCV

`opencv-python-headless` (not `opencv-python`) to avoid pulling in GUI dependencies on the headless Colab VM. Using the non-headless version causes `libGL.so.1` errors.

## Dueling architecture matters for Pong

Without the dueling head (separate V and A streams), DQN learns ~20-30% slower on Pong. The value stream learns that most states are neutral (~0 value), while the advantage stream learns which actions are good at each state.

## Smooth L1 (Huber) loss > MSE

MSE penalizes large TD errors quadratically, causing instability when the target network is stale. Smooth L1 (Huber) caps the gradient for large errors, making training more robust to Q-value overestimation.

## Epsilon decay: 50K steps to 0.01

Pong needs only ~20K steps to start showing positive returns. By 50K steps, epsilon reaches 0.01 (1% random actions) — the agent is exploiting a decent policy. The solved threshold (avg100 > 18) typically triggers around step 80K-120K.

## Pong is the "hello world" of Atari DQN

If Pong doesn't converge (avg100 > 18 within 500 episodes), something is fundamentally wrong with the DQN implementation. Common bugs: forgetting to detach target Q-values, updating target network too frequently, or forgetting to normalize frame pixels to [0,1].

## Checkpoint resume works but epsilon state lost

The checkpoint saves model + optimizer + metrics, but epsilon is computed from step count. On resume, epsilon continues decaying from the step count, which is correct (not from a saved epsilon value that might be stale).
