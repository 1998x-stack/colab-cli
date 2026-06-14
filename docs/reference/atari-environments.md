# Atari 2600 Gymnasium Environments — Complete Reference

Data from gymnasium 1.3.0 + ale-py (Colab, 2026-06-14). 104 games, 2 observation modes (pixel & RAM).

## Quick Facts

- **104 games** registered as `ALE/GameName-v5`
- **All use Discrete action spaces** — 3 to 18 actions per game
- **Pixel obs**: (210, 160, 3) uint8 RGB for all games
- **RAM obs**: (128,) uint8 for all games — accessed via `gym.make("ALE/Pong-v5", obs_type="ram")`
- **No max_episode_steps** by default — use `TimeLimit` wrapper or set manually
- **No separate "ram" env IDs** — RAM is an `obs_type` parameter in v5, not a separate env

## Observation Modes

### Pixel mode (default)
```python
env = gym.make("ALE/Pong-v5")
# obs: Box(210, 160, 3), uint8, values 0-255
# RGB image of the Atari screen
```

### RAM mode
```python
env = gym.make("ALE/Pong-v5", obs_type="ram")
# obs: Box(128,), uint8, values 0-255
# Raw Atari 2600 RAM — 128 bytes
```
RAM mode is much lighter (128-dim vector vs 210×160×3 image), useful for MLP-based agents. The RAM contains game state (positions, scores, lives) but semantics vary per game.

## Standard Atari Wrappers

```python
import gymnasium as gym
import numpy as np
import cv2

def make_atari(env_id, frame_skip=4, screen_size=84, gray_scale=True):
    env = gym.make(env_id,
                   frameskip=1,          # we handle frame skip manually
                   repeat_action_probability=0.0,  # deterministic
                   full_action_space=False)  # use minimal action set

    # Frame skip + max-pool over last 2 frames (handles sprite flicker)
    env = gym.wrappers.AtariPreprocessing(
        env,
        screen_size=screen_size,
        grayscale_obs=gray_scale,
        frame_skip=frame_skip,
        noop_max=30,          # random no-ops at reset
    )

    # Stack last 4 frames (gives velocity information)
    env = gym.wrappers.FrameStackObservation(env, 4)

    return env

# Final obs shape for DQN/PPO: (4, 84, 84) — 4 stacked grayscale frames
```

## Action Space Distribution

Most games use 4, 6, or 18 actions:

| Action Count | Games | Examples |
|-------------|-------|----------|
| 3 | 3 | Freeway, Pong |
| 4 | 6 | Breakout, VideoPinball |
| 5 | 5 | Boxing, Tennis |
| 6 | 14 | SpaceInvaders, Qbert, MsPacman |
| 7 | 1 | (rare) |
| 8 | 3 | Pitfall, Pitfall2 |
| 9 | 8 | BeamRider, Enduro, KungFuMaster |
| 10 | 7 | BankHeist, NameThisGame |
| 12 | 1 | (rare) |
| 14 | 3 | Asteroids, WizardOfWor |
| 16 | 1 | (rare) |
| 18 | 52 | **Half of all games** — Seaquest, MontezumaRevenge, etc. |

**Always use `full_action_space=False`** unless you have a specific reason. It maps to the minimal viable action set (the numbers in the table above). `full_action_space=True` gives all 18 joystick combos for every game.

## Game Categories

### Paddle (3 games) — horizontal/vertical ball bouncing
Simple, few actions, good for quick DQN tests.

| Game | Actions | Notes |
|------|---------|-------|
| Pong | 3 | Classic. DQN solves in ~1M frames |
| Breakout | 4 | DQN played it first (Nature 2015). ~10M frames |
| VideoPinball | 4 | Pinball physics, faster than Breakout |

### Shooter (46 games) — largest category
Move + shoot in 2D space. Wide range of difficulty.

| Game | Actions | Notes |
|------|---------|-------|
| SpaceInvaders | 6 | Classic. DQN ~5M frames |
| Seaquest | 18 | Underwater, scrolling. 18 actions is overkill |
| BeamRider | 9 | Fast-paced, rail shooter |
| Asteroids | 14 | Floating movement, wrap-around |
| DemonAttack | 6 | DQN baseline ~2M frames |
| Phoenix | 6 | Shield mechanic |
| RiverRaid | 18 | Scrolling shooter, fuel management |
| ChopperCommand | 18 | Helicopter, rescue/shoot |
| Centipede | 6 | Trackball-style movement |
| Defender | 18 | Complex — radar, multiple threats |
| Gravitar | 18 | Physics-based, very hard |
| Zaxxon | 18 | Isometric scrolling shooter |
| BattleZone | 18 | 3D tank combat (vector graphics) |
| AirRaid | 4 | Simple scrolling shooter |
| Assault | 6 | Scrolling, tank-like |
| Carnival | 4 | Shooting gallery |
| Crossbow | 18 | Protect friends, shoot enemies |
| Darkchambers | 18 | Top-down dungeon shooter |
| FishingDerby | 8 | Competitive fishing |
| Galaxian | 6 | Space Invaders predecessor |
| Gopher | 4 | Protect carrots from gopher |
| Gyruss | 18 | Tube shooter, Tempest-like |
| IceHockey | 6 | 2v2 hockey |
| Jamesbond | 18 | Multi-stage, vehicle + foot |
| JourneyEscape | 6 | Avoid obstacles, reach ship |
| Kangaroo | 4 | Platformer-shooter hybrid |
| Krull | 6 | Multi-screen adventure |
| LaserGates | 4 | Tunnel defense |
| MarioBros | 18 | Two-player arcade, competitive |
| Millipede | 18 | Centipede successor |
| MissileCommand | 6 | Defend cities from missiles |
| NameThisGame | 10 | Underwater shooter, shark |
| Pooyan | 4 | Shoot wolves, rescue pigs |
| RoadRunner | 18 | Avoid coyote, eat birdseed |
| Robotank | 18 | First-person tank, radar |
| Skiing | 2 | Downhill slalom |
| Solaris | 18 | Space exploration, star map |
| StarGunner | 18 | Horizontal scroller |
| Tennis | 5 | Pong variant |
| TimePilot | 18 | Free-roaming 360° shooter |
| Trondead | 18 | Lightcycle-style |
| Turmoil | 6 | Shoot aliens, collect prizes |
| Tutankham | 18 | Tunnel exploration shooter |
| UpNDown | 6 | Car jumping, color matching |
| YarsRevenge | 6 | Shield + cannon mechanic |

### Platformer (6 games)

| Game | Actions | Notes |
|------|---------|-------|
| MontezumaRevenge | 18 | **Hardest Atari game**. Sparse reward, rooms, keys. Needs curiosity/exploration |
| Pitfall | 8 | Side-scrolling, avoid obstacles |
| Pitfall2 | 8 | Pitfall sequel, more complex |
| PrivateEye | 18 | Detective, multiple cases |
| DonkeyKong | 4 | Climb, avoid barrels |
| Frogger | 4 | Cross road + river |

### Racing (3 games)

| Game | Actions | Notes |
|------|---------|-------|
| Enduro | 9 | Day/night endurance racing |
| Freeway | 3 | Chicken cross road. Simple |
| KungFuMaster | 9 | Beat-em-up, side-scrolling |

### Puzzle (27 games)

| Game | Actions | Notes |
|------|---------|-------|
| Qbert | 6 | Hop on cubes, change colors |
| Tetris | 5 | Classic block stacking |
| Amidar | 18 | Fill rectangles, avoid enemies |
| Asterix | 18 | Collect objects, avoid Romans |
| Atlantis | 4 | Defend underwater city |
| Atlantis2 | 4 | Atlantis sequel |
| BankHeist | 10 | Rob banks, avoid cops |
| Berzerk | 18 | Maze shooter, talk to robots |
| Bowling | 5 | 10-pin bowling |
| Boxing | 5 | Punch-out style |
| Casino | 4 | Slot machine + blackjack |
| CrazyClimber | 18 | Climb building, avoid windows |
| DoubleDunk | 5 | 2v2 basketball |
| ElevatorAction | 18 | Spy, use elevators |
| Entombed | 18 | Maze, avoid zombies |
| Et | 4 | Infamous. Fall in pits, find parts |
| Kaboom | 4 | Bucket catch |
| KeystoneKapers | 18 | Catch thief in mall |
| KingKong | 18 | Climb, avoid barrels |
| MsPacman | 6 | Pac-Man with moving fruit |
| Pacman | 4 | Classic dot-eating |
| Surround | 4 | Snake-like, trap opponent |
| WizardOfWor | 14 | Maze shooter, 2-player |
| WordZapper | 6 | Word scramble |
| Zookeeper | 18 | Jump over animals, rescue girl |
| MrDo | 18 | Dig dug-style |
| Koolaid | 4 | Maze, collect ingredients |

### Adventure (3 games)

| Game | Actions | Notes |
|------|---------|-------|
| Adventure | 6 | Find keys, defeat dragons |
| HauntedHouse | 6 | Explore mansion, find urn |
| Venture | 18 | Explore rooms, collect treasure |

### Board/Card (6 games)

| Game | Actions | Notes |
|------|---------|-------|
| Backgammon | 3 | Backgammon vs AI |
| BasicMath | 6 | Arithmetic quiz |
| Blackjack | 4 | Blackjack |
| Checkers | 4 | (registered as VideoCheckers) |
| Othello | 18 | Reversi |
| VideoChess | 18 | Chess |

### Other (13 games — misc/sports/oddities)

| Game | Actions | Notes |
|------|---------|-------|
| Earthworld | 18 | Swordquest series, obscure |
| FlagCapture | 3 | Capture the flag |
| Frostbite | 18 | Build igloo, avoid polar bear |
| Hangman | 18 | Word-guessing |
| Hero | 18 | Rambo-style, rescue miners |
| HumanCannonball | 4 | Launch human, hit target |
| Klax | 18 | Catch colored tiles (puzzle) |
| LostLuggage | 18 | Catch falling luggage |
| MiniatureGolf | 4 | Mini golf |
| SirLancelot | 6 | Jousting, rescue damsel |
| SpaceWar | 18 | Gravity-based ship combat |
| Superman | 18 | Fly, catch criminals |
| TicTacToe3D | 18 | 4×4×4 3D tic-tac-toe |
| VideoCube | 18 | Rubik's cube |

## Difficulty Tiers

### Easy (solve in <5M frames with DQN)
Pong, Breakout, Freeway, Boxing, Bowling, VideoPinball, Atlantis, Carnival, FishingDerby, IceHockey, Tennis

### Medium (5-20M frames)
SpaceInvaders, BeamRider, Qbert, Seaquest, DemonAttack, Asterix, BankHeist, Enduro, MsPacman, NameThisGame, Phoenix, RoadRunner, Robotank, Skiing, StarGunner, TimePilot, Tutankham, UpNDown

### Hard (20-50M frames)
Asteroids, BattleZone, Berzerk, Centipede, ChopperCommand, CrazyClimber, Defender, ElevatorAction, Gopher, Gravitar, Jamesbond, Kangaroo, Krull, KungFuMaster, MontezumaRevenge, Pitfall, Pitfall2, PrivateEye, RiverRaid, Solaris, Venture, WizardOfWor, Zaxxon

### Extreme (50M+ or needs specialized algorithms)
MontezumaRevenge (exploration problem — needs curiosity/RND/Go-Explore), Pitfall (sparse rewards)

## Full Game List (104)

Sorted alphabetically:

```
Adventure, AirRaid, Alien, Amidar, Assault, Asterix, Asteroids,
Atlantis, Atlantis2, Backgammon, BankHeist, BasicMath, BattleZone,
BeamRider, Berzerk, Blackjack, Bowling, Boxing, Breakout, Carnival,
Casino, Centipede, ChopperCommand, CrazyClimber, Crossbow, Darkchambers,
Defender, DemonAttack, DonkeyKong, DoubleDunk, Earthworld,
ElevatorAction, Enduro, Entombed, Et, FishingDerby, FlagCapture,
Freeway, Frogger, Frostbite, Galaxian, Gopher, Gravitar, Hangman,
HauntedHouse, Hero, HumanCannonball, IceHockey, Jamesbond,
JourneyEscape, Kaboom, Kangaroo, KeystoneKapers, KingKong, Klax,
Koolaid, Krull, KungFuMaster, LaserGates, LostLuggage, MarioBros,
MiniatureGolf, MontezumaRevenge, MrDo, MsPacman, NameThisGame,
Othello, Pacman, Phoenix, Pitfall, Pitfall2, Pong, Pooyan, PrivateEye,
Qbert, Riverraid, RoadRunner, Robotank, Seaquest, SirLancelot, Skiing,
Solaris, SpaceInvaders, SpaceWar, StarGunner, Superman, Surround,
Tennis, Tetris, TicTacToe3D, TimePilot, Trondead, Turmoil, Tutankham,
UpNDown, Venture, VideoCheckers, VideoChess, VideoCube, VideoPinball,
WizardOfWor, WordZapper, YarsRevenge, Zaxxon
```

## Training Time Estimates (Colab T4 GPU)

Atari training is measured in **frames** not episodes. Standard DQN/PPO benchmarks:

| Algorithm | Frames to Train | Time @ 400 fps | Colab 10min? |
|-----------|----------------|----------------|-------------|
| DQN (1 game) | 10M | ~7 hours | **No** → Kaggle |
| PPO (1 game) | 10M | ~7 hours | **No** → Kaggle |
| Quick test (DQN) | 500k | ~20 min | Borderline |

Atari training needs Kaggle (30h/week GPU) or a local GPU. Colab's ~10 minute window is only sufficient for a smoke test (confirm env works, check first 50k frames of learning).

### Quick Smoke Test Script

```python
import gymnasium as gym
import ale_py  # registers ALE namespace
import time

env = gym.make("ALE/Pong-v5", obs_type="ram")
obs, _ = env.reset()
t0 = time.time()
for _ in range(10000):
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, _ = env.reset()
elapsed = time.time() - t0
print(f"10k random steps: {elapsed:.1f}s → {10000/elapsed:.0f} fps")
env.close()
```

Expect ~400-800 fps on T4 GPU with RAM mode, ~100-200 fps with pixel mode + wrappers.

## Key Differences from Gym (0.21) / Gymnasium (0.29)

1. **Env IDs**: `ALE/Pong-v5` not `PongNoFrameskip-v4`. The `ALE/` namespace is new in gymnasium 1.x.
2. **No separate RAM envs**: `ALE/Pong-ram-v5` does NOT exist. Use `gym.make("ALE/Pong-v5", obs_type="ram")`.
3. **Frameskip control**: Pass `frameskip=1` to `gym.make()` and use `AtariPreprocessing` wrapper for frame skip.
4. **import ale_py required**: `import ale_py` must run before `gym.make()` to register the ALE namespace.
5. **full_action_space**: Default is True (18 actions). Use `full_action_space=False` for the minimal action set.

## Minimal Working Example

```python
import ale_py  # noqa — must be imported before gym.make()
import gymnasium as gym

# RAM mode (MLP-friendly, 128-dim)
env = gym.make("ALE/Pong-v5", obs_type="ram", full_action_space=False)
print(f"RAM obs: {env.observation_space.shape}, actions: {env.action_space.n}")

# Pixel mode (CNN, 210x160x3)
env = gym.make("ALE/Pong-v5", full_action_space=False)
print(f"Pixel obs: {env.observation_space.shape}, actions: {env.action_space.n}")
env.close()
```

## Data Source

Full 104-game specs collected from gymnasium 1.3.0 on Colab (2026-06-14). Raw JSON at `tmp/classic_atari_reference.json` (585MB — includes full obs/act space arrays for every game).
