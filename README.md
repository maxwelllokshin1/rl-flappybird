<div align="center">

# Flappy Bird RL

### A Q-learning agent that teaches itself to play Flappy Bird

<p>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/pygame-2.x-green?logo=pygame&logoColor=white" alt="Pygame">
  <img src="https://img.shields.io/badge/RL-Q--Learning-orange" alt="Q-Learning">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License">
  <img src="https://img.shields.io/badge/status-in%20progress-yellow" alt="Status">
</p>

<!-- Swap this for an actual screenshot or GIF of the agent playing -->
<img src="assets/demo.gif" width="300" alt="Agent playing Flappy Bird">

</div>

---

## Table of Contents

- [About](#-about)
- [How the Agent Learns](#-how-the-agent-learns)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Credits](#-credits)
- [License](#-license)

---

## About

A Flappy Bird clone (built on Pygame) paired with a tabular **Q-learning** agent that learns to play through trial and error — no hand-coded control logic, just states, actions, and rewards.

> Replace this paragraph with 2–3 sentences on what makes your version interesting: what you changed, what you're experimenting with, current performance, etc.

---

## How the Agent Learns

<details>
<summary><b>State representation</b></summary>

<br>

The agent discretizes the raw game info into a bucketed state:

| Component | Description |
|---|---|
| Bird Y position | floored to nearest 20px |
| Bird velocity | floored to nearest 2 |
| Gap top / bottom | floored to nearest 50px |
| Horizontal distance to pipe (dx) | floored to nearest 20px |

</details>

<details>
<summary><b>Action space</b></summary>

<br>

Two actions per frame:

- `jump`
- `no_jump`

</details>

<details>
<summary><b>Reward shaping</b></summary>

<br>

| Event | Reward |
|---|---|
| Death (collision / out of bounds) | `-10` |
| Successfully passing a pipe | `+21` |
| Inside the gap, on target | `+10` |
| Drifting toward the gap edge | `+5` |
| Drifting away from the gap edge | `-5` |
| Too close to ceiling/floor | `-11` / `-5` |

</details>

<details>
<summary><b>Training loop</b></summary>

<br>

1. Observe state
2. Choose action (epsilon-greedy)
3. Apply action, step physics
4. Observe reward + next state
5. Update Q-value via the Bellman equation
6. Decay epsilon after each episode
7. Periodically persist the Q-table to disk

</details>

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

**Train / watch the agent play:**

```bash
python main.py
```

- The agent's Q-table is saved automatically to `flappy_qtable.json` every 10 episodes.
- Delete `flappy_qtable.json` to start training from scratch.
- Progress (episode length, epsilon, state count) is logged to the console.

**Play manually:**

> Note the current codebase always routes input through the RL agent — add a `--human` flag or similar if you want manual play alongside training.

---

## Project Structure

```
.
├── main.py              # Game loop, physics, rendering (Pygame)
├── rl_player.py          # Q-learning agent: state, action, reward, updates
├── flappy_qtable.json    # Saved Q-table (generated after training)
├── assets/                # Sprites, sounds, background
└── README.md
```

---

## Configuration

Key hyperparameters live in `rl_player.py`:

| Variable | Meaning | Default |
|---|---|---|
| `learning_rate` | How much each update shifts the Q-value | `0.3` |
| `DISCOUNT` | Weight given to future rewards | `0.9` |
| `epsilon` | Initial exploration rate | `1` |
| `EPSILON_MIN` | Floor for exploration rate | `0.05` |
| `MULTIPLIER` | Per-episode epsilon decay factor | `1` |

> `MULTIPLIER` currently doesn't decay epsilon — set it below `1` (e.g. `0.995`) so the agent gradually shifts from exploring to exploiting.

---

## Roadmap

- [ ] Fix epsilon decay (`MULTIPLIER < 1`)
- [ ] Reduce state space to relative features (distance-to-gap-center vs. absolute position)
- [ ] Add learning rate decay
- [ ] Experiment with potential-based reward shaping
- [ ] Try SARSA / Double Q-learning for comparison
- [ ] Replace tabular Q-table with a small neural net (DQN)

---

## Credits

Base game built on the [Flappy Bird Pygame clone](https://github.com/Amey-Thakur/FLAPPY-BIRD-USING-PYGAME) by Amey Thakur & Mega Satish.

RL agent and training loop by <your name>.

---

## License

This project is licensed under the [MIT License](LICENSE).

</div>