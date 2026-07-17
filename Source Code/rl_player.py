
import json
import os
import ast
import csv

SAVE_FILE = "flappy_qtable.json"
LOG_FILE = "training_log.csv"
EVAL_LOG_FILE = "eval_log.csv"

best_pipes = 0  # add near the top with the other globals
VERBOSE = True
best_table={}
MULTIPLIER = 0.99
EPSILON_MIN = 0.05
epsilon = 1
EXPLORE_JUMP_PROB = 0.0305
learning_rate = 0.3
LR_MIN = 0.01
LR_MULTIPLIER = 0.995

EVAL_INTERVAL = 50   # run one greedy (epsilon=0) episode every N episodes
eval_mode = False

DISCOUNT = 0.9
episode_frame_count = 0
episode_count = 0

SHAPING_SCALE = 0.2          # keeps positional shaping from dwarfing death/pipe rewards
VEL_PENALTY_DY_THRESHOLD = 100  # "well below target" cutoff, in raw pixels
VEL_PENALTY_SCALE = 0.02

episode_pipe_passes=0

BEST_SAVE_FILE = "flappy_qtable_best.json"
best_eval_pipes = 0

def save_if_best(eval_pipes):
    global best_eval_pipes
    if eval_pipes > best_eval_pipes:
        best_eval_pipes = eval_pipes
        data = {
            "best_table": {str(key): value for key, value in best_table.items()},
            "epsilon": epsilon,
            "episode_count": episode_count,
            "best_pipes": best_pipes,
        }
        with open(BEST_SAVE_FILE, "w") as f:
            json.dump(data, f)
        log(f"[BEST SAVED] new record: {eval_pipes} pipes at episode {episode_count}")

def save_decisions():
    data = {
        "best_table": {str(key): value for key, value in best_table.items()},
        "epsilon": epsilon,
        "episode_count": episode_count,
        "best_pipes": best_pipes,
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f)
    print(f"[SAVED] {len(best_table)} states, epsilon={epsilon:.4f}, best_pipes={best_pipes}")


def load_decisions():
    global best_table, epsilon, episode_count, best_pipes
    if not os.path.exists(SAVE_FILE):
        print("[LOAD] no save file found, starting fresh")
        return
    with open(SAVE_FILE, "r") as f:
        data = json.load(f)
    best_table = {ast.literal_eval(k): v for k, v in data["best_table"].items()}
    epsilon = data.get("epsilon", 1)
    episode_count = data.get("episode_count", 0)
    best_pipes = data.get("best_pipes", 0)
    print(f"[LOADED] {len(best_table)} states, epsilon={epsilon:.4f}, best_pipes={best_pipes}, resuming at episode {episode_count}")

def log(msg):
    if VERBOSE:
        print(msg)  

def floor_to_two(value):
    return (value // 2) * 2

def floor_to_five(value):
    return (value // 5) * 5

def floor_to_10(value):
    return (value // 10) * 10

def floor_to_20(value):
    return (value // 20) * 20

def floor_to_50(value):
    return (value // 50) * 50

def state(bird_pos, bird_vel, pipe_pos): # take constant stream of info
    global best_table
    

    # pipe: nearest upcoming bottom/top pair (pipes are stored as
    # [bottom0, top0, bottom1, top1, ...] and old ones are never pruned,
    # so pipe_pos[0] is stale off-screen debris, not the next obstacle)
    upcoming = [pipe for pipe in pipe_pos if pipe.right >= bird_pos.left]
    if upcoming:
        nearest_x = min(pipe.centerx for pipe in upcoming) # take the closest pipe
        bottom_pipe = next(pipe for pipe in upcoming if pipe.centerx == nearest_x and pipe.bottom >= 900) # this just means the bottom of the image is above 900px downward
        top_pipe = next(pipe for pipe in upcoming if pipe.centerx == nearest_x and pipe.bottom < 900) # this just means the bottom of the image is below 900px downward
        
        gap_center = (top_pipe.bottom + bottom_pipe.top) // 2
        raw_dx = bird_pos.x - bottom_pipe.x
        raw_dy = bird_pos.y - gap_center
        
        # bird y, bird vel, pipe top bottom, pipe bottom top, dx 
        cur_state = (floor_to_two(bird_vel), floor_to_50(raw_dx), floor_to_50(raw_dy))
        # now to put these values in buckets:
        # how close can these values get? RANGE
        # how finely do you actually need to distinguish, for the purpose of deciding jump vs no-jump?
    else:
        raw_dx = -100
        raw_dy = bird_pos.y - 450
        cur_state = (floor_to_two(bird_vel), -100, floor_to_50(raw_dy))
    
    if not best_table.get(cur_state):
        best_table[cur_state] = {"jump": 0, "no_jump": 0, "jump_n": 0, "no_jump_n": 0}
    
    raw_state = (bird_vel, raw_dx, raw_dy)
    
    return cur_state, raw_state


def action(state):
    # player jump
    import random
    jump_state = False
    
    eff_epsilon = 0 if eval_mode else epsilon # force greedy during eval episodes
    rand_choice = random.random()
    if rand_choice < eff_epsilon:
        jump_state = random.random() < EXPLORE_JUMP_PROB
    else:
        # TODO 4: look up value of (state, jump) and (state, no-jump) and pick higher value
        jump_value = best_table[state]["jump"]
        no_jump_value = best_table[state]["no_jump"]
        if jump_value > no_jump_value:
            # the jump state
            jump_state = True
    
    # TODO 7: MUST RETURN WHAT ACTION TAKEN
    log(f"[ACTION] state={state} epsilon={eff_epsilon:.4f} eval={eval_mode} mode={'explore' if rand_choice < eff_epsilon else 'exploit'} -> jump={jump_state}")
    return jump_state



def potential(raw_dy):
    # Higher potential = better state. Using negative distance means
    # "closer to the gap center" is always more rewarding than "farther."
    return -abs(raw_dy) * 5

def velocity_penalty(raw_dy, raw_vel):
    # extra penalty for falling fast while already well below target
    if raw_dy > VEL_PENALTY_DY_THRESHOLD and raw_vel > 0:
        return -VEL_PENALTY_SCALE * raw_vel
    if raw_dy < -VEL_PENALTY_DY_THRESHOLD and raw_vel < 0:
        return -VEL_PENALTY_SCALE * abs(raw_vel)
    return 0

def reward(state, raw_state, next_state, next_raw_state, game_active):
    global episode_pipe_passes
    _, dx, dy = state
    _, next_dx, next_dy = next_state
    raw_vel, raw_dx, raw_dy = raw_state
    next_raw_vel, next_raw_dx, next_raw_dy = next_raw_state

    if not game_active:
        return -10

    shaping = SHAPING_SCALE * (potential(next_raw_dy) - potential(raw_dy))
    shaping += velocity_penalty(next_raw_dy, next_raw_vel)

    bonus = 0
    if dx <= 0 and next_dx > 0:   # passed a pipe
        bonus = 21
        episode_pipe_passes += 1
        log(f"[PIPE PASSED] dx={dx} -> next_dx={next_dx} (total this episode: {episode_pipe_passes})")

    return shaping + bonus

def get_learning_rate(state, action_key):
    n = best_table[state].get(f"{action_key}_n", 0)
    return max(1.0 / (1 + n), LR_MIN)

def update_decision(state, action, reward, next_state, game_active):
    global episode_frame_count
    episode_frame_count += 1
    action_key = "jump" if action else "no_jump"

    old_value = best_table[state][action_key]
    lr = get_learning_rate(state, action_key)

    if not game_active:
        target = reward
    else:
        next_values = best_table[next_state]
        best_next_value = max(next_values["jump"], next_values["no_jump"])
        target = reward + DISCOUNT * best_next_value

    new_value = old_value + lr * (target - old_value)
    best_table[state][action_key] = new_value
    best_table[state][f"{action_key}_n"] = best_table[state].get(f"{action_key}_n", 0) + 1
    log(f"[UPDATE] state={state} action={action_key} reward={reward} old={old_value:.2f} -> new={new_value:.2f}")


def log_qvalue_stats():
    if not best_table:
        return
    all_values = [v for entry in best_table.values() for k, v in entry.items() if not k.endswith("_n")]
    log(f"[QSTATS] min={min(all_values):.2f} max={max(all_values):.2f} states={len(best_table)}")


def _write_csv_row(path, header, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

def decay_epsilon():
    global epsilon, episode_count, episode_frame_count, eval_mode, episode_pipe_passes
    was_eval = eval_mode
    episode_count += 1
    log(f"[EPISODE {episode_count}] survived {episode_frame_count} frames | pipes={episode_pipe_passes} | epsilon={epsilon:.4f} | eval={was_eval} | states={len(best_table)}")
    
    if was_eval:
        _write_csv_row(EVAL_LOG_FILE, ["episode_count", "frames", "pipes"], [episode_count, episode_frame_count, episode_pipe_passes])
        save_if_best(episode_pipe_passes)
        eval_mode = False
    else:
        _write_csv_row(LOG_FILE, ["episode_count", "frames", "pipes", "epsilon"], [episode_count, episode_frame_count, episode_pipe_passes, epsilon])
        epsilon = max(epsilon * MULTIPLIER, EPSILON_MIN)
        
    episode_frame_count = 0
    episode_pipe_passes = 0
    
    if episode_count % EVAL_INTERVAL == 0 and not was_eval:
        eval_mode = True   # next episode runs greedy

    log_qvalue_stats()

    if episode_count % 10 == 0:
        save_decisions()