
import json
import os
import ast


SAVE_FILE = "flappy_qtable.json"

def save_decisions():
    data = {
        "best_table": {str(key): value for key, value in best_table.items()},
        "epsilon": epsilon,
        "episode_count": episode_count,
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f)
    print(f"[SAVED] {len(best_table)} states, epsilon={epsilon:.4f}")


def load_decisions():
    global best_table, epsilon, episode_count
    if not os.path.exists(SAVE_FILE):
        print("[LOAD] no save file found, starting fresh")
        return
    with open(SAVE_FILE, "r") as f:
        data = json.load(f)
    best_table = {ast.literal_eval(k): v for k, v in data["best_table"].items()}
    epsilon = data.get("epsilon", 1)
    episode_count = data.get("episode_count", 0)
    print(f"[LOADED] {len(best_table)} states, epsilon={epsilon:.4f}, resuming at episode {episode_count}")

best_table={}
learning_rate = 0.3
MULTIPLIER = 1
EPSILON_MIN = 0.05
epsilon = 1
exploration_value = 0.05

def floor_to_two(value):
    return (value // 2) * 2

def floor_to_five(value):
    return (value // 5) * 5

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
        # bird y, bird vel, pipe top bottom, pipe bottom top, dx 
        cur_state = (floor_to_20(bird_pos.y), floor_to_two(bird_vel), floor_to_50(top_pipe.bottom), floor_to_50(bottom_pipe.top), floor_to_20(bird_pos.x- bottom_pipe.x))
        # now to put these values in buckets:
        # how close can these values get? RANGE
        # how finely do you actually need to distinguish, for the purpose of deciding jump vs no-jump?
        if not best_table.get(cur_state):
            best_table[cur_state] = { "jump": 0, "no_jump": 0}
        return cur_state
    
    
    cur_state = (floor_to_20(bird_pos.y), floor_to_two(bird_vel), -100, -100, -100)
    
    if cur_state not in best_table:
        best_table[cur_state] = { "jump": 0, "no_jump": 0}
    
    return cur_state


def action(state):
    # player jump
    import random
    jump_state = False
    
    rand_choice = random.random()
    if rand_choice < epsilon:
        print("LEARNING BASED")
        if random.random() < exploration_value:
            jump_state = True
    else:
        print("VALUE BASED")
        # TODO 4: look up value of (state, jump) and (state, no-jump) and pick higher value
        jump_value = best_table[state]["jump"]
        no_jump_value = best_table[state]["no_jump"]
        if jump_value > no_jump_value:
            # the jump state
            jump_state = True
    
    # TODO 7: MUST RETURN WHAT ACTION TAKEN
    print(f"[ACTION] state={state} epsilon={epsilon:.4f} mode={'explore' if rand_choice < epsilon else 'exploit'} -> jump={jump_state}")
    return jump_state



def reward(state, next_state, game_active):
    # TODO 8: comparison between state and next state:
    
    _, _, gap_top, gap_bottom, dx = state
    next_bird_y, next_vel, next_gap_top, next_gap_bottom, next_dx = next_state
    
    if not game_active: # on death
        return -10
    if dx <= 0 and next_dx > 0: # pipe was ahead of bird and is now behind
        return 21
    if next_gap_top == -100: # no pipe currently tracked - nothing to align to yet
        distance_from_ceiling = next_bird_y - (-100)   # smaller = closer to ceiling
        distance_from_floor = 900 - next_bird_y         # smaller = closer to floor
        danger_margin = min(distance_from_ceiling, distance_from_floor)
        
        if distance_from_ceiling < 100 and next_vel < 0: # too close to the top
            return -11
        elif distance_from_floor < 100 and next_vel > 0: # too close to the top
            return -5
        elif distance_from_ceiling < 200 and next_vel < 0:
            return -3 # somewhat close, mild caution
        elif distance_from_floor < 200 and next_vel > 0:
            return -3 # somewhat close, mild caution
        else:
            return 0    # safely mid-screen, neutral as before
        
    else:
        if (next_bird_y < next_gap_top and next_vel > 0) or (next_bird_y > next_gap_bottom and next_vel < 0): # too close to the top
            return 5
        elif (next_bird_y < next_gap_top and next_vel < 0) or (next_bird_y > next_gap_bottom and next_vel > 0): # too close to the top
            return -5
        
    if next_gap_top <= next_bird_y <= next_gap_bottom: # on target: is the bird's y bucket inside the gap?
        return 10
    
    return -5 # off target


DISCOUNT = 0.9
episode_frame_count = 0
def update_decision(state, action, reward, next_state, game_active):
    global episode_frame_count
    episode_frame_count += 1
    
    # TODO 6: MUST read a value store. dict keyed by (discretized_state, action) -> estimated value
    # function reads from and later updates
    old_value = best_table[state]["jump" if action else "no_jump"]
    
    if not game_active:
        target = reward
    else:
        next_values = best_table[next_state]
        best_next_value = max(next_values["jump"], next_values["no_jump"])
        target = reward + DISCOUNT * best_next_value
        
        
    new_value = old_value + learning_rate * (target - old_value)
    best_table[state]["jump" if action else "no_jump"] = new_value
    print(f"[UPDATE] state={state} action={'jump' if action else 'no_jump'} reward={reward} old={old_value:.2f} -> new={new_value:.2f}")

    
episode_count = 0

def decay_epsilon():
    global epsilon, episode_count, episode_frame_count
    episode_count += 1
    print(f"[EPISODE {episode_count}] survived {episode_frame_count} frames | epsilon={epsilon:.4f} | states={len(best_table)}")
    episode_frame_count = 0
    epsilon = max(epsilon * MULTIPLIER, EPSILON_MIN)
    if episode_count % 10 == 0:
        save_decisions()