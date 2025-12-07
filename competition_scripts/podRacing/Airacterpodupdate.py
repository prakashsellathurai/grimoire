import sys
import math
from enum import Enum
import random

class CHARACTER(Enum):
    RACER = 1
    SHIELDER = 2

# ======================================
# READ STATIC RACE DATA
# ======================================
laps = int(input())
checkpoint_count = int(input())

checkpoints = []
for i in range(checkpoint_count):
    checkpoint_x, checkpoint_y = [int(j) for j in input().split()]
    checkpoints.append((checkpoint_x, checkpoint_y))

# ======================================
# PRECALCULATE RACING LINE THROUGH CHECKPOINTS
# ======================================
# We aim a bit *past* each checkpoint, in the direction of the next checkpoint,
# to smooth the path and reduce sharp turns.
RACING_OFFSET = 500  # tune this (e.g. 300–800) for different cornering behavior

racing_targets = []
for i in range(checkpoint_count):
    curr_x, curr_y = checkpoints[i]
    next_x, next_y = checkpoints[(i + 1) % checkpoint_count]

    dx = next_x - curr_x
    dy = next_y - curr_y
    dist = math.hypot(dx, dy)

    if dist != 0:
        nx = dx / dist
        ny = dy / dist
    else:
        nx = ny = 0.0

    # Target slightly ahead of checkpoint center in the direction of next checkpoint
    target_x = curr_x + nx * RACING_OFFSET
    target_y = curr_y + ny * RACING_OFFSET

    racing_targets.append((target_x, target_y))


# ======================================
# POD CLASS
# ======================================
class Pod:
    def __init__(self, basespeed=60, maxspeed=100, boost_used=False, strategy: CHARACTER = CHARACTER.RACER):
        # Previous position (for drift calculation)
        self.prev_x = None
        self.prev_y = None

        self.basespeed = basespeed
        self.maxspeed = maxspeed
        self.boost_used = boost_used
        self.thrust = basespeed

        self.target_x = None
        self.target_y = None

        self.strategy = strategy

    def update(self, x, y, vx, vy, angle, next_cp_id):
        # Use precalculated racing target instead of raw checkpoint center
        base_tx, base_ty = racing_targets[next_cp_id]

        # Compute distance to (real) checkpoint center for boost logic
        cp_x, cp_y = checkpoints[next_cp_id]
        dx_cp = cp_x - x
        dy_cp = cp_y - y
        dist_cp = math.hypot(dx_cp, dy_cp)

        # Angle to checkpoint center (for alignment / boost decisions)
        desired_angle = math.degrees(math.atan2(dy_cp, dx_cp))
        ang_diff = (desired_angle - angle + 540) % 360 - 180  # [-180, 180]

        # Velocity-based prediction from previous position (if available)
        if self.prev_x is not None:
            vx_calc = x - self.prev_x
            vy_calc = y - self.prev_y
        else:
            vx_calc = vy_calc = 0

        predict_turns = 3

        # Thrust based on alignment with checkpoint (use angle difference, not raw angle)
        alignment = math.cos(math.radians(ang_diff))
        alignment = max(0.0, alignment)  # Don't accelerate when facing away
        self.thrust = int(self.basespeed + (self.maxspeed - self.basespeed) * alignment)

        # Extra slow-down when very close to checkpoint to avoid overshooting
        if dist_cp < 800:
            self.thrust = int(self.thrust * 0.6)

        # Smart BOOST: only when almost straight and far away
        if (not self.boost_used) and abs(ang_diff) < 8 and dist_cp > 4000:
            self.thrust = "BOOST"
            self.boost_used = True

        # Use racing target + drift compensation
        self.target_x = base_tx - vx * predict_turns
        self.target_y = base_ty - vy * predict_turns

        # save previous pos
        self.prev_x = x
        self.prev_y = y

    def updateOppenent(self, x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2):
        if self.strategy == CHARACTER.SHIELDER:
            # Here you can implement interception logic using opponent position / future CP
            # Example idea: target between opponent and its next checkpoint.
            pass
        else:
            pass

    def updateOppenents(self, oppenent_states):
        (x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2) = random.choice(oppenent_states)
        self.updateOppenent(x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2)

    def action(self):
        self.thrust = self.thrust if self.thrust == "BOOST" else abs(int(self.thrust))
        print(f"{int(self.target_x)} {int(self.target_y)} {self.thrust}")


# Two bots: racer + blocker/back pod
bots = [
    Pod(),  # main racer
    Pod(basespeed=10, strategy=CHARACTER.SHIELDER, boost_used=False)  # shielder / support
]

# ======================================
# GAME LOOP
# ======================================
while True:
    # Read our pods
    for i in range(2):
        x, y, vx, vy, angle, next_cp_id = [int(j) for j in input().split()]
        bots[i].update(x, y, vx, vy, angle, next_cp_id)

    # Read opponent pods
    oppenents = []
    for i in range(2):
        x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2 = [int(j) for j in input().split()]
        oppenents.append((x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2))

    # Update based on opponents (for shielder logic, if/when you add it)
    for i in range(2):
        bots[i].updateOppenents(oppenents)

    # Output actions
    for bot in bots:
        bot.action()
