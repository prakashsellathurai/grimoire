import sys
import math

# =========================
#   CONFIG
# =========================
BASE_SPEED = 60      # Base thrust speed
MAX_SPEED = 100      # Maximum thrust speed
PREDICT_TURNS = 3.14  # Look ahead for drift compensation


# =========================
#   POD AI
# =========================
class PodAI:
    """
    Per-pod AI:
    - Velocity from previous position
    - Drift compensation with predicted target
    - Cosine-based thrust scaling
    - BOOST decision (actual availability controlled by hive)
    - SHIELD based on distance & angle to opponent (gated by hive)
    """
    def __init__(self, base_speed, mode, checkpoints):
        self.prev_x = None
        self.prev_y = None
        self.prev_vx = 0
        self.prev_vy = 0
        self.BASE_SPEED = base_speed
        self.mode = mode          # "racer" or "blocker"
        self.checkpoints = checkpoints

    def compute_action(
        self,
        x,
        y,
        angle,
        cp_id,
        checkpoints,
        opponent_cp_id,
        opponent_x,
        opponent_y,
        can_boost,
        allow_shield=True
    ):
        """
        Returns: (target_x, target_y, thrust, used_boost)
        thrust is either int or "BOOST" or "SHIELD"
        used_boost is a bool so the hive can mark BOOST as consumed.
        """

        used_boost = False

        # --- Choose which checkpoint to aim for, based on mode ---
        if self.mode == "racer":
            target_cp_id = cp_id              # my own next checkpoint
        else:  # "blocker"
            target_cp_id = opponent_cp_id     # opponent's next checkpoint

        # --- Velocity from previous position ---
        if self.prev_x is not None:
            vx = x - self.prev_x
            vy = y - self.prev_y
        else:
            vx, vy = 0, 0

        # --- Current checkpoint target ---
        next_checkpoint_x, next_checkpoint_y = checkpoints[target_cp_id]

        dx = next_checkpoint_x - x
        dy = next_checkpoint_y - y
        next_checkpoint_dist = math.hypot(dx, dy)

        # Angle to checkpoint (absolute)
        angle_to_cp = math.degrees(math.atan2(dy, dx))
        next_checkpoint_angle = angle_to_cp - angle

        # Normalize angle to [-180, 180]
        while next_checkpoint_angle > 180:
            next_checkpoint_angle -= 360
        while next_checkpoint_angle < -180:
            next_checkpoint_angle += 360

        # --- Predict future position and compensate for drift ---
        target_x = next_checkpoint_x - vx * PREDICT_TURNS
        target_y = next_checkpoint_y - vy * PREDICT_TURNS

        # --- Thrust based on angle using cosine ---
        thrust = int(
            self.BASE_SPEED
            + (MAX_SPEED - self.BASE_SPEED) * math.cos(math.radians(next_checkpoint_angle))
        )

        # Clamp thrust between 0 and MAX_SPEED just in case
        if isinstance(thrust, int):
            if thrust < 0:
                thrust = 0
            elif thrust > MAX_SPEED:
                thrust = MAX_SPEED

        # --- Smart BOOST usage (shared via hive with can_boost) ---
        if (
            can_boost
            and isinstance(thrust, int)
            and abs(next_checkpoint_angle) < 10
            and next_checkpoint_dist > 1500
        ):
            thrust = "BOOST"
            used_boost = True

        # --- SHIELD decision (distance & angle to opponent) ---
        #   Final permission (allow_shield) is given by the hive.
        if self.mode == "blocker" and allow_shield and thrust != "BOOST":
            dx_op = opponent_x - x
            dy_op = opponent_y - y
            dist_to_opponent = math.hypot(dx_op, dy_op)

            # Angle to opponent (relative to pod facing)
            angle_to_opp = math.degrees(math.atan2(dy_op, dx_op))
            opp_angle = angle_to_opp - angle
            while opp_angle > 180:
                opp_angle -= 360
            while opp_angle < -180:
                opp_angle += 360

            # Only shield if very close and roughly in front of us
            if dist_to_opponent < 150 and abs(opp_angle) < 45:
                thrust = "SHIELD"

        # Store state for next turn
        self.prev_x, self.prev_y = x, y
        self.prev_vx, self.prev_vy = vx, vy

        return int(target_x), int(target_y), thrust, used_boost

    def is_blocker(self):
        return self.mode == "blocker"


# =========================
#   HIVE
# =========================
class PodHive:
    """
    Shared brain for both pods.
    Holds both PodAI instances and shared state/logic:
    - Single shared BOOST for both pods
    - Prevent SHIELD when our two pods are too close.
    - Lap/progress tracking for both my pods and both opponent pods.
    """
    def __init__(self, base_speed, checkpoints, total_laps):
        self.checkpoints = checkpoints
        self.total_laps = total_laps
        self.checkpoint_count = len(checkpoints)

        self.boost_used = False   # shared BOOST across both pods

        self.pod_ais = [
            PodAI(base_speed, "racer",   checkpoints),
            PodAI(base_speed, "blocker", checkpoints)
        ]

        # Lap / checkpoint tracking (per pod)
        self.my_laps = [0, 0]
        self.opp_laps = [0, 0]
        self.prev_my_cp = [None, None]
        self.prev_opp_cp = [None, None]

    def _update_laps(self, pods, prev_cps, laps):
        """
        Increment lap when cp_id wraps around (e.g. 5 -> 0).
        """
        for i, pod in enumerate(pods):
            cp_id = pod["cp_id"]
            prev_cp = prev_cps[i]
            if prev_cp is not None and cp_id < prev_cp:
                laps[i] += 1
            prev_cps[i] = cp_id

    def _progress(self, lap, cp_id, x, y):
        """
        Scalar progress along the race:
        higher value = further ahead.
        Uses (lap, checkpoint index, and distance to next checkpoint).
        """
        cp_x, cp_y = self.checkpoints[cp_id]
        dist = math.hypot(cp_x - x, cp_y - y)
        # Distance is divided so it slightly affects ordering but not dominate
        return lap * self.checkpoint_count + cp_id - dist / 100000.0

    def compute_turn(self, my_pods, opp_pods):
        # --- Lap tracking update for *all* pods ---
        self._update_laps(my_pods, self.prev_my_cp, self.my_laps)
        self._update_laps(opp_pods, self.prev_opp_cp, self.opp_laps)

        # --- Compute progress for my pods ---
        my_progress = []
        for i, pod in enumerate(my_pods):
            prog = self._progress(
                self.my_laps[i],
                pod["cp_id"],
                pod["x"],
                pod["y"]
            )
            my_progress.append(prog)

        # --- Compute progress for opponent pods ---
        opp_progress = []
        for i, pod in enumerate(opp_pods):
            prog = self._progress(
                self.opp_laps[i],
                pod["cp_id"],
                pod["x"],
                pod["y"]
            )
            opp_progress.append(prog)

        # Which of my pods is ahead?
        if my_progress[0] >= my_progress[1]:
            front_idx, back_idx = 0, 1
        else:
            front_idx, back_idx = 1, 0

        best_my = my_progress[front_idx]
        best_opp = max(opp_progress)
        best_opp_idx = opp_progress.index(best_opp)

        # --- Decide roles for this turn ---
        # Default both racer
        roles = ["racer", "racer"]

        # 2) If opponents are ahead of both bots -> both racer
        if best_opp > best_my:
            roles[0] = "racer"
            roles[1] = "racer"
        else:
            # 1) One pod is ahead -> closest becomes racer, other blocker
            roles[front_idx] = "racer"
            roles[back_idx] = "blocker"

        # Use leading opponent as reference for blocker targeting & shield logic
        opponent_ref = opp_pods[best_opp_idx]
        opponent_x = opponent_ref["x"]
        opponent_y = opponent_ref["y"]
        opponent_cp_id = opponent_ref["cp_id"]

        # Distance between our own two pods
        dx_my = my_pods[0]["x"] - my_pods[1]["x"]
        dy_my = my_pods[0]["y"] - my_pods[1]["y"]
        dist_my_pods = math.hypot(dx_my, dy_my)

        # If our pods are too close, globally disable SHIELD this turn
        POD_PROXIMITY_LIMIT = 100  # ~collision distance (2 * 400) plus small margin
        allow_shield = dist_my_pods > POD_PROXIMITY_LIMIT

        outputs = []
        for idx in range(2):
            pod = my_pods[idx]
            ai = self.pod_ais[idx]

            # Apply dynamic mode
            ai.mode = roles[idx]

            tx, ty, thrust, used_boost = ai.compute_action(
                pod["x"],
                pod["y"],
                pod["angle"],
                pod["cp_id"],
                self.checkpoints,
                opponent_cp_id,
                opponent_x,
                opponent_y,
                can_boost=(not self.boost_used),
                allow_shield=allow_shield
            )

            if used_boost:
                self.boost_used = True

            outputs.append(f"{tx} {ty} {thrust}")

        return outputs


# =========================
#   MAIN GAME SETUP
# =========================
laps = int(input())
checkpoint_count = int(input())
checkpoints = []
for _ in range(checkpoint_count):
    checkpoint_x, checkpoint_y = [int(j) for j in input().split()]
    checkpoints.append((checkpoint_x, checkpoint_y))

# Create the shared hive (which in turn owns both PodAIs)
hive = PodHive(BASE_SPEED, checkpoints, laps)

# =========================
#   GAME LOOP
# =========================
while True:
    my_pods = []
    opp_pods = []

    # Read my 2 pods
    for _ in range(2):
        # x: x position of your pod
        # y: y position of your pod
        # vx: x speed of your pod (ignored, we recompute like in your first code)
        # vy: y speed of your pod
        # angle: angle of your pod
        # next_check_point_id: next check point id of your pod
        x, y, vx_in, vy_in, angle, next_check_point_id = [int(j) for j in input().split()]
        my_pods.append({
            "x": x,
            "y": y,
            "angle": angle,
            "cp_id": next_check_point_id
        })

    # Read opponent 2 pods
    for _ in range(2):
        x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2 = [int(j) for j in input().split()]
        opp_pods.append({
            "x": x_2,
            "y": y_2,
            "cp_id": next_check_point_id_2
        })

    # Let the hive compute both commands
    outputs = hive.compute_turn(my_pods, opp_pods)

    # Output for both pods
    print(outputs[0])
    print(outputs[1])
