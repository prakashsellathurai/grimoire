import sys
import math
from enum import Enum
import random

# ======================================
# CONFIGURATION VARIABLES
# ======================================
# Speed settings
BASE_SPEED = 60
MAX_SPEED = 100

# Prediction settings
PREDICTION_MULTIPLIER = 0.5
MAX_PREDICTION_TURNS = 3
PREDICTION_DISTANCE_DIVISOR = 1000

# Distance thresholds
CLOSE_CHECKPOINT_DIST = 1000
MEDIUM_CHECKPOINT_DIST = 2000

# Boost settings
BOOST_MIN_DISTANCE = 6000
BOOST_MAX_ANGLE = 5
BOOST_MIN_VEL_ALIGNMENT = 0.95

# Turn settings
SHARP_TURN_ANGLE = 90
SHARP_TURN_MAX_THRUST = 30

# Speed factors
MAX_POD_SPEED = 600  # Normalize factor

# Shielder settings
SHIELDER_PREDICTION_TURNS = 2
SHIELDER_THRUST = 100
SHIELDER_ESCORT_DISTANCE = 800  # How close to stay to racer
SHIELDER_INTERCEPT_DISTANCE = 2000  # Distance to start intercepting threats
SHIELDER_BLOCK_THRUST = 100  # Thrust when blocking
SHIELDER_ESCORT_THRUST = 80  # Thrust when escorting
SHIELDER_EARLY_GAME_TURNS = 3  # Number of turns to aggressively disturb opponents at start
SHIELDER_DISTURBANCE_MODE = True  # Start in disturbance mode
SHIELDER_PARALLEL_THRESHOLD = 1500  # Distance to consider "parallel" to opponent
SHIELDER_CHECKPOINT_DENY_DISTANCE = 2500  # How close opponent must be to checkpoint to deny

# Collision avoidance between own pods
FRIENDLY_COLLISION_DISTANCE = 1200  # Distance to start avoiding own pod
FRIENDLY_COLLISION_SLOW_DISTANCE = 800  # Distance to slow down significantly
FRIENDLY_COLLISION_MIN_THRUST = 20  # Minimum thrust when avoiding collision

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
# POD CLASS
# ======================================
class Pod:
    def __init__(self, basespeed=BASE_SPEED, maxspeed=MAX_SPEED, boost_used=False, strategy:CHARACTER=CHARACTER.RACER):
        self.prev_x = None
        self.prev_y = None

        self.basespeed = basespeed
        self.maxspeed = maxspeed
        self.boost_used = boost_used
        self.thrust = basespeed

        self.target_x = None
        self.target_y = None

        self.strategy = strategy
        
        # Shielder specific
        self.racer_pod = None  # Reference to racer pod
        self.my_x = None
        self.my_y = None
        self.my_next_cp_id = None
        self.turn_count = 0  # Track turns for early game aggression
        self.my_vx = None
        self.my_vy = None

    def update(self, x, y, vx, vy, angle, next_cp_id):
        # Store current position for shielder logic
        self.my_x = x
        self.my_y = y
        self.my_vx = vx
        self.my_vy = vy
        self.my_next_cp_id = next_cp_id
        self.turn_count += 1
        
        next_cp_x, next_cp_y = checkpoints[next_cp_id]

        # Distance to checkpoint
        dx = next_cp_x - x
        dy = next_cp_y - y
        dist = math.hypot(dx, dy)

        # Angle to checkpoint
        target_angle = math.degrees(math.atan2(dy, dx))
        ang_diff = (target_angle - angle + 180) % 360 - 180

        # Current speed
        current_speed = math.hypot(vx, vy)
        
        # Velocity direction vs checkpoint direction
        if current_speed > 0:
            velocity_angle = math.degrees(math.atan2(vy, vx))
            vel_to_cp_diff = (target_angle - velocity_angle + 180) % 360 - 180
        else:
            vel_to_cp_diff = 0

        # Predictive targeting based on momentum
        predict_turns = min(MAX_PREDICTION_TURNS, int(dist / PREDICTION_DISTANCE_DIVISOR))
        self.target_x = next_cp_x + vx * predict_turns * PREDICTION_MULTIPLIER
        self.target_y = next_cp_y + vy * predict_turns * PREDICTION_MULTIPLIER

        # ====================================
        # ENHANCED SPEED TUNING
        # ====================================
        
        # Factor 1: Alignment (how well we're pointed at target)
        alignment_factor = math.cos(math.radians(abs(ang_diff)))
        alignment_factor = max(0, alignment_factor)  # 0 to 1
        
        # Factor 2: Velocity alignment (are we moving toward checkpoint?)
        vel_alignment = math.cos(math.radians(abs(vel_to_cp_diff)))
        vel_alignment = max(0, vel_alignment)
        
        # Factor 3: Distance factor (brake near checkpoints)
        if dist < CLOSE_CHECKPOINT_DIST:
            dist_factor = dist / CLOSE_CHECKPOINT_DIST  # Slow down as we approach
        elif dist < MEDIUM_CHECKPOINT_DIST:
            dist_factor = 0.8 + (dist - CLOSE_CHECKPOINT_DIST) / CLOSE_CHECKPOINT_DIST * 0.2  # Gradual
        else:
            dist_factor = 1.0  # Full speed when far
        
        # Factor 4: Speed efficiency (are we going too fast in wrong direction?)
        speed_ratio = current_speed / MAX_POD_SPEED  # Normalize (max speed ~600)
        if vel_alignment < 0.5 and speed_ratio > 0.5:
            # High speed but wrong direction - brake hard
            efficiency_penalty = 0.3
        else:
            efficiency_penalty = 1.0

        # Combine factors
        speed_multiplier = alignment_factor * vel_alignment * dist_factor * efficiency_penalty
        
        # Calculate thrust
        self.thrust = int(self.basespeed + (self.maxspeed - self.basespeed) * speed_multiplier)
        self.thrust = max(0, min(100, self.thrust))  # Clamp 0-100
        
        # Special case: very sharp turn needed
        if abs(ang_diff) > SHARP_TURN_ANGLE:
            self.thrust = min(self.thrust, SHARP_TURN_MAX_THRUST)  # Heavy brake for sharp turns
        
        # Smart BOOST: straight line, far distance, good alignment
        if (not self.boost_used and 
            abs(ang_diff) < BOOST_MAX_ANGLE and 
            dist > BOOST_MIN_DISTANCE and 
            vel_alignment > BOOST_MIN_VEL_ALIGNMENT):
            self.thrust = "BOOST"
            self.boost_used = True

        # Debug output (optional)
        print(f"# Dist:{int(dist)} Ang:{int(ang_diff)} Speed:{int(current_speed)} VelAlign:{vel_alignment:.2f} Thrust:{self.thrust}", file=sys.stderr)

        self.prev_x = x
        self.prev_y = y

    def updateOppenent(self, x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2):
        if self.strategy == CHARACTER.SHIELDER and self.racer_pod:
            # PRIORITY 1: Protect and clear path for racer
            
            # Get racer position and next checkpoint
            racer_x = self.racer_pod.my_x
            racer_y = self.racer_pod.my_y
            racer_next_cp = self.racer_pod.my_next_cp_id
            racer_cp_x, racer_cp_y = checkpoints[racer_next_cp]
            
            # Calculate if opponent is threat to racer
            # Threat = opponent near racer's path to checkpoint
            
            # Vector from racer to checkpoint
            racer_to_cp_x = racer_cp_x - racer_x
            racer_to_cp_y = racer_cp_y - racer_y
            
            # Vector from racer to opponent
            racer_to_opp_x = x_2 - racer_x
            racer_to_opp_y = y_2 - racer_y
            
            # Project opponent onto racer's path
            path_length = math.hypot(racer_to_cp_x, racer_to_cp_y)
            if path_length > 0:
                # Normalize path vector
                path_nx = racer_to_cp_x / path_length
                path_ny = racer_to_cp_y / path_length
                
                # Dot product = how far along path opponent is
                projection = racer_to_opp_x * path_nx + racer_to_opp_y * path_ny
                
                # Perpendicular distance from path
                perp_dist = abs(racer_to_opp_x * path_ny - racer_to_opp_y * path_nx)
                
                # Is opponent a threat? (on path and close)
                is_threat = (projection > 0 and 
                            projection < path_length and 
                            perp_dist < SHIELDER_INTERCEPT_DISTANCE)
            else:
                is_threat = False
            
            # Distance from shielder to racer
            dist_to_racer = math.hypot(racer_x - self.my_x, racer_y - self.my_y)
            
            if is_threat:
                # INTERCEPT MODE: Block opponent threat
                print(f"# SHIELDER: Intercepting threat!", file=sys.stderr)
                # Predict opponent position
                self.target_x = x_2 + vx_2 * SHIELDER_PREDICTION_TURNS
                self.target_y = y_2 + vy_2 * SHIELDER_PREDICTION_TURNS
                self.thrust = SHIELDER_BLOCK_THRUST
                
            elif dist_to_racer > SHIELDER_ESCORT_DISTANCE:
                # ESCORT MODE: Stay close to racer, clear path ahead
                print(f"# SHIELDER: Escorting racer", file=sys.stderr)
                # Position between racer and next checkpoint
                escort_x = racer_x + racer_to_cp_x * 0.3
                escort_y = racer_y + racer_to_cp_y * 0.3
                self.target_x = escort_x
                self.target_y = escort_y
                self.thrust = SHIELDER_ESCORT_THRUST
                
            else:
                # SECONDARY GOAL: Race normally (handled by update())
                print(f"# SHIELDER: Following checkpoints", file=sys.stderr)
                # Keep default target from update() method
                pass

    def check_friendly_collision(self, other_pod):
        """Check if this pod should avoid collision with friendly pod"""
        if not other_pod or not other_pod.my_x:
            return False, 0
        
        # Distance between pods
        dist = math.hypot(other_pod.my_x - self.my_x, other_pod.my_y - self.my_y)
        
        # Predict future positions (3 turns ahead)
        my_future_x = self.my_x + self.my_vx * 3
        my_future_y = self.my_y + self.my_vy * 3
        other_future_x = other_pod.my_x + other_pod.my_vx * 3
        other_future_y = other_pod.my_y + other_pod.my_vy * 3
        
        future_dist = math.hypot(other_future_x - my_future_x, other_future_y - my_future_y)
        
        # Check if we're on collision course
        on_collision_course = future_dist < dist and dist < FRIENDLY_COLLISION_DISTANCE
        
        return on_collision_course or dist < FRIENDLY_COLLISION_DISTANCE, dist

    def avoid_friendly_collision(self, other_pod):
        """Adjust target and thrust to avoid collision with friendly pod"""
        is_collision_risk, dist = self.check_friendly_collision(other_pod)
        
        if not is_collision_risk:
            return  # No adjustment needed
        
        # Determine who is ahead (closer to their checkpoint)
        my_cp_x, my_cp_y = checkpoints[self.my_next_cp_id]
        my_dist_to_cp = math.hypot(my_cp_x - self.my_x, my_cp_y - self.my_y)
        
        other_cp_x, other_cp_y = checkpoints[other_pod.my_next_cp_id]
        other_dist_to_cp = math.hypot(other_cp_x - other_pod.my_x, other_cp_y - other_pod.my_y)
        
        # If same checkpoint, compare distances
        if self.my_next_cp_id == other_pod.my_next_cp_id:
            i_am_behind = my_dist_to_cp > other_dist_to_cp
        else:
            # Different checkpoints - use checkpoint ID as tiebreaker
            # Lower next checkpoint ID = ahead in race
            i_am_behind = self.my_next_cp_id < other_pod.my_next_cp_id
        
        if i_am_behind:
            print(f"# POD COLLISION AVOIDANCE: I'm behind, slowing down. Dist={int(dist)}", file=sys.stderr)
            
            # Slow down significantly
            if dist < FRIENDLY_COLLISION_SLOW_DISTANCE:
                # Very close - brake hard
                self.thrust = FRIENDLY_COLLISION_MIN_THRUST
                print(f"# HARD BRAKE: dist={int(dist)}", file=sys.stderr)
            else:
                # Reduce thrust proportionally
                reduction_factor = (dist - FRIENDLY_COLLISION_SLOW_DISTANCE) / (FRIENDLY_COLLISION_DISTANCE - FRIENDLY_COLLISION_SLOW_DISTANCE)
                reduction_factor = max(0, min(1, reduction_factor))
                
                if isinstance(self.thrust, str):  # Don't modify BOOST
                    pass
                else:
                    self.thrust = int(FRIENDLY_COLLISION_MIN_THRUST + (self.thrust - FRIENDLY_COLLISION_MIN_THRUST) * reduction_factor)
                    print(f"# GRADUAL SLOW: thrust={self.thrust}, factor={reduction_factor:.2f}", file=sys.stderr)
            
            # Adjust target slightly to the side to avoid direct collision
            # Calculate perpendicular offset
            dx = other_pod.my_x - self.my_x
            dy = other_pod.my_y - self.my_y
            if abs(dx) + abs(dy) > 0:
                # Perpendicular vector
                perp_x = -dy
                perp_y = dx
                norm = math.hypot(perp_x, perp_y)
                if norm > 0:
                    perp_x /= norm
                    perp_y /= norm
                    
                    # Offset target by 300 units perpendicular
                    self.target_x += perp_x * 300
                    self.target_y += perp_y * 300
        if oppenent_states:
            if self.strategy == CHARACTER.SHIELDER:
                # Pass both opponents to shielder logic
                self.updateShielderStrategy(oppenent_states)
            else:
                # Racer doesn't need opponent info
                pass

    def updateShielderStrategy(self, opponents):
        """Advanced shielder strategy with early game aggression"""
        if not self.racer_pod or len(opponents) < 2:
            return
        
        # EARLY GAME: Aggressive disturbance mode
        if self.turn_count <= SHIELDER_EARLY_GAME_TURNS:
            print(f"# SHIELDER: EARLY GAME DISTURBANCE MODE (Turn {self.turn_count})", file=sys.stderr)
            
            # Find closest opponent to disturb
            closest_opp = None
            min_dist = float('inf')
            
            for opp in opponents:
                x_2, y_2, vx_2, vy_2, angle_2, next_cp_id_2 = opp
                dist = math.hypot(x_2 - self.my_x, y_2 - self.my_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_opp = opp
            
            if closest_opp:
                x_2, y_2, vx_2, vy_2, angle_2, next_cp_id_2 = closest_opp
                
                # Target opponent's predicted position aggressively
                # Aim ahead of them to cut them off
                self.target_x = x_2 + vx_2 * 3
                self.target_y = y_2 + vy_2 * 3
                self.thrust = SHIELDER_BLOCK_THRUST
                print(f"# SHIELDER: Targeting opponent at ({int(x_2)}, {int(y_2)}) dist={int(min_dist)}", file=sys.stderr)
                return
        
        # MID/LATE GAME: Check for checkpoint denial opportunities
        for opp in opponents:
            x_2, y_2, vx_2, vy_2, angle_2, next_cp_id_2 = opp
            
            # Get opponent's next checkpoint
            opp_cp_x, opp_cp_y = checkpoints[next_cp_id_2]
            
            # Distance from opponent to their checkpoint
            opp_to_cp_dist = math.hypot(opp_cp_x - x_2, opp_cp_y - y_2)
            
            # Distance from shielder to opponent
            shielder_to_opp_dist = math.hypot(x_2 - self.my_x, y_2 - self.my_y)
            
            # Distance from shielder to opponent's checkpoint
            shielder_to_opp_cp = math.hypot(opp_cp_x - self.my_x, opp_cp_y - self.my_y)
            
            # Check if we're parallel (similar distance to checkpoint as opponent)
            distance_diff = abs(shielder_to_opp_cp - opp_to_cp_dist)
            
            # CHECKPOINT DENIAL: If parallel and opponent approaching checkpoint
            if (shielder_to_opp_dist < SHIELDER_PARALLEL_THRESHOLD and 
                distance_diff < SHIELDER_PARALLEL_THRESHOLD and
                opp_to_cp_dist < SHIELDER_CHECKPOINT_DENY_DISTANCE):
                
                print(f"# SHIELDER: CHECKPOINT DENIAL! Blocking opponent from CP", file=sys.stderr)
                print(f"# Opp dist to CP: {int(opp_to_cp_dist)}, Shielder-Opp dist: {int(shielder_to_opp_dist)}", file=sys.stderr)
                
                # Position between opponent and their checkpoint
                # Calculate interception point
                block_ratio = 0.7  # Position 70% toward checkpoint from opponent
                intercept_x = x_2 + (opp_cp_x - x_2) * block_ratio
                intercept_y = y_2 + (opp_cp_y - y_2) * block_ratio
                
                # Add velocity prediction to cut them off
                intercept_x += vx_2 * 2
                intercept_y += vy_2 * 2
                
                self.target_x = intercept_x
                self.target_y = intercept_y
                self.thrust = SHIELDER_BLOCK_THRUST
                return
        
        # DEFAULT: Protect racer mode
        self.updateOppenent(*opponents[0])  # Use existing protection logic

    def action(self):
        print(f"{int(self.target_x)} {int(self.target_y)} {self.thrust}")


# Two bots: racer + escort shielder
bots = [
    Pod(),
    Pod(strategy=CHARACTER.SHIELDER, boost_used=False)
]

# Link shielder to racer
bots[1].racer_pod = bots[0]

# ======================================
# GAME LOOP
# ======================================
while True:
    # Read our pods
    for i in range(2):
        x, y, vx, vy, angle, next_cp_id = [int(j) for j in input().split()]
        bots[i].update(x, y, vx, vy, angle, next_cp_id)

    # Read opponent pods
    opponent_states = []
    for i in range(2):
        x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2 = [int(j) for j in input().split()]
        bots[i].updateOppenent(x_2, y_2, vx_2, vy_2, angle_2, next_check_point_id_2)


    # Output actions
    for bot in bots:
        bot.action()