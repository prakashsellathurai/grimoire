import sys
import math

# Pod Racing AI Strategy:
# Simple drift compensation and variable thrust logic outperforms
# complex PID control logic used in early versions.
# BOOST mechanic is available but not yet implemented.

# State tracking variables
boost_used = False  # Flag to track if boost has been used (one-time use)
prev_x, prev_y = None, None  # Previous position for velocity calculation
prev_vx, prev_vy = 0, 0  # Previous velocity components
basespeed = 60  # Base thrust speed
maxspeed = 100  # Maximum thrust speed
while True:
    # Read input: pod position, next checkpoint info, opponent position
    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]
    
    # Calculate current velocity from position change
    if prev_x is not None:
        vx = x - prev_x
        vy = y - prev_y
    else:
        vx, vy = 0, 0
    
    # Predict future position and compensate for drift
    # Look ahead 3 turns to account for momentum and trajectory
    predict_turns = 3
    target_x = next_checkpoint_x - vx * predict_turns
    target_y = next_checkpoint_y - vy * predict_turns
    
    # Calculate thrust based on turn angle - reduce when turning sharply
    # Use cosine to scale thrust from basespeed to maxspeed based on alignment
    thrust = int(basespeed + (maxspeed- basespeed) * math.cos(math.radians(next_checkpoint_angle)))
    
    # Smart boost usage: only use when straight line, far distance, and well aligned
    if not boost_used and abs(next_checkpoint_angle) < 10 and next_checkpoint_dist > 5000:
        thrust = "BOOST"
        boost_used = True
    
    # Check for potential collision with opponent and defend if needed
    dist_to_opponent = math.sqrt((opponent_x - x)**2 + (opponent_y - y)**2)
    if dist_to_opponent < 900:  # Close to collision range (400+400 + buffer)
        # TODO: Consider using SHIELD if you implement it
        pass
    
    # Store current state for next iteration's velocity calculation
    prev_x, prev_y = x, y
    prev_vx, prev_vy = vx, vy
    
    # Output: target position and thrust
    print(f"{int(target_x)} {int(target_y)} {thrust}")