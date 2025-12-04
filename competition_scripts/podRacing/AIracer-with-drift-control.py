import sys
import math

boost_used = False
prev_x, prev_y = None, None
prev_vx, prev_vy = 0, 0
basespeed = 60
maxspeed = 100
while True:
    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
    opponent_x, opponent_y = [int(i) for i in input().split()]
    
    # Calculate velocity
    if prev_x is not None:
        vx = x - prev_x
        vy = y - prev_y
    else:
        vx, vy = 0, 0
    
    # Predict future position and compensate for drift
    predict_turns = 3
    target_x = next_checkpoint_x - vx * predict_turns
    target_y = next_checkpoint_y - vy * predict_turns
    
    # Thrust based on angle - reduce when turning sharply
    thrust = int(basespeed + (maxspeed- basespeed) * math.cos(math.radians(next_checkpoint_angle)))
    
    # Smart boost usage: straight line, far distance, aligned
    if not boost_used and abs(next_checkpoint_angle) < 10 and next_checkpoint_dist > 5000:
        thrust = "BOOST"
        boost_used = True
    
    # Check for potential collision with opponent
    dist_to_opponent = math.sqrt((opponent_x - x)**2 + (opponent_y - y)**2)
    if dist_to_opponent < 900:  # Close to collision range (400+400 + buffer)
        # Consider using SHIELD if you implement it
        pass
    
    # Update history
    prev_x, prev_y = x, y
    prev_vx, prev_vy = vx, vy
    
    print(f"{int(target_x)} {int(target_y)} {thrust}")