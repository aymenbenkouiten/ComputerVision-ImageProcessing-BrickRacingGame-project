import cv2
import numpy as np
import pickle
import streamlit as st
from KalmanFilter import KalmanFilter
from Color import Color

KF=KalmanFilter(0.1, [10, 10])
lo=np.array([105,80,60])
hi=np.array([120,255,150])
car_color = (0, 0, 255)  # Blue in BGR format
# Define black image window size
black_window_size = (500, 500,3)
black_window = np.zeros(black_window_size, dtype=np.uint8)
# Define initial positions for the car
car_position=(250, 450)
#car size 
w=30
#parametre generation d'obstacle 
gen_proba=0.020
num_obstacle=3
# Define obstacle properties
obstacle_radius = 10
obstacle_color = (51, 250, 250)  # Red color in BGR
obstacle_speed = 2

# Minimum and maximum y-coordinates for obstacle generation
min_y = 0
max_y = black_window.shape[0] - obstacle_radius * 2

# List to store active obstacles
obstacle_centers = []

#initialize game score
game_score=0  # keeps track of the score

#parametre mouvement avec vitesse
posX=250
posY=450
pas=10
vitesse=100

# Define a resize factor to reduce the image size
resize_factor = 0.2
inverse_resize_factor = 1 / resize_factor


"""Color detection code """





"""score and leaderboard managment """


def save_scores(scores):
  with open("leaderboard.pickle", "wb") as f:
    pickle.dump(scores, f)

def read_scores():
  try:
    with open("leaderboard.pickle", "rb") as f:
      return pickle.load(f)
  except FileNotFoundError:
    return []

def update_leaderboard(score):
  scores = read_scores()
  if (len(scores)>0):
    if score > max(scores):
      print("New high score")
  scores.append(score)
  scores.sort(reverse=True)
  scores = scores[:5]  # Keep only the top 5 scores
  save_scores(scores)

def display_leaderboard():
  scores = read_scores()
  s=["Leaderboard: "]
  print("Leaderboard:")
  for i, score in enumerate(scores, 1):
    print(f"{i}. {score}")
    s.append(f"{i}. {score}")
  return "\n".join(s)









"""Obstacle generation and mouvment code"""

def generate_obstacle_poisson(n, min_distance,height,old_centers):
  """
  Generates n obstacles with minimum distance between them using basic Poisson disc sampling.

  Args:
    n: Number of obstacles to generate.
    min_distance: Minimum distance between obstacles.
    width: Width of the playable area.
    height: Height of the playable area.

  Returns:
    List of obstacle centers (x, y) coordinates.
  """

  obstacle_centers = []
  active_list = []

  # Generate initial random point
  y = min_distance
  x = np.random.uniform(min_distance, height - min_distance)
  active_list.append((x, y))
  obstacle_centers.append((x, y))
  max_time=50
  i=0
  while len(obstacle_centers) < n and i<max_time:
    # Choose a random active point
    active_point = active_list[np.random.randint(len(active_list))]

    # Generate random candidate points in annulus
    r_min = min_distance
    r_max = 2 * min_distance
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(r_min, r_max)

    candidate_y = active_point[1] 
    candidate_x = active_point[0] + r * np.sin(theta)
    if(candidate_x>height - min_distance or candidate_x<min_distance):
       continue
    # Check for collisions with existing obstacles
    is_valid = True
    for existing_x, existing_y in obstacle_centers:
      distance = np.sqrt(((candidate_x - existing_x) ** 2) + ((candidate_y - existing_y) ** 2))
      if distance < min_distance+15 :
        is_valid = False
        break
    for old_ob in old_centers:
      distance = np.sqrt(((candidate_x - old_ob[0]) ** 2) + ((candidate_y - old_ob[1]) ** 2))
      if distance < min_distance+15 :
        is_valid = False
        break

    if is_valid:
      obstacle_centers.append((candidate_x, candidate_y))
      active_list.append((candidate_x, candidate_y))
    i+=1

  return obstacle_centers






def generate_obstacle(n):

# Generate random y-coordinate within limits
   index=min(obstacle_speed*2,len(obstacle_centers))
   old_list=obstacle_centers[-index:]
   candidates=generate_obstacle_poisson(n,obstacle_radius,black_window.shape[1],old_list)
   for c in candidates:
       if np.random.rand() < gen_proba:
           # Append new obstacle center to list
           point=[]
           point.append(int(c[0]))
           point.append(c[1])
           obstacle_centers.append(point)

def update_obstacles():
  global obstacle_centers
  global game_score

  for i, center in enumerate(obstacle_centers):
    global obstacle_speed
    global num_obstacle
    global gen_proba
    # Update y-coordinate based on speed
    center[1] += obstacle_speed

    # Check if obstacle reaches the bottom
    if center[1]+obstacle_radius > black_window.shape[0]:
      cv2.circle(black_window, (center[0],center[1]), obstacle_radius+obstacle_speed, [0,0,0], -1)
      del obstacle_centers[i]
      game_score+=1
      
      if game_score % 20 ==0 and gen_proba<0.2 :
         gen_proba+=0.004
      if game_score % 5 ==0 and num_obstacle<10 :
        num_obstacle+=1
      if game_score % 10 ==0  and obstacle_speed<15:
        obstacle_speed+=1

def draw_obstacles(black_window):
  for center in obstacle_centers:
    cv2.circle(black_window, (center[0],center[1]-obstacle_speed), obstacle_radius, [0,0,0], -1)
    cv2.circle(black_window, tuple(center), obstacle_radius, obstacle_color, -1)


def game(cam_stream,game_stream,score,Leader,car_position=(250, 450),car_color = (0, 0, 255),num_obstacle=3,posX=250,posY=450,pas=10,black_window_size = (500, 500,3),resize_factor = 0.2):
    #initialize  leaderboard
    
    black_window = np.zeros(black_window_size, dtype=np.uint8)
    Leader.text(display_leaderboard())   #contains at the end of the execution the leadrboard 
    
    cap=cv2.VideoCapture(0)
    #intial car position 
    cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), car_color, -1)
    while(True):
        ret,frame=cap.read()
        cv2.flip(frame,1,frame)
        small_frame=Color.resize_frame(frame,resize_factor)
        image,mask,points=Color.my_detect_inrange(small_frame,20,8000)
        # etat=KF.predict().astype(np.int32)  #pridict position with kalman filter
    
        # obstacles update
        generate_obstacle(num_obstacle)
        update_obstacles()
        score.text(f'SCORE : {game_score}')
        draw_obstacles(black_window)
    
        # Check for collisions with obstacles
        collision_detected=False

        for obstacle_center in obstacle_centers:
            obstacle_rect = ((obstacle_center[0] - obstacle_radius, obstacle_center[1] - obstacle_radius),
                             (2 * obstacle_radius, 2 * obstacle_radius))
            car_rect = ((car_position[0], car_position[1]), (30, 30))
    
            # Check for overlap between car and obstacle rectangles
            if (
                obstacle_rect[0][0] < car_rect[0][0] + car_rect[1][0] and
                obstacle_rect[0][0] + obstacle_rect[1][0] > car_rect[0][0] and
                obstacle_rect[0][1] < car_rect[0][1] + car_rect[1][1] and
                obstacle_rect[0][1] + obstacle_rect[1][1] > car_rect[0][1]
            ):
                print("Collision detected!")
                # You can add code here to handle the collision, such as stopping the game or reducing the player's score.
                collision_detected = True  # Set the flag to True
                update_leaderboard(game_score)
                
        

                cv2.putText(frame,f"GameOver", ((frame.shape[0]//2)-50, (frame.shape[1]//2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0, 255), 4)
                cam_stream.image(frame,channels="BGR")
                
                # print(f'your score is: {game_score}')
                score.text(f'SCORE : {game_score}')
                Leader.text(display_leaderboard())
                # display_leaderboard()
    
                break  # Exit the inner loop
    
        if collision_detected:
            break  # Exit the outer loop
    
    
        #detect color and move car 
    
        if len(points)>0 :
            #show detected object
            cv2.circle(frame,(points[0][0],points[0][1]),points[0][2],(0,0,255),2)
            cv2.putText(frame,str(points[0][3]),(points[0][0],points[0][1]),
            cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    
    
            if(abs(points[0][0]-(posX+w/2))>pas/2):
                cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), (0,0,0), -1)
                if points[0][0]>posX+w/2 and posX+w+pas<(black_window.shape[0]):
                    posX+=pas
                else:
                    if posX-pas>0:
                        posX-=pas
    
                     # Calculate the corresponding position in the black image window
            
            car_position = tuple([posX,posY])
            # Draw a rectangle representing the car on the black window
            cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), car_color, -1)
    
    
    
        # if mask is not None:
        #     cv2.imshow("mask",mask) # the mask 
    
        game_stream.image(black_window,channels="BGR")# the game window
        cam_stream.image(frame,channels="BGR")
        # cv2.imshow("jeux",black_window) # the game window
        # cv2.imshow("image",frame)  #the image with the object detected
        
    
    #keyboard movment 
        key = cv2.waitKey(20) & 0xFF
        # Check for arrow keys
        if key == ord('d'):
          if car_position[0] +30+w < black_window.shape[1]:
             posX+=w 
             cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), (0,0,0), -1)
             car_position = tuple([posX,posY])
             # Draw a rectangle representing the car on the black window
             cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), car_color, -1)
          # Move left
        elif key == ord('a'):
          if car_position[0] -30-w > 0:
             posX-=w
             cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), (0,0,0), -1)
             car_position = tuple([posX,posY])
             # Draw a rectangle representing the car on the black window
             cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), car_color, -1)
    
    
    
    
        if cv2.waitKey(20)&0xFF==ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
 

 
    
if __name__=="__main__":
     
     game()    
     
     