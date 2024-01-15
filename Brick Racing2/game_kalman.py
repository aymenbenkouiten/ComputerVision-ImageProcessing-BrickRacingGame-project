import cv2
import numpy as np
import pickle
from KalmanFilter import KalmanFilter
from Color import Color

#initialisation du filtre de kalman
KF=KalmanFilter(0.1, [10, 10])

lo=np.array([105,80,60])
hi=np.array([120,255,150])
#lo = np.array([150,95,215])
#hi = np.array([180,130,260])


def init():
  
  global car_color   # Blue in BGR format
  # Define black image window size
  global black_window_size 
  global black_window
  # Define initial positions for the car
  global car_position 
  global w
  #parametre generation d'obstacle 
  global gen_proba
  global num_obstacle
  # Define obstacle properties
  global obstacle_radius
  global obstacle_color   # Red color in BGR
  global obstacle_speed 
  
  # Minimum and maximum y-coordinates for obstacle generation
  global min_y 
  global max_y 
  # List to store active obstacles
  global obstacle_centers
  
  #initialize game score
  global game_score  # keeps track of the score
  
  #parametre mouvement avec vitesse
  global posX
  global posY
  global pas
  global vitesse

  # Define a resize factor to reduce the image size
  global resize_factor 
  global inverse_resize_factor 

  car_color = (0, 0, 255)  # Blue in BGR format
  # Define black image window size
  black_window_size = (500, 500,3)
  black_window = np.zeros(black_window_size, dtype=np.uint8)
  # Define initial positions for the car
  car_position=(250, 450)
  #car size s
  w=30
  #parametre generation d'obstacle 
  gen_proba=0.04
  num_obstacle=3
  # Define obstacle properties
  obstacle_radius = 10
  obstacle_color = (51, 250, 250)  # Red color in BGR
  obstacle_speed = 4
  
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

# Resize the captured frame before grayscale conversion
def resize_frame(frame,factor):
    h, w = int(len(frame) * factor), int(len(frame[0]) * factor)
    resized_image = np.zeros((h,w,3),frame.dtype)

    for i in range(h):
        for j in range(w):
            original_i = int(i / factor)
            original_j = int(j / factor)
            resized_image[i,j] = frame[original_i][original_j]
    return resized_image

def my_in_range(frame, down, up):
    height, width = len(frame), len(frame[0])
    mask=np.zeros((frame.shape[0],frame.shape[1]),frame.dtype)

    for i in range(height):
        for j in range(width):
            if down[0] <= frame[i][j][0] <= up[0] and down[1] <= frame[i][j][1] <= up[1] and down[2] <= frame[i][j][2] <= up[2]:
                mask[i][j] = 255

    return mask

def my_find_contours(frame):
    contours = []
    visited = set()

    def is_valid(x, y):
        return 0 <= x < len(frame) and 0 <= y < len(frame[0])

    def dfs(x, y):
        stack = [(x, y)]
        current_contour = []

        while stack:
            current_x, current_y = stack.pop()

            if (current_x, current_y) not in visited and is_valid(current_x, current_y) and np.all(frame[current_x, current_y] >= 140):
                visited.add((current_x, current_y))

                #IMPORTAAAAAAAAAANT 
                current_contour.append((current_y, current_x))

                stack.extend(
                    [(current_x + dx, current_y + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
                )

        return current_contour

    for x in range(len(frame)):
        for y in range(len(frame[0])):
            if (x, y) not in visited and frame[x, y] >= 140:
                contour = dfs(x, y)
                if len(contours)==0 or my_contour_area(contours[0])>my_contour_area(contour):
                    contours.append(contour)
                else:
                    contours.insert(0,contour)
    return contours

def my_contour_area(contour):
    area = 0
    for i in range(len(contour)):
        x = contour[i][0]
        y = contour[i][1] 
        next_x = contour[(i + 1) % len(contour)][0]
        next_y = contour[(i + 1) % len(contour)][1]
        area += x * next_y - next_x * y

    area = 0.5 * abs(area)
    return area

def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def my_min_enclosing_circle(points, max_iterations=100):
    center = points[0]

    for _ in range(max_iterations):
        radius = max(distance(center, point) for point in points)
        for point in points:
            if distance(center, point) > radius:
                for i in range(2):
                    center[i] += 0.1 * (point[i] - center[i])
    return center, radius

def my_detect_inrange(image, surfaceMin, surfaceMax):
    points = []
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = my_in_range(image, lo, hi)
    elements = my_find_contours(mask)

    for element in elements:
        if my_contour_area(element) > surfaceMin*resize_factor and my_contour_area(element) < surfaceMax*resize_factor:
            ((x, y), rayon) = my_min_enclosing_circle(element)
            points.append(np.array([int(x*inverse_resize_factor), int(y*inverse_resize_factor), int(rayon*inverse_resize_factor), int(my_contour_area(element)*inverse_resize_factor)]))
    return image, mask, points



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
      scores.append(score)
      scores.sort(reverse=True)
      scores = scores[:5]  # Keep only the top 5 scores
      save_scores(scores)
      print("New high score")
      return "New high score"
  return ""  
      
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

  # global obstacle_centers
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
   global obstacle_centers
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
  # global black_window

  for i, center in enumerate(obstacle_centers):
    global obstacle_speed
    global num_obstacle
    global gen_proba
    # Update y-coordinate based on speed
    center[1] += obstacle_speed

    # Check if obstacle reaches the bottom
    if center[1] > black_window.shape[0]:
      
      cv2.circle(black_window, (center[0],center[1]), obstacle_radius+obstacle_speed, [0,0,0], -1)
      # cv2.circle(black_window, (center[0],center[1]), obstacle_radius, [0,0,0], -1)
      
      del obstacle_centers[i]
      
      game_score+=1
      if game_score % 20 ==0 and gen_proba<0.2 :
         gen_proba+=0.004
      if game_score % 5 ==0 and num_obstacle<10 :
        num_obstacle+=1
      if game_score % 10 ==0  and obstacle_speed<15:
        obstacle_speed+=1

def draw_obstacles(black_window):
  global obstacle_centers
  
  for center in obstacle_centers:
    cv2.circle(black_window, (center[0],center[1]-obstacle_speed), obstacle_radius, [0,0,0], -1)
    cv2.circle(black_window, tuple(center), obstacle_radius, obstacle_color, -1)

    
def game_kalman(cam_stream,game_stream,score,Leader,high_score,car_position=(250, 450),car_color = (0, 0, 255),num_obstacle=3,posX=250,posY=450,pas=10,black_window_size = (500, 500,3),resize_factor = 0.2):
    
   init() 
   
   Leader.text(display_leaderboard())   #contains at the end of the execution the leadrboard
   cap=cv2.VideoCapture(0)
   #intial car position 
   cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), car_color, -1)
   while(True):
       ret,frame=cap.read()
       cv2.flip(frame,1,frame)
       small_frame=resize_frame(frame,resize_factor)
       image,mask,points=my_detect_inrange(small_frame,20,8000)
       etat=KF.predict().astype(np.int32)  #pridict position with kalman filter
       
       # obstacles update
   
       generate_obstacle(num_obstacle)
       update_obstacles()
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
               cv2.putText(frame,f"GameOver", ((frame.shape[0]//2)-50, (frame.shape[1]//2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0, 255), 4)
               cam_stream.image(frame,channels="BGR")
                
                # cv2.imshow("image",frame)
               new_high_score =update_leaderboard(game_score) 
               if len(new_high_score)>1:
                  high_score.text(new_high_score)
               # print(f'your score is: {game_score}')
               score.text(f'SCORE : {game_score}')
               Leader.text(display_leaderboard())
               display_leaderboard()
    
                
               break  # Exit the inner loop
   
       if collision_detected:
           break  # Exit the outer loop
   
   
       #if position detected mouve car according to position and update kaman filter 
       if len(points)>0 :
           KF.update(np.expand_dims(points[0][:1], axis=-1))
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
   
       #no object detected use the kalman filter prediction to move the car
       else:
           if(abs(etat[0]-(posX+w/2))>pas/2):
               cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), (0,0,0), -1)
               if etat[0]>posX+w/2 and posX+w+pas<(black_window.shape[0]):
                   posX+=pas
               else:
                   if posX-pas>0:
                       posX-=pas
   
                    # Calculate the corresponding position in the black image window
           
           car_position = tuple([posX,posY])
           # Draw a rectangle representing the car on the black window
           cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), car_color, -1)
   
       
       game_stream.image(black_window,channels="BGR")# the game window
       cam_stream.image(frame,channels="BGR")
      #  cv2.imshow("jeux",black_window)
      #  cv2.imshow("image",frame)
       
   
       if cv2.waitKey(20)&0xFF==ord('q'):
           break
   
   cap.release()
   cv2.destroyAllWindows()  
   
if __name__ == "__main__":
    
   init() 
   leader_board=read_scores()
   cap=cv2.VideoCapture(0)
   #intial car position 
   cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), car_color, -1)
   while(True):
       ret,frame=cap.read()
       cv2.flip(frame,1,frame)
       small_frame=resize_frame(frame,resize_factor)
       image,mask,points=my_detect_inrange(small_frame,20,8000)
       etat=KF.predict().astype(np.int32)  #pridict position with kalman filter
       
       # obstacles update
   
       generate_obstacle(num_obstacle)
       update_obstacles()
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
               print(f'your score is: {game_score}')
               display_leaderboard()
   
               break  # Exit the inner loop
   
       if collision_detected:
           break  # Exit the outer loop
   
   
       #if position detected mouve car according to position and update kaman filter 
       if len(points)>0 :
           KF.update(np.expand_dims(points[0][:1], axis=-1))
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
   
       #no object detected use the kalman filter prediction to move the car
       else:
           if(abs(etat[0]-(posX+w/2))>pas/2):
               cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), (0,0,0), -1)
               if etat[0]>posX+w/2 and posX+w+pas<(black_window.shape[0]):
                   posX+=pas
               else:
                   if posX-pas>0:
                       posX-=pas
   
                    # Calculate the corresponding position in the black image window
           
           car_position = tuple([posX,posY])
           # Draw a rectangle representing the car on the black window
           cv2.rectangle(black_window, (car_position[0], car_position[1]), (car_position[0] + 30, car_position[1] + 30), car_color, -1)
   
       
   
       cv2.imshow("jeux",black_window)
       cv2.imshow("image",frame)
       
   
       if cv2.waitKey(int(pas*1000/vitesse))&0xFF==ord('q'):
           break
   
   cap.release()
   cv2.destroyAllWindows()