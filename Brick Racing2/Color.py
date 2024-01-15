import cv2
import numpy as np



# Define a resize factor to reduce the image size
resize_factor = 0.2
inverse_resize_factor = 1 / resize_factor
lo=np.array([100,150,60])
hi=np.array([140,255,150])
class Color():
 
    # Resize the captured frame before grayscale conversion
    @staticmethod
    def resize_frame(frame,factor):
        h, w = int(len(frame) * factor), int(len(frame[0]) * factor)
        resized_image = np.zeros((h,w,3),frame.dtype)
    
        for i in range(h):
            for j in range(w):
                original_i = int(i / factor)
                original_j = int(j / factor)
                resized_image[i,j] = frame[original_i][original_j]
        return resized_image
    @staticmethod
    def my_in_range(frame, down=np.array([100,150,60]), up=np.array([140,255,150])):
        height, width = len(frame), len(frame[0])
        mask=np.zeros((frame.shape[0],frame.shape[1]),frame.dtype)
    
        for i in range(height):
            for j in range(width):
                if down[0] <= frame[i][j][0] <= up[0] and down[1] <= frame[i][j][1] <= up[1] and down[2] <= frame[i][j][2] <= up[2]:
                    mask[i][j] = 255
    
        return mask
    @staticmethod
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
                    if len(contours)==0 or Color.my_contour_area(contours[0])>Color.my_contour_area(contour):
                        contours.append(contour)
                    else:
                        contours.insert(0,contour)
        return contours
    @staticmethod
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
    @staticmethod
    def distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    @staticmethod
    def my_min_enclosing_circle(points, max_iterations=100):
        center = points[0]
    
        for _ in range(max_iterations):
            radius = max(Color.distance(center, point) for point in points)
            for point in points:
                if Color.distance(center, point) > radius:
                    for i in range(2):
                        center[i] += 0.1 * (point[i] - center[i])
        return center, radius
    @staticmethod
    def my_detect_inrange(image, surfaceMin, surfaceMax):
        points = []
        image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask = Color.my_in_range(image, lo, hi)
        elements = Color.my_find_contours(mask)
    
        for element in elements:
            if Color.my_contour_area(element) > surfaceMin*resize_factor and Color.my_contour_area(element) < surfaceMax*resize_factor:
                ((x, y), rayon) = Color.my_min_enclosing_circle(element)
                points.append(np.array([int(x*inverse_resize_factor), int(y*inverse_resize_factor), int(rayon*inverse_resize_factor), int(Color.my_contour_area(element)*inverse_resize_factor)]))
        return image, mask, points
    
    
    @staticmethod  
    def detect_color(frame):
            
            small_frame=Color.resize_frame(frame,resize_factor)
            image,mask,points= Color.my_detect_inrange(small_frame,100,10000)
            if points :
                cv2.circle(frame,(points[0][0],points[0][1]),points[0][2],(0,0,255),2)
                cv2.putText(frame,str(points[0][3]),(points[0][0],points[0][1]),
                cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            # if mask is not None:
                # cv2.imshow("mask",mask)
                # cv2.imshow("captured_frame",frame)
                # cv2.imshow("image",image)    
                
            return frame 
        
if __name__=="__main__":
    
    #change to zero for default cam
    VideoCap = cv2.VideoCapture(0)
    
    lo=np.array([100,150,60])
    # lo= np.array([120,10,160])
    hi=np.array([140,255,150])
    # hi=np.array([170,120,245])
    
    while True:
        
        ret, captured_frame = VideoCap.read()
        
        image,mask,points= Color.detect_color(captured_frame)
        
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break
    
    VideoCap.release()
    cv2.destroyAllWindows()