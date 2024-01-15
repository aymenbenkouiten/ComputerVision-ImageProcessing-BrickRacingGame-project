import cv2
import numpy as np
from Color import Color


def bitwise_not_custom(img):
    """
    Perform bitwise NOT operation on the input image using nested loops.

    Parameters:
    - img: Input image (numpy array).

    Returns:
    - Result of the bitwise NOT operation.
    """
    rows, cols = img.shape
    result = np.zeros_like(img, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            result[i, j] = 255 - img[i, j]  # 255 - pixel_value is equivalent to bitwise NOT

    return result
def bitwise_and_custom(img1, img2, mask):
    """
    Perform bitwise AND operation on two input images with a mask.

    Parameters:
    - img1: First input image (numpy array).
    - img2: Second input image (numpy array).
    - mask: Binary mask (numpy array).

    Returns:
    - Result of the bitwise AND operation with the mask.
    """
    # if img1.shape != img2.shape or img1.shape != mask.shape:
    #     raise ValueError("Input images and mask must have the same shape.")

    rows, cols = img1.shape[:2]
    result = np.zeros_like(img1, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if mask[i, j] != 0:
                result[i, j] = img1[i, j] & img2[i, j]

    return result

class advanced():
   
    @staticmethod
    def remove_green_screen(image,lower = np.array([20,50, 100]),upper = np.array([35,255,255])):
      
        if image is None:
            print("Error: Unable to read the image.")
            return None
        
        #image=Color.resize_frame(image,0.6)

        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask for the green screen color range,boundaries must be in hsv
        # mask = cv2.inRange(hsv_image, lower, upper)
        mask = Color.my_in_range(hsv_image, lower, upper)
   
        # Invert the mask to keep the non-green areas
        # inverted_mask = cv2.bitwise_not(mask)
        inverted_mask = bitwise_not_custom(mask)
    
        # Create a black background image
        black_background = np.zeros_like(image)
    
        # Replace the green screen area with the replacement color
        # result = cv2.bitwise_and(image, image, mask=inverted_mask)
        # result += cv2.bitwise_and(black_background, black_background, mask=mask)
        result =bitwise_and_custom(image, image, mask=inverted_mask)
        result += bitwise_and_custom(black_background, black_background, mask=mask)
        # cv2.circle(result,(300,250),20,(0,255,0),5)
        # print(hsv_image[300,250])
        return result
   
    
    @staticmethod
    
    # def invisibility_cloak_frame(frame,background_path="background/background.jpg",lower = np.array([95,30,130]),upper = np.array([120,120,170])): #grey
    def invisibility_cloak_frame(frame,background_path="background/background.jpg",lower = np.array([20,50, 100]),upper = np.array([35,255,255])):
    # def invisibility_cloak_frame(frame,background_path="background/background.jpg",lower = np.array([120,10,160]),upper = np.array([170,120,245])):
        #frame=Color.resize_frame(frame,0.2)
        # Convert the frame from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
        # Create a mask for the red color range
        # mask = cv2.inRange(hsv_frame, lower, upper) #OPENCV
        mask = Color.my_in_range(hsv_frame, lower, upper)
        
        # cv2.imshow("mask",mask)
        # cv2.imwrite("mask.jpg",mask)
        
        # Invert the mask to keep the non-red areas
        # inverted_mask = cv2.bitwise_not(mask)
        inverted_mask = bitwise_not_custom(mask)
 
        # Replace the cloak color with the background (assuming a static background)
        background = cv2.imread(background_path)  # Replace with the actual path
        # result = cv2.bitwise_and(frame, frame, mask=inverted_mask)
        # result += cv2.bitwise_and(background, background, mask=mask)
        result =bitwise_and_custom(frame, frame, mask=inverted_mask)
        result += bitwise_and_custom(background, background, mask=mask)
        #Uncomment to set the color
        # cv2.circle(result,(300,250),20,(0,255,0),5)
        # print(hsv_frame[300,250])
        
        return result
    
    @staticmethod
    def main(choice="Green screen"):
        
        if choice.lower == "Green screen":
            method = advanced.remove_green_screen
        else:
            method = advanced.invisibility_cloak_frame    
        
        # cap = cv2.VideoCapture(video_path)
        cap = cv2.VideoCapture(0)
    
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return
        
        while True:
            ret, frame = cap.read()
    
            if not ret:
                break
    
            result = method(frame)
            # Display the result
            cv2.imshow(choice, result)
    
            if cv2.waitKey(30) & 0xFF == ord("q"):  # Press 'Esc' to exit
                break
    
        cap.release()
        cv2.destroyAllWindows()
           
    
if __name__=="__main__":
    
  
    advanced.main(choice="invisibilty")
   
    