import streamlit as st
import cv2
import numpy as np
from Filters import Filters
from Metrics import Metrics
from advanced import advanced
from Color import Color
from game import game
from game_kalman import game_kalman
import time


# Mock data

# img_default = cv2.imread("images/algeria.png",cv2.IMREAD_GRAYSCALE) 
img_default = np.zeros((500,500,1))

def apply_filter(img,filter,kernel_size,kernel_shape,iterations=1):
 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if filter.lower() == "mean":
        return Filters.filtreMoy(img,vois=kernel_size)
    if filter.lower() == "median":
        return Filters.filtreMed(img,vois=kernel_size) 
    if filter.lower() == "sobel":
        return Filters.Sobel(img,kernel_size=kernel_size)
    if filter.lower() == "dilatation":
        return Filters.filter_Morph(img,iterations=iterations,type="dilate",kernel_shape=kernel_shape.lower(),size_kernel=kernel_size)
    if filter.lower() == "erosion":
        return Filters.filter_Morph(img,iterations=iterations,type="erosion",kernel_shape=kernel_shape.lower(),size_kernel=kernel_size)   
    if filter.lower() == "open":
        return Filters.Open_Close_Morph(img,type="open",iterations=iterations,size_kernel=kernel_size)
    if filter.lower() == "close":
        return Filters.Open_Close_Morph(img,type="close",iterations=iterations,size_kernel=kernel_size,kernel_shape=kernel_shape.lower())
    #else
    return Filters.filter2D(img , kernel = Metrics.generate_filter_kernel(filter_type=filter.lower(),size=kernel_size))
  

# Function for the main camera streaming aka advanced
def main(choice="Green screen"):

        global stream_out
        
        #which function to select
        if choice.lower() == "green screen":
                method = advanced.remove_green_screen
        elif choice.lower() == "invisibility cloak":
                method = advanced.invisibility_cloak_frame
        else :
                method= Color.detect_color
        
        cap = cv2.VideoCapture(0)
    
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            cv2.flip(frame,1,frame)
            result = method(frame)
             
            stream_out.image(result,channels="BGR") #update stream
            
            if cv2.waitKey(30) & 0xFF == ord("q"):  # Press 'q' to exit
                break
    
        cap.release()
        cv2.destroyAllWindows() 
        
def background_image():
#set background for invisibilty cloak
        global stream_out
        cap = cv2.VideoCapture(0)
    
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return
        number = 4
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result =frame.copy()
            
            time.sleep(1)
            number -= 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = ((frame.shape[0]//2)+50, (frame.shape[1]//2))  
            font_scale = 4
            font_color = (255, 255, 255)  # White color in BGR
            line_thickness = 4

            cv2.putText(frame,f"{number}", position, font, font_scale, font_color, line_thickness)
            stream_out.image(frame,channels="BGR")
            
            
            if number == 0:
                copy_result =result.copy()
                cv2.putText(copy_result,"Cheezz", ((frame.shape[0]//2)-100, (frame.shape[1]//2)-50 ), font, font_scale, (0,255, 255) , line_thickness)
                stream_out.image(copy_result,channels="BGR")
                time.sleep(0.5)
                
                cv2.imwrite("background/background.jpg",result) 
                stream_out.image(result,channels="BGR")
                break
    
        cap.release()
        cv2.destroyAllWindows() 
  
######UI##    
    
st.set_page_config(page_title="Bricks Racing", page_icon="📷", layout="wide", initial_sidebar_state="expanded")

# Filters Tab
with st.expander("Filters"):
    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_contents = uploaded_file.read()
        np_array = np.frombuffer(file_contents, np.uint8)
        # Decode the NumPy array into an image using OpenCV
        img_array = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    else:
        # Use the default mock image if no image is uploaded
        img_array = img_default
        
    col1, col2 ,col3= st.columns(3)
        
    input_image = col1.image(img_array, caption="Input image")
    filter_type = col2.selectbox("Choose Filter", ["Mean", "Median", "Gaussian", "Laplacian", "Sobel", "Emboss", "Dilatation", "Erosion", "Open", "Close"], index=0)
    kernel_size = col2.number_input("Kernel Size", min_value=1, max_value=20, value=1)
    kernel_shape = col2.selectbox("Choose Kernel Shape", ["Rect", "Cross"], index=0)
    iterations = col2.number_input("Iterations", min_value=1, max_value=10, value=1)
    filtered_image = col3.image(np.zeros_like(img_default), caption="Result image")
    if col2.button("Generate"):
        filtered_image.image(apply_filter(img_array, filter_type, kernel_size, kernel_shape, iterations))

# Advanced Tab
with st.expander("Advanced"):
    col1, col2= st.columns(2)
    stream_out = col1.image(np.zeros_like(img_default), caption="Stream")
    method_choice = col2.radio("Method", ["Color detection", "Green screen", "Invisibility Cloak"], index=2)
    if col2.button("Background"):
        background_image()
    if col2.button("Start"):
        main(method_choice)

# Game Tab
with st.expander("Game"):
    
    col1, col2 ,col3= st.columns(3)
    
    game_stream = col1.image(np.zeros((256, 256, 3), dtype=np.uint8), caption="Game")
    game_cam_stream = col2.image(np.zeros((256, 256, 3), dtype=np.uint8), caption="Camera Stream")
    score = col3.text("SCORE : ")
    Leader = col3.text("Leader : ")
    high_score = col3.text("\n\n")
    # Kalman_filter = st.checkbox(label="Kalman",value=False)
    if col3.button("Start Game"):
        
        # game(game_cam_stream,game_stream,score,Leader)
        game_kalman(game_cam_stream,game_stream,score,Leader,high_score)
