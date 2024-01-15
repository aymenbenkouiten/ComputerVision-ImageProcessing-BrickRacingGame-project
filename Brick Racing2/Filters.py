# median and mean
# gauss laplacien
# erosion et dilatation aka morphologiques
import copy
import numpy as np
import cv2
from Metrics import Metrics
from convolution import Convolution
from time import time

class Filters():

        
    @staticmethod
    def filtreMoy(img, vois=7):
        
        img_height, img_width = img.shape
        pad= vois // 2
        
        #Add padding
        img_copy = np.zeros((img_height + pad + pad, img_width + pad + pad), dtype=img.dtype)
        img_copy[pad:pad+img_height, pad:pad+img_width] = img
        
        img_result = np.zeros(img_copy.shape, img_copy.dtype)
        
        img_height_new = img_height+pad+pad
        img_width_new = img_width+pad+pad

        for y in range(img_height_new):   
            for x in range(img_width_new):

                if(y < (pad) or y > (img_height_new - pad) or x < (pad) or x > (img_width_new - pad)):
                    img_result[y, x] = img_copy[y, x]
                else:
                    n = vois//2
                    imgvois = img_copy[int(y - n):int(y + n + 1), int(x - n):int(x+n+1)]
                    # imgtoy[y, x] = np.mean(imgvois)
                    img_result[y, x] = Metrics.mean(imgvois)
               
            # if (y%10 == 0):
                # print(y)
        #crop back to original
        img_result = img_result[pad:pad+img_height, pad:pad+img_width]                
        return img_result
    
    
    @staticmethod
    def filtreMed(img, vois=7):
        
        img_height, img_width = img.shape
        pad= vois // 2
        
        #Add padding
        img_copy = np.ones((img_height + pad + pad, img_width + pad + pad), dtype=img.dtype)
        img_copy[pad:pad+img_height, pad:pad+img_width] = img
        
        img_result = np.ones(img_copy.shape, img_copy.dtype)
        
        img_height_new = img_height+pad+pad
        img_width_new = img_width+pad+pad
        
    
        for y in range(img_height_new):
            for x in range(img_width_new):
                if(y < pad or y > (img_height_new - pad) or x < pad or x > (img_width_new - pad)):
                    img_result[y, x] = img_copy[y, x]
                else:
                    n = pad
                    imgvois = img_copy[int(y - n):int(y + n + 1), int(x - n):int(x+n+1)]

                    #get voisinage
                    t = np.zeros((vois*vois), img.dtype)
                    hv,wv=imgvois.shape

                    for yv in range(hv):
                        for xv in range(wv):
                            # t[yv*vois + xv] = imgmed[yv, xv]
                            t[yv*vois + xv] = imgvois[yv, xv]
                           
                        
                    img_result[y,x]= Metrics.median(t) 
               
            # if (y%10 == 0):
                # print(y)
        #crop back to original
        img_result = img_result[pad:pad+img_height, pad:pad+img_width]                
        return img_result
    
    @staticmethod
    def filter2D(img, kernel):
         
         """an implementation of CV2.filter2D for linear filters
 
         Returns:
             result img: numpy array of the filtred input image 
         """
         img_height, img_width = img.shape
         kernel_height, kernel_width = kernel.shape
     
         # Ensure the kernel has an odd size
         assert kernel_height % 2 == 1 and kernel_width % 2 == 1, "Kernel size must be odd"
     
         # Calculate the padding needed for convolution
         pad_y = kernel_height // 2
         pad_x = kernel_width // 2
         
         # Create an empty output image
         img_result = np.zeros_like(img)
         
         for y in range(pad_y,img_height - pad_y):
             for x in range (pad_x,img_width - pad_x) :
                 # Extract the local region around the pixel
                 img_region = img[y - pad_y:y + pad_y + 1, x - pad_x:x + pad_x + 1]
                 
                 # Perform the convolution
                 img_result[y, x] = Convolution.convolution(img_region,kernel)
        
         #crop back to original
        #  img_result = img_result[pad_y:pad_y+img_height, pad_x:pad_x+img_width]
                 
         return img_result
     
    @staticmethod
    def filter_Morph(img,type="EROSION",iterations=1,kernel_shape="RECT",size_kernel=1):
         """Performs either dilatation or erosion
 
         Args:
             img (ndarray): image tob eused
             kernel (ndarray): cross or rect
             type (str, optional): Erosion or Dilate. Defaults to "EROSION".
 
         Returns:
             ndarray: image after Filter
         """
         if type.lower()=="erosion":
             method = Convolution.erode
         else:
            method = Convolution.dilate
         
         
         img_binary = Convolution.threshold(img,128,255)
            
         kernel = Convolution.create_structuring_element(kernel_shape,size_kernel)   
         
         img_height, img_width = img_binary.shape
         kernel_height, kernel_width = kernel.shape
     
         # Ensure the kernel has an odd size
         assert kernel_height % 2 == 1 and kernel_width % 2 == 1, "Kernel size must be odd"
     
         # Calculate the padding needed for convolution
         pad_y = kernel_height // 2
         pad_x = kernel_width // 2
         
        #  img_result = np.zeros_like(img)
         img_result = img_binary.copy()
         img_copy = img_binary.copy()
         # Iterate over each pixel in the input image
         for _ in range(iterations):
            for y in range(pad_y,img_height - pad_y):
                
                for x in range(pad_x,img_width - pad_x):
                    # Extract the local region around the pixel
                    img_region = img_copy[y - pad_y:y + pad_y + 1, x - pad_x:x + pad_x + 1]
                    
                    # Perform the convolution
                    img_result[y, x] = method(img_region,kernel)
                  
                    
                # if y%10==0 :print(y)
            img_copy = img_result.copy()  
               
         return img_result
     
    @staticmethod 
    def Open_Close_Morph(img,type="OPEN",iterations=1,kernel_shape="RECT",size_kernel=1):
         """Performs either OPEN or CLOSE
 
         Args:
             img (ndarray): image tob eused
             type (str, optional): OPEN or CLOSE. Defaults to "OPEN".
             kernel_shape (ndarray): cross or rect
             size_kernel (int): Defaults to 1
 
         Returns:
             ndarray: image after Filter
         """

         # Ensure the kernel has an odd size
         assert size_kernel % 2 == 1, "Kernel size must be odd"
        
         img_binary = Convolution.threshold(img,128,255)
        
        #  img_result = np.zeros_like(img)
         img_result = img_binary.copy()
         img_copy = img_binary.copy()
         # Iterate over each pixel in the input image
         for _ in range(iterations):
          
            if type.lower()=="open":
               
               img_result = Filters.filter_Morph(img_copy,"erosion",iterations=1,kernel_shape=kernel_shape,size_kernel=size_kernel)
               img_result = Filters.filter_Morph(img_result,"dilate",iterations=1,kernel_shape=kernel_shape,size_kernel=size_kernel)
               
            else:
               img_result = Filters.filter_Morph(img_copy,"dilate",iterations=1,kernel_shape=kernel_shape,size_kernel=size_kernel)
               img_result = Filters.filter_Morph(img_result,"erosion",iterations=1,kernel_shape=kernel_shape,size_kernel=size_kernel)      
              
            img_copy = img_result.copy()  
               
         return img_result 
     
    @staticmethod
    def Sobel(img,kernel_size=3):
         
         """an implementation of Sobel filter
 
         Returns:
             result img: numpy array of the filtred input image 
         """
         
         assert kernel_size >= 3 ,"kernel size for Sobel must be at least 3 "
         
         Hkernel,Vkernel=Metrics.generate_filter_kernel(filter_type="Sobel",size=kernel_size)
         
         img_copy = img.copy()
         img_copy=np.array(img_copy,dtype=np.float32)
         
         gradient_x = Filters.filter2D(img_copy,Hkernel)
         gradient_y = Filters.filter2D(img_copy,Vkernel)
         
         magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
         
         magnitude = np.array(magnitude,dtype=img.dtype)
   
         return magnitude 
   
if __name__=="__main__":
    img = cv2.imread("images/barca.png", cv2.IMREAD_GRAYSCALE)
    
    
    # HSober_kernel = np.array([[-1, 0, 1],[ -2, 0, 2], [-1, 0, 1]])
    # VSober_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) 
    
    HSober_kernel,VSober_kernel = Metrics.generate_filter_kernel(filter_type="Sobel",size=3)
    
    # emboss_kernel = np.array([[-2, -1, 0],[-1,  1, 1],[ 0,  1, 2]])
    emboss_kernel = Metrics.generate_filter_kernel(filter_type="emboss",size=3)
    
    # laplacien_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacien_kernel = Metrics.generate_filter_kernel(filter_type="laplacian",size=3)
    
    # gauss_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
    gauss_kernel = Metrics.generate_filter_kernel(filter_type="gaussian",size=3)
    
    # erosion_kernel = Convolution.create_structuring_element(shape="rect",size=1)
    
    # img_mean = Filters.filtreMoy(img)
    # img_med = Filters.filtreMed(img)
    # img_laplacien = Filters.filter2D(img,laplacien_kernel)
    # img_emboss = Filters.filter2D(img,emboss_kernel)
    # img_gauss = Filters.filter2D(img,gauss_kernel)
    # img_binary = Convolution.threshold(img,128,255)
    img_erodecross = Filters.filter_Morph(img,iterations=1,type="erosion",kernel_shape="cross",size_kernel=5)
    img_erode = Filters.filter_Morph(img,iterations=1,type="erosion",kernel_shape="RECT",size_kernel=5)
    # img_dilatecross = Filters.filter_Morph(img,iterations=1,type="dilate",kernel_shape="cross",size_kernel=5)
    # img_dilate = Filters.filter_Morph(img,iterations=1,type="dilate",kernel_shape="dilate",size_kernel=5)
    # img_sobel = Filters.Sobel(img,kernel_size=3)
    
    # img_close = Filters.Open_Close_Morph(img,type="close",iterations=1,size_kernel=5)

    # cv2.imshow("before",img)
    
    # cv2.imshow("mean",img_mean)
    # cv2.imshow("median",img_med)
    # cv2.imshow("median",img_res)
    # cv2.imshow("laplacien",img_laplacien)
    # cv2.imshow("gauss",img_gauss)
    cv2.imshow("erode",img_erode)
    cv2.imshow("erode cross",img_erodecross)
    # cv2.imshow("dilate",img_dilate)
    # cv2.imshow("dilate cross",img_dilatecross)
    # cv2.imshow("binary",img_binary)
    # cv2.imshow("sobel",img_sobel)
    # cv2.imshow("close",img_close)
    # cv2.imshow("emboss",img_emboss)

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()