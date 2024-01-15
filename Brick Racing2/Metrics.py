from Sort import Sort
import numpy as np

class Metrics():
    @staticmethod
    def mean(array):
        h,w = array.shape
        total,count=0,0
        
        y=0
        while(y<h):
            x=0
            while(x<w):
                total += array[y,x]
                count += 1
                x += 1
            y += 1    
        return (total/count)
    
    @staticmethod
    def median(array):
        # array.sort()
        
        array=Sort.quick_sort(array)
       
        if len(array)%2:
            return array[int(len(array)/2)]
        
        return int((array[int(len(array)/2)-1]+array[int(len(array)/2)])/2)
   
    @staticmethod
    def generate_filter_kernel(filter_type, size=3):
        """
        Generate a kernel for a specific filter type.
    
        Parameters:
        - filter_type (str): Type of filter ('laplacian', 'gaussian', 'emboss', etc.).
        - size (int, optional): Size of the kernel for filters that use a size parameter.
    
        Returns:
        - kernel (numpy array): The generated filter kernel.
        """
        # Ensure the kernel has an odd size
        assert size % 2 == 1, "Kernel size must be odd"
        
        if filter_type.lower() == 'laplacian':
   
            kernel = np.ones((size, size)) * -1
            center_value = -1 * (size**2 - 1)
            kernel[size // 2, size // 2] = center_value
    
        elif filter_type.lower() == 'gaussian':

            kernel = np.zeros((size, size))
            center = size // 2
            sigma = 1  # You can adjust the standard deviation (sigma) based on your requirements
            for i in range(size):
                 for j in range(size):
                    distance = (i - center)**2 + (j - center)**2
                    kernel[i, j] = np.exp(-distance / (2 * sigma**2))
    
            kernel /= kernel.sum()
             
        elif filter_type.lower() == 'emboss':
            # Fixed emboss filter formula
            # kernel = np.array([[0, 1, 0],
            #                    [0, 0, 0],
            #                    [0, -1, 0]])
            kernel = np.array([[-2, -1, 0],[-1,  1, 1],[ 0,  1, 2]])
    
        # Add more filter types as needed
        elif filter_type.lower() == 'sobel':
            half_size = size // 2
            sobel_x = np.zeros((size, size))
            sobel_y = np.zeros((size, size))
        
            for i in range(size):
                for j in range(size):
                    sobel_x[i, j] = (j - half_size) / (4 * half_size)
                    sobel_y[i, j] = (i - half_size) / (4 * half_size)
        
            sobel_x /= np.abs(sobel_x).sum()
            sobel_y /= np.abs(sobel_y).sum()
            return (sobel_x,sobel_y)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
    
        return kernel
    
if __name__=="__main__":
    
    print(Metrics.median([134,50,32,626,32,10]))