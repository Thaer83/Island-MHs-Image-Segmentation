
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def load_image(file_name):
    
    img = Image.open(file_name)
    #print(img.format, img.size, img.mode)
    
    #The size attribute is a 2-tuple containing width and height (in pixels)
    #attribute defines the number and names of the bands in the image, and also the pixel type and depth. 
    #Common modes are “L” (luminance) for greyscale images, “RGB” for true color images, and “CMYK” for pre-press images
    if img.mode =='RGB':
        gray_img=img.convert('L') #Convert photo to gray scale
    else:
        gray_img = img
    
    gray_img_array=np.asarray(gray_img) #Convert variable type to numpy array
    
    #------------------- Computing the histogram
    #his = histogram (gray_img_array)
    BINS = np.array(range(0,257))
    his = np.histogram(gray_img_array, bins=BINS, range=None, normed=None, weights=None, density=None) # his is a tuple
    
    #----- for Otsu ----------------
    Lmax= 256;
    probR= np.zeros(Lmax)
    Total_pixels = gray_img.size[0] * gray_img.size[1]
    n_countR = his[0]
    for i in range(Lmax):
        probR[i]=n_countR[i]/Total_pixels;
    
    #plt.imshow(gray_img_array,cmap='gray', vmin = 0, vmax = 255)
    #plt.show()
    
    return his[0], gray_img, probR