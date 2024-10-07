import time, os, sys
import matplotlib as mpl
import time
from PIL import Image
import skimage.io as io
import cv2
import scipy
import scipy.ndimage
from tifffile import imwrite
import tifffile as tif
import numpy as np
import skimage.io
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
from tifffile import imwrite
import scipy
from scipy.ndimage import shift
from scipy.fft import fft
import imagej
from tifffile import TiffFile, imsave
from skimage.transform import rotate


#%%

save_dir= "path" #check that imagej is installed on the right path
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#%% #Pre-process on the raw light field image 

def pre_processing(file_path):
   LFM1=skimage.io.imread(file_path)
   img_rotated=ndimage.rotate(LFM1,-0.025,reshape=False)  #Rotation correction in degrees of the microlens
   LFM1=img_rotated.astype(np.int32)
   LFMcrop2=LFM1
   h,w=LFMcrop2.shape[:2]
   fraction1=15/14.12                    #fraction1 & fraction2 defined by FFT+PRE-PROCESSING.PY to find the pitch between 2 microlens 
   fraction2=15/14.12
   img_resized=np.array(Image.fromarray(LFMcrop2,mode='I').resize((int(fraction1*w),int(fraction2*h)),Image.Resampling.BILINEAR)) #reshape the image to have 15 perpsectives (int values)
   return img_resized


#%% 


#Detection map of microlens

Nx=140                               #Number of microlenses in X and Y on the raw image (140 x 140 for example)
Ny=140
def calibrate_LFM(img):
    LFM=img;
    x0=10                            #Absolute coordinates x0,y0 of the first microlens on the raw image in X and Y
    y0=10
    LFM=np.zeros([Nx,Ny,15,15])      #4D array to store spatial and angular dimensions
    XX=np.zeros([Nx,Ny])
    YY=np.zeros([Nx,Ny])
    CenterX=np.zeros([Nx,Ny])        # Microlens centers parameterizing the 4D array of the raw image
    CenterY=np.zeros([Nx,Ny])
    for ii in range(0,Nx):
        for jj in range(0,Ny):
            if ii==0 and jj==0:
                CenterX[ii,jj]=x0
                CenterY[ii,jj]=y0
            elif ii==0 and jj!=0:
                CenterX[ii,jj]=CenterX[ii,jj-1]+15
                CenterY[ii,jj]=CenterY[ii,jj-1]
            elif ii!=0 and jj==0:
                CenterX[ii,jj]=CenterX[ii-1,jj]
                CenterY[ii,jj]=CenterY[ii-1,jj]+15
            else:
                CenterX[ii,jj]=CenterX[ii,jj-1]+15
                CenterY[ii,jj]=CenterY[ii,jj-1]
            intermediate=img[int(CenterY[ii,jj]):int(CenterY[ii,jj])+15,int(CenterX[ii,jj]):int(CenterX[ii,jj])+15] #Iteratively add 15 pixels to shift to the next center.
            result=np.where(intermediate==np.amax(intermediate))        #Center of mass detection
            a=np.asarray(result)
            ymax=a[0,0]
            xmax=a[1,0]
            XX[ii,jj]=xmax
            YY[ii,jj]=ymax
            LFM[ii,jj]=img[int(CenterY[ii,jj]+ymax)-7:int(CenterY[ii,jj]+ymax)+8,int(CenterX[ii,jj]+xmax)-7:int(CenterX[ii,jj]+xmax)+8]           
    return LFM,CenterX,CenterY,XX,YY

def compute_intensity(LFM):            #image at z=0
    I0=np.sum(LFM,3)
    I0=np.sum(I0,2)
    return I0

def compute_LFM(img,CenterX,CenterY):  #LFM[x, y, u, v] extract positions and perspective to create volumetric images 
    LFM=np.zeros([Nx,Ny,15,15])
    for ii in range(0,Nx):
        for jj in range(0,Ny):  
            LFM[ii,jj,:,:]=img[int(CenterY[ii,jj]+CY[ii,jj])-7:int(CenterY[ii,jj]+CY[ii,jj])+8,int(CenterX[ii,jj]+CX[ii,jj])-7:int(CenterX[ii,jj]+CX[ii,jj])+8]   
            #LFM[ii,jj,:,:]=img[int(CenterY[ii,jj])-7:int(CenterY[ii,jj])+8,int(CenterX[ii,jj])-7:int(CenterX[ii,jj])+8]
    return LFM

def rendered_focus(C,returnVal=False):                    #Shift and Sum algorithm
    rendered=np.zeros((rendered_height,rendered_width))
    center_uv=int(side/2)
    for u in(range(0,side)):
        for v in range(0,side):
            shift_x, shift_y=C*(center_uv-u),C*(center_uv-v)
            rendered[:, :]+=shift(radiance[:,:,u,v],(shift_x,shift_y))
    final=rendered/(side*side)
    if returnVal:
        return final

file_path0='REF.tif'       #Image reference without illumination
file_path='sample.tif'         #Image with sample 
calibration_image=pre_processing(file_path0) 
bright=pre_processing(file_path)
LFM,centerX,centerY,CX,CY=calibrate_LFM(calibration_image) #Centering of the Raw Image

#%% LFM_image

LFM01=compute_LFM((bright-calibration_image)/(calibration_image+1),centerX,centerY) # Normalization of the raw image to enhance contrast
aa=compute_intensity(LFM01)
imwrite('intensity.tif',aa)
plt.imshow(aa)

#%% STACK 3D of all images

for i in range(0,6):
    for j in range(1,7):
        file_path0=os.path.join(save_dir,f'refX{j}_{i}.tif')       
        calibration_image=pre_processing(file_path0)
        file_path=os.path.join(save_dir,f'imageX{j}_{i}.tif')
        bright=pre_processing(file_path)
        bright=(bright-calibration_image)/(calibration_image+1)
        LFM,centerX,centerY,CX,CY=calibrate_LFM(calibration_image)
        LFM01=compute_LFM(bright,centerX,centerY)                            
        LFM02=LFM01
        aa=compute_intensity(LFM01)
        rendered_height,rendered_width=LFM01.shape[0],LFM01.shape[1]
        side=LFM02.shape[2]                                            #side=pitch/2
        radiance=LFM01                                                 #4 dimensions of the LFM array 
        C=0.5  
        final_image=rendered_focus(C,returnVal=True)
        rendered_images=[]                                             #Array to store the volumetric image 
        C_values=np.linspace(-2.5,2.5,40)                              # coefficient C that depends on the depth
        for C in (C_values):                                           # Apply Shift and Sum algorithm: set the depth limits and the number of slices
            rendered_image=rendered_focus(C,returnVal=True)
            rendered_images.append(rendered_image)  
        rendered_images_stack=np.stack(rendered_images)
        file_path = os.path.join(save_dir,f'stack3D3_{j}_{i}.tif')     #individual stack with FOV of camera 
        imwrite(file_path,rendered_images_stack)
        
        
#%% Stitching process along X

path ="C:/Users/pheni/OneDrive/Documents/Fiji.app/" #check that imagej is installed on the right path
ij=imagej.init(path,headless=False)
macros = [f'''
run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=[{path}] layout_file={i}_3D.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save memory (but be slower)] image_output=[Fuse and display]");
saveAs("Tiff", "{path}/stack3D3_{i}.tif");
''' for i in range(1,7)]
for macro in macros:
    ij.py.run_macro(macro)

#rotation correction

file_paths=["C:/Users/pheni/OneDrive/Documents/Fiji.app/stack3D3_{i}.tif".format(i) for i in range(1,7)]
rotation_factors=[4.8,2.5,0,-2.3,-4.7,-7.5]          # Rotation factors on each stitched image along the X-axis.
crop_values = {
    'stack3D3_1.tif':(29,25),        # Crop of zero pixels values for the top and bottom of the X stitchingimages
    'stack3D3_2.tif':(14,15),
    'stack3D3_3.tif':(10,10),
    'stack3D3_4.tif':(13,15),
    'stack3D3_5.tif':(25,25),
    'stack3D3_6.tif':(38,39)
}  

#Apply the rotation and crop

for file_path,rotation_factor in zip(file_paths,rotation_factors):
    with TiffFile(file_path) as tif:
        images=[page.asarray() for page in tif.pages]
    rotated_and_cropped_images=[]
    for image in images:
        image_rotated=rotate(image,rotation_factor)
        height,width=image_rotated.shape[:2]
        top_crop,bottom_crop=crop_values[os.path.basename(file_path)]
        cropped_image=image_rotated[top_crop:height-bottom_crop,:]
        rotated_and_cropped_images.append(cropped_image)
    stacked_images=np.stack(rotated_and_cropped_images,axis=0)
    imsave("C:/Users/pheni/OneDrive/Documents/Fiji.app/stack3D3_1{}.tif".format(os.path.basename(file_path)[9]),stacked_images)

#final stitch in Y with the macro 

macro8=f'''
run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=[{path}] layout_file=FINAL_3D.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save memory (but be slower)] image_output=[Fuse and display]");
saveAs("Tiff", "{path}/FINAL_stitch.tif");
'''
ij.py.run_macro(macro8)        
        
#%%

#This part describes the process of 3x3 

generated_stacks=[]
for i in range(1,10):
    stack_name=f'sample_low_res{i}.tif'             #9 images with low resolution that are shifted in X and Y with 1/2 of a microlens
    image_path=os.path.join(save_dir,stack_name)
    stack=io.imread(image_path)
    generated_stacks.append(stack)
    
# 3X3 algorithm 

depth=generated_stacks[0].shape[0]
combined_stack=[]
for d in range(depth):
    images=[stack[d] for stack in generated_stacks]
    AA=np.zeros([420,420])                          #Size of the image with enhanced resolution
    for ii in range(140):                            
        for jj in range(140):                     
            AA[0+3*ii,0+3*jj]=images[0][ii,jj]      #Put the right acquisition pattern to place the pixels correctly
            AA[0+3*ii,1+3*jj]=images[1][ii,jj] 
            AA[0+3*ii,2+3*jj]=images[2][ii,jj]  
            AA[1+3*ii,0+3*jj]=images[3][ii,jj]  
            AA[1+3*ii,1+3*jj]=images[4][ii,jj]  
            AA[1+3*ii,2+3*jj]=images[5][ii,jj]  
            AA[2+3*ii,0+3*jj]=images[6][ii,jj] 
            AA[2+3*ii,1+3*jj]=images[7][ii,jj] 
            AA[2+3*ii,2+3*jj]=images[8][ii,jj]
    combined_stack.append(AA)

combined_stack=np.array(combined_stack)
imwrite(os.path.join(save_dir,'start_3x3_0.5.tif'),combined_stack) #Final image 3x3.tif
max_intensity_projection=np.max(combined_stack,axis=0)
imwrite(os.path.join(save_dir,'bright_max_intensity_projection.tiff'),max_intensity_projection)