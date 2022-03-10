import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

def path_to_image(x,y):
  # Read image.
  img = cv2.imread('C:/Users/adish/OneDrive/Desktop/circle.jpg', cv2.IMREAD_COLOR)
  
  # Convert to grayscale.
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Blur using 3 * 3 kernel.
  gray_blurred = cv2.blur(gray, (3, 3))

  # Apply Hough transform on the blurred image.
  detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,param2=30,
          minRadius = 1)
  
  detected=detected_circles[0]
  pt_count=0
  print("No. of Circles Detected:", len(detected))

  for i in range(len(detected)):
      center=tuple(detected[i][:2])

      radius=detected[i][-1]

      pt=(x,y)

      d=0
      for j in range(len(pt)):
        d+=(pt[j]-center[j])**2

      d=np.sqrt(d)

      if d<radius:
        pt_count+=1
        print("Lies in Circle",i+1)

  if pt_count==0:
    print("The point Lies Outside of all circles.")


if __name__=="__main__":
    args=sys.argv
    x=int(args[1])
    y=int(args[2])
    path_to_image(x,y)
   

