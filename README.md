#### Sandra Maria Joseph


9th March 2022

# To find circular regions in a given image.
An image contains different shapes: circles, squares, curves etc.. So, the objective is to detect the circular region in the given image.  Image Processing techniques can be used to detect these shapes.

### Image Processing :
Image processing is a way to convert an image to a digital aspect and perform certain functions on it, in order to get an enhanced image or extract other useful information from it. It is a type of signal time when the input is an image, such as a video frame or image and output can be an image or features associated with that image. Usually, the Image Processing system includes treating images as two equal symbols while using the set methods used.

### Programming Languages and Libraries :
Python Programming Language is used. The major libraries used include :

  1.OpenCV     : Image Processing Libraries
  
  2.matplotlib   : Visualization
  
  3.numpy        : Mathematical Operations
  
  4.sys          : For manipulating run time environment in Python

### Methodology Used:
  1.Loading the Image
  
  2.Converting to grayscale.
  
  3.Noise Reduction using Blurring
  
  4.Applying Hough transform on the blurred image to detect the circle. It returns an array containing the centers and radii of all the circles that are detected.
  
  5.Identifying whether the given pixel point, say, (50,50), lies inside or outside the circle.
  
  6.Video Classifier Using Deep Learning

# Video Classifier
As the world is moving towards automation, researchers are focussing on capturing and analyzing video. Video classification is similar to image classification, in that the algorithm uses feature extractors, such as convolutional neural networks (CNNs), to extract feature descriptors from a sequence of images and then classify them into categories. Video classification using deep learning provides a means to analyze, classify, and track activity contained in visual data sources, such as a video stream. Video classification has many applications, such as human activity recognition, gesture recognition, anomaly detection, and surveillance.

### Programming Languages and Libraries :
Python Programming Language is used. The major libraries used include :
  1.Tensorflow 
  
  2.OpenCV
  
  3.Matplotlib
  
  4.Pandas
  
  5.numpy
  
  6.Skimage
  
### Methodology
  1.Importing the Required Libraries
  
  2.Loading Imagenet weights Using RESNET architecture.
  
  3.Capturing the video from the given path
  
  4.Finding the frames of the video
  
  5.Iterating through each frame
  
  6.Preparing the each frame for the Resnet model
  
  7.Getting the predicted probabilities for each class
  
  8.Ploting the Image with the Predicted Probabilities

