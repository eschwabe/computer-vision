CSE P 576 Computer Vision - Projects
====================================

HW 1
====

Discussion:
-----------
This package includes the filtering program source code, written assignment, and 
images. The entire source code is included because minor changes were made to 
source files besides Project1.cpp to clean up some of the UI and change some of 
the provided methods.

Platform:
---------
- Windows 7
- Visual Studio 2010

Bells and Whistles:
-------------------
1.  Implemented an improved image padding method.
    The filters all use the "reflected" padding method except for the rotation 
    operation, which uses the black pixel padding method.
    
2.  Created an "Ansel Adams" image from the Yosemite image.
    The image is included with the filename "YosemiteAnselAdams.png".
    The following filters and parameters were applied to create the image:
    - Bilateral sigma=1.0 sigmaI=10.0
    - Sharpen sigma=5.0 alpha=1.0
    - Black/White
    
3.  Implemented a simple crazy filter.
    The crazy filter is a simple modification of the bilateral filter that
    generates an interesting result. By inverting the pixel intensity weight,
    the output image shifts the edges creating a sort of shadowing effect. The
    filter is not complex, so I assume its not worth many extra points, but I
    found it interesting nonetheless.


HW 2
====

Discussion:
-----------
This package includes the stitching program source code, written assignment, and 
images. The entire source code is included because minor changes were made to 
source files besides Project2.cpp to clean up some of the UI and change some of 
the provided methods.

Platform:
---------
- Windows 7
- Visual Studio 2010

Bells and Whistles:
-------------------
1.  Create a panorama that stitches together the six Mt. Rainier photographs.
    Each step of the panorama creation is included in the images with filenames
    'panorama-rainier-*.png'. The final image has the completed panorama.
    
2.  Create your own panorama using three or more images.
    The images were captured from around my neighbor. The files 
    'panorama-custom-*.jpg' are the original images downsized to ~600x400 
    pixels. The files 'panorama-custom-stitch-*.jpg' are each step of creating 
    the panorama.
    
3.  Do a better job of blending the two images.
    The stitching algorithm uses image feathering to merge the overlapping
    pixels between the two images; making the seams nearly invisible in most
    cases. There are some situations where the edge is slightly visible, mainly
    when two edges are nearly touching. The feathering is implemented by
    creating a center-weight map for each image and using the weights when
    computing the resulting pixels values when the images overlap.



HW 3
====

Discussion:
-----------
This package includes the stereo program source code, written assignment, and 
images. The entire source code is included because minor changes were made to 
source files besides Project3.cpp to clean up some of the UI and change some of 
the provided methods.

Note that the segmentation implementation is not very efficient. Running with
the default parameters takes ~1 minute to run. The segments image included
was created using the default parameters except a grid size of 10.

Platform:
---------
- Windows 7 64-bit
- Visual Studio 2010

Bells and Whistles:
-------------------
1.  Compute one of the best disparities on the 'cones.txt' image.
    The Magic Stereo button implements the Adaptive Support-Weight algorithm
    to compute the disparity window. The algorithm was implemented based on the
    "Adaptive Support-Weight Approach for Correspondence Search, PAMI 2006"
    paper.
    
    Note that the implementation uses some code from a third-party site to 
    perform the conversion from RGB to CIELab color space. See:
    http://www.csee.wvu.edu/~xinl/source.html
    
    Unfortunately, the implementation does not appear to work properly, so a
    significant reducation in the error score was not achieved. But, some minor
    improvements over SSD were accomplished. With more time, the algorithm
    could probably be fixed to obtain better results. Also, the full window
    size in the paper (35x35) could not be achieved because of memory 
    limitations.
    
    Parameters
    Param1 = color weight = 5.0
    Param2 = spatial weight = 10.0
    
    Parameters (hardcoded)
    Radius = 10 (window size = 21x21)
    Cost Truncation = 40
    
    Disparity Image: cones-disparity.png
    Error Image: cones-error.png
    
    ERROR SCORE = 11.96



HW 4
====

Discussion:
-----------
This package includes the face detection program source code, executable, 
written assignment, and images. The entire source code is included because minor 
changes were made to source files besides Project4.cpp to clean up some of the 
UI and change some of the provided methods.

Step 3 Results:
```
FILE    ORIGINAL                    Thres.  Min Max     XY-Dist Scale-Dist
3c.png  barca2.jpg                  7.0     30  45      35      15
3d.png  275854471_b318d497ef.jpg    5.0     30  40      30      15
3e.png  group.jpg                   6.0     30  35      20      15
3f.png  group-photo.jpg             5.5     30  35      25      15
```

Platform:
---------
- Windows 7 64-bit
- Visual Studio 2010


Bells and Whistles:
-------------------
1.  Add the fourth feature to the randomly generated features.
    
    The InitializeFeatures() function has been modified to support generating
    the 2x2 box grid feature. Example before and after outputs are shown in the
    images 'barca-three-features.png' and 'barca-four-features.png'
    respectively. The images were created using the default settings; the only
    difference is that each classifier was generated with three or four features.
    
    As shown in the images, the forth feature does appear to increase accuracy
    in some cases. Some of the incorrect detections at the top and bottom of the
    image are removed, however a few new incorrect detections are also added as
    well. In particular, adding the forth feature causes the detector to find
    faces in shirts more often. Overall, the net change seems to be an 
    improvement.
    
2.  Try training using different datasets sizes and numbers of candidate weak 
    classifiers. 
    
    A new classifer was trained using 12500 training images and 4000 weak
    classifiers. The training configuration can be found in the file
    'TrainingDataMassive.txt'. The resulting classifier is stored in the file
    'classifier-massive.txt'. The classifier was run on the image with a group
    of people; some of which are partially hidden by other people in the image
    (275854471_b318d497ef.jpg). Using the same parameters as Step 3 (see above),
    the resulting face detections were slightly improved. The hidden faces
    were not detected, but many of the false detections were fixed by using the
    improved classifier. Comparing the files '3d-massive-classifier.png' and
    '3d.png' shows the results.
    
    In general, increasing the amount of training data and weak classifiers will
    improve the accuracy of the strong classifier since it is based on a larger
    set of data.
