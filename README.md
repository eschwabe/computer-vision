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
Windows 7
Visual Studio 2010

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
