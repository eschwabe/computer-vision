#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "Matrix.h"

// -----------------------------------------------------------------------------
// HELPER METHODS
// -----------------------------------------------------------------------------

/**
* Draw detected Harris corners. Draws a red cross on top of detected corners.
*     
* @param interestPts - interest points
* @param numInterestsPts - number of interest points
* @param imageDisplay - image used for drawing    
*/
void MainWindow::DrawInterestPoints(CIntPt *interestPts, int numInterestsPts, QImage &imageDisplay)
{
    int i;
    int r, c, rd, cd;
    int w = imageDisplay.width();
    int h = imageDisplay.height();

    for(i=0;i<numInterestsPts;i++)
    {
        c = (int) interestPts[i].m_X;
        r = (int) interestPts[i].m_Y;

        for(rd=-2;rd<=2;rd++)
            if(r+rd >= 0 && r+rd < h && c >= 0 && c < w)
                imageDisplay.setPixel(c, r + rd, qRgb(255, 0, 0));

        for(cd=-2;cd<=2;cd++)
            if(r >= 0 && r < h && c + cd >= 0 && c + cd < w)
                imageDisplay.setPixel(c + cd, r, qRgb(255, 0, 0));
    }
}

/**
* Compute interest point descriptors
* 
* If the descriptor cannot be computed, i.e. it's too close to the boundary of
* the image, its descriptor length will be set to 0. I've implemented a very simple 8 
* dimensional descriptor. Feel free to improve upon this. 
*
* @param image - input image
* @param interestPts - array of interest points
* @param numInterestsPts - number of interest points
*/
void MainWindow::ComputeDescriptors(QImage image, CIntPt *interestPts, int numInterestsPts)
{
    int r, c, cd, rd, i, j;
    int w = image.width();
    int h = image.height();
    double *buffer = new double [w*h];
    QRgb pixel;

    // Descriptor parameters
    double sigma = 2.0;
    int rad = 4;

    // Computer descriptors from green channel
    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            buffer[r*w + c] = (double) qGreen(pixel);
        }

        // Blur
        SeparableGaussianBlurImage(buffer, w, h, sigma);

        // Compute the desciptor from the difference between the point sampled at its center
        // and eight points sampled around it.
        for(i=0;i<numInterestsPts;i++)
        {
            int c = (int) interestPts[i].m_X;
            int r = (int) interestPts[i].m_Y;

            if(c >= rad && c < w - rad && r >= rad && r < h - rad)
            {
                double centerValue = buffer[(r)*w + c];
                int j = 0;

                for(rd=-1;rd<=1;rd++)
                    for(cd=-1;cd<=1;cd++)
                        if(rd != 0 || cd != 0)
                        {
                            interestPts[i].m_Desc[j] = buffer[(r + rd*rad)*w + c + cd*rad] - centerValue;
                            j++;
                        }

                        interestPts[i].m_DescSize = DESC_SIZE;
            }
            else
            {
                interestPts[i].m_DescSize = 0;
            }
        }

        delete [] buffer;
}

/**
* Draw matches between images. Draws a green line between matches.
*
* @param matches - matching points
* @param numMatches - number of matching points
* @param image1Display - image to draw matches
* @param image2Display - image to draw matches
*/
void MainWindow::DrawMatches(CMatches *matches, int numMatches, QImage &image1Display, QImage &image2Display)
{
    int i;
    // Show matches on image
    QPainter painter;
    painter.begin(&image1Display);
    QColor green(0, 250, 0);
    QColor red(250, 0, 0);

    for(i=0;i<numMatches;i++)
    {
        painter.setPen(green);
        painter.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter.setPen(red);
        painter.drawEllipse((int) matches[i].m_X1-1, (int) matches[i].m_Y1-1, 3, 3);
    }

    QPainter painter2;
    painter2.begin(&image2Display);
    painter2.setPen(green);

    for(i=0;i<numMatches;i++)
    {
        painter2.setPen(green);
        painter2.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter2.setPen(red);
        painter2.drawEllipse((int) matches[i].m_X2-1, (int) matches[i].m_Y2-1, 3, 3);
    }
}

/**
* Given a set of matches computes the "best fitting" homography.
* 
* @param matches - matching points
* @param numMatches - number of matching points
* @param h - returned homography
* @param isForward - direction of the projection (true = image1 -> image2, false = image2 -> image1)
*/
bool MainWindow::ComputeHomography(CMatches *matches, int numMatches, double h[3][3], bool isForward)
{
    int error;
    int nEq=numMatches*2;

    dmat M=newdmat(0,nEq,0,7,&error);
    dmat a=newdmat(0,7,0,0,&error);
    dmat b=newdmat(0,nEq,0,0,&error);

    double x0, y0, x1, y1;

    for (int i=0;i<nEq/2;i++)
    {
        if(isForward == false)
        {
            x0 = matches[i].m_X1;
            y0 = matches[i].m_Y1;
            x1 = matches[i].m_X2;
            y1 = matches[i].m_Y2;
        }
        else
        {
            x0 = matches[i].m_X2;
            y0 = matches[i].m_Y2;
            x1 = matches[i].m_X1;
            y1 = matches[i].m_Y1;
        }

        //Eq 1 for corrpoint
        M.el[i*2][0]=x1;
        M.el[i*2][1]=y1;
        M.el[i*2][2]=1;
        M.el[i*2][3]=0;
        M.el[i*2][4]=0;
        M.el[i*2][5]=0;
        M.el[i*2][6]=(x1*x0*-1);
        M.el[i*2][7]=(y1*x0*-1);

        b.el[i*2][0]=x0;
        //Eq 2 for corrpoint
        M.el[i*2+1][0]=0;
        M.el[i*2+1][1]=0;
        M.el[i*2+1][2]=0;
        M.el[i*2+1][3]=x1;
        M.el[i*2+1][4]=y1;
        M.el[i*2+1][5]=1;
        M.el[i*2+1][6]=(x1*y0*-1);
        M.el[i*2+1][7]=(y1*y0*-1);

        b.el[i*2+1][0]=y0;

    }
    int ret=solve_system (M,a,b);
    if (ret!=0)
    {
        freemat(M);
        freemat(a);
        freemat(b);

        return false;
    }
    else
    {
        h[0][0]= a.el[0][0];
        h[0][1]= a.el[1][0];
        h[0][2]= a.el[2][0];

        h[1][0]= a.el[3][0];
        h[1][1]= a.el[4][0];
        h[1][2]= a.el[5][0];

        h[2][0]= a.el[6][0];
        h[2][1]= a.el[7][0];
        h[2][2]= 1;
    }

    freemat(M);
    freemat(a);
    freemat(b);

    return true;
}

// -----------------------------------------------------------------------------
// IMAGE METHODS
// -----------------------------------------------------------------------------

// Convert the single channel buffer back into the image
// Channels outside the 0-255 range are truncated
// Buffer assumed to be the same size as the image
static void ImageConvertBuffer(QImage *image, double *buffer)
{
    for(int r = 0; r < image->height(); r++)
    {
        for(int c = 0; c < image->width(); c++)
        {
            double p = buffer[(r*image->width())+c];

            // convert to integer value and truncate
            int red = (int)floor(p + 0.5);
            int green = (int)floor(p + 0.5);
            int blue = (int)floor(p + 0.5);
            red = std::max(red,0);
            red = std::min(red,255);
            green = std::max(green,0);
            green = std::min(green,255);
            blue = std::max(blue,0);
            blue = std::min(blue,255);

            // set pixel
            image->setPixel(c, r, qRgb(red, green, blue));
        }
    }
}

// -----------------------------------------------------------------------------
// BUFFER METHODS
// -----------------------------------------------------------------------------

// Creates a copy of a single channel buffer.
// Buffer copy must be freed by caller.
static double* BufferSingleCreateCopy(double *buffer, const int& bWidth, const int& bHeight)
{
    // initialize new buffer
    double *newBuffer = new double[ bWidth * bHeight ];

    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            newBuffer[ (r*bWidth)+c ] = buffer[ (r*bWidth)+c ];
        }
    }

    return newBuffer;
}

// Applies a kernel filter to a single channel buffer. Values are updated in the buffer.
// The kernel can be any odd numbered height and width in size.
static void BufferSingleApplyKernel(
    double *buffer, const int& bWidth, const int& bHeight, 
    const double* kernel, const int kHeight, const int kWidth)
{
    // compute horizontal and vertical kernel radius
    int kHeightRadius = kHeight/2;
    int kWidthRadius = kWidth/2;

    // create copy of original buffer
    double* copyBuffer = BufferSingleCreateCopy(buffer, bWidth, bHeight);

    // for each pixel in the image
    for(int r=0;r<bHeight;r++)
    {
        for(int c=0;c<bWidth;c++)
        {
            double outValue = 0.0;

            // convolve the kernel at each pixel
            for(int rd=-kHeightRadius; rd<=kHeightRadius; rd++)
            {
                for(int cd=-kWidthRadius; cd<=kWidthRadius; cd++)
                {
                    // get the original pixel value from copy buffer
                    int pRow = r + (rd);
                    int pCol = c + (cd);

                    // verify pixel is within buffer range
                    double p = 0.0;
                    if(pRow >= 0 && pRow < bHeight && pCol >= 0 && pCol < bWidth)
                    {
                        p = copyBuffer[ (pRow*bWidth) + pCol ];
                    }

                    // get the value of the kernel
                    double weight = kernel[(rd + kHeightRadius)*kWidth + (cd + kWidthRadius)];

                    // apply weights
                    outValue += weight*p;
                }
            }

            // store mean pixel in the buffer
            buffer[ (r*bWidth) + c ] = outValue;
        }
    }
}

// Add a color offset value to every single channel
static void BufferSingleApplyOffset(double *buffer, const int& bWidth, const int& bHeight, const double& offset)
{
    // add offsets
    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            buffer[(r*bWidth)+c] += offset;
        }
    }
}

// Scale buffer values to the min/max range
static void BufferSingleScale(double *buffer, const int& bWidth, const int& bHeight, const double& min, const double& max)
{
    // find buffer min max
    double bMin = 0.0;
    double bMax = 0.0;

    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            if( buffer[(r*bWidth)+c] < bMin )
            {
                bMin = buffer[(r*bWidth)+c];
            }
            if( buffer[(r*bWidth)+c] > bMax )
            {
                bMax = buffer[(r*bWidth)+c];
            }
        }
    }

    // scale buffer values
    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            // shift value range to 0
            if( bMin > 0.0 )
                buffer[(r*bWidth)+c] -= bMin;
            else if(bMin < 0.0)
                buffer[(r*bWidth)+c] += bMin;

            // scale value
            buffer[(r*bWidth)+c] = (max / bMax) * buffer[(r*bWidth)+c];
        }
    }
}

// -----------------------------------------------------------------------------
// KERNEL METHODS
// -----------------------------------------------------------------------------

// Normalize kernel values so the sum of the absolute values is 1
// Kernel must only contain a single axis (horizontal or vertical)
static void KernelNormalize(double* kernel, const int& size)
{
    double norm = 0.0;

    // compute normal
    for(int i=0; i<size; i++)
    {
        norm += std::abs(kernel[i]);
    }

    // normalize kernel
    double net = 0.0;
    for(int i=0; i<size; i++)
    {
        kernel[i] /= norm;
        net += kernel[i];
    }
}

// Build a horizontal/vertical gaussian kernel with the specified sigma, radius, and size
// Size must be 2*radius+1
static double* KernelBuildSeperableGaussian(const double& sigma, const int& radius, const int& size) 
{
    // create kernel
    double *kernel = new double [size];

    // compute kernel weights
    for(int x=-radius; x<=radius; x++)
    {
        double value = std::exp( -std::pow((double)x,2) / (2*std::pow(sigma,2)) );
        kernel[x+radius] = value;
    }

    // normalize kernel
    KernelNormalize(kernel, size);

    return kernel;
}

// Build a horizontal/vertical guassian first derivative kernal
// Size must be 2*radius+1
static double* KernelBuildFirstDervGuassian(const double& sigma, const int& radius, const int& size) 
{
    // create first derivative horizontal to convolve with the image
    double* fdKernel = new double[size];

    // compute horizontal first derivative kernel weights
    for(int x=-radius; x<=radius; x++)
    {
        // first derivate equation
        // ( x / sigma^2 ) * ( e ^ (-x^2 / 2*sigma^2) )
        double value = (-x / (std::pow(sigma,2))) * (std::exp(-std::pow((double)x,2) / (2*std::pow(sigma,2))));
        fdKernel[x+radius] = value;
    }

    // normalize kernel
    KernelNormalize(fdKernel, size);

    return fdKernel;
}

// -----------------------------------------------------------------------------
// BUTTON METHODS
// -----------------------------------------------------------------------------

/**
* Blur a single channel floating point image with a Gaussian. This code should 
* be very similar to the code you wrote for assignment 1.
*    
* @param image - input and output image
* @param w - image width
* @param h - image height
* @param sigma - standard deviation of Gaussian
*/
void MainWindow::SeparableGaussianBlurImage(double *image, int w, int h, double sigma)
{
    // kernel radius and size
    int radius = 3*sigma;
    int size = 2*radius + 1;

    // create kernel to convolve with the image
    double *kernel = KernelBuildSeperableGaussian(sigma, radius, size);

    // apply kernel to buffer in both horizonal and vertical directions
    BufferSingleApplyKernel(image, w, h, kernel, 1, size);
    BufferSingleApplyKernel(image, w, h, kernel, size, 1);
}

/**
* Detect Harris corners.
*
* @param image - input image
* @param sigma - standard deviation of Gaussian used to blur corner detector
* @param thres - threshold for detecting corners
* @param interestPts - returned interest points
* @param numInterestsPts - number of interest points returned
* @param imageDisplay - image returned to display (for debugging)
*/
void MainWindow::HarrisCornerDetector(QImage image, double sigma, double thres, CIntPt **interestPts, int &numInterestsPts, QImage &imageDisplay)
{
    int w = image.width();
    int h = image.height();
    QRgb pixel;

    // initialize interest points
    numInterestsPts = 0;

    // compute the corner response using just the green channel
    double *buffer = new double [w*h];
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            buffer[(r*w) + c] = (double)qGreen(pixel);
        }
    }

    // kernel radius and size
    int radius = 3*sigma;
    int size = 2*radius + 1;

    // create first derivative gaussian kernel
    double *fdKernel = KernelBuildFirstDervGuassian(sigma, radius, size);

    // compute the x and y derivatives on the image stored in “buffer”

    // create buffer copies
    double *horzBuffer = BufferSingleCreateCopy(buffer, w, h);
    double *vertBuffer = BufferSingleCreateCopy(buffer, w, h);
    double *dxdyBuffer = BufferSingleCreateCopy(buffer, w, h);

    // apply first derivative in horizontal and vertical directions
    BufferSingleApplyKernel(horzBuffer, w, h, fdKernel, 1, size);
    BufferSingleApplyKernel(vertBuffer, w, h, fdKernel, size, 1);

    // create dx^2, dy^2, and dxdy buffers
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            dxdyBuffer[(r*w) + c] = horzBuffer[(r*w) + c] * vertBuffer[(r*w) + c];
        }
    }

    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            horzBuffer[(r*w) + c] = std::pow(horzBuffer[(r*w) + c], 2);
            vertBuffer[(r*w) + c] = std::pow(vertBuffer[(r*w) + c], 2);
        }
    }

    // blur x^2, dy^2, and dxdy buffers with seperable gaussian kernel
    SeparableGaussianBlurImage(horzBuffer, w, h, sigma);
    SeparableGaussianBlurImage(vertBuffer, w, h, sigma);
    SeparableGaussianBlurImage(dxdyBuffer, w, h, sigma);

    // compute the Harris response using determinant(H)/trace(H)
    // H = [ a b | c d ]
    double *harrisBuffer = BufferSingleCreateCopy(buffer, w, h);

    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            double Ha = horzBuffer[(r*w) + c];
            double Hb = dxdyBuffer[(r*w) + c];
            double Hc = dxdyBuffer[(r*w) + c];
            double Hd = vertBuffer[(r*w) + c];

            harrisBuffer[(r*w) + c] = 0.0;
            if( Ha != 0.0 && Hd != 0.0 )
            {
                harrisBuffer[(r*w) + c] = ((Ha*Hd) - (Hb*Hc)) / (Ha+Hd);
            }
        }
    }

    // initialize interest points
    std::vector<CIntPt> iPoints;

    // find peaks in the response that are above the threshold “thres”
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            if( harrisBuffer[(r*w) + c] > thres )
            {
                bool largest = true;
                double hCur = harrisBuffer[(r*w) + c];
                
                // search nearby pixels, check if largest
                for(int rd=-1; rd<=1; rd++)
                {
                    for(int cd=-1; cd<=1; cd++)
                    {
                        // get the adjacent harris value
                        int hRow = r + (rd);
                        int hCol = c + (cd);

                        // set flag if adjacent value larger and exists
                        if(hRow >= 0 && hRow < h && hCol >= 0 && hCol < w)
                        {
                            if( harrisBuffer[(hRow*w) + hCol] > hCur )
                            {
                                largest = false;
                            }
                        }
                    }
                }

                // if largest, interest point found
                if( largest == true )
                {
                    CIntPt pt;
                    pt.m_X = c;
                    pt.m_Y = r;
                    iPoints.push_back(pt);
                }
            }
        }
    }

    // allocate interest points
    numInterestsPts = iPoints.size();
    *interestPts = new CIntPt[numInterestsPts];

    // store the interest point locations in “interestPts”
    for(int pos=0; pos<iPoints.size(); pos++)
    {
        (*interestPts)[pos] = iPoints[pos];
    }

    // scale harris points for display
    BufferSingleScale(harrisBuffer, w, h, 0.0, 255.0);

    // display harris buffer
    ImageConvertBuffer(&imageDisplay, harrisBuffer);

    // display the interest points
    DrawInterestPoints(*interestPts, numInterestsPts, imageDisplay);

    // cleanup
    delete [] buffer;
    delete [] horzBuffer;
    delete [] vertBuffer;
    delete [] harrisBuffer;
    delete [] fdKernel;
}

/**
* Find matching interest points between images.
*     
* @param image1 - first input image
* @param interestPts1 - interest points corresponding to image 1
* @param numInterestsPts1 - number of interest points in image 1
* @param image2 - second input image
* @param interestPts2 - interest points corresponding to image 2
* @param numInterestsPts2 - number of interest points in image 2
* @param matches - set of matching points to be returned
* @param numMatches - number of matching points returned
* @param image1Display - image used to display matches
* @param image2Display - image used to display matches
*/
void MainWindow::MatchInterestPoints(
    QImage image1, CIntPt *interestPts1, int numInterestsPts1,
    QImage image2, CIntPt *interestPts2, int numInterestsPts2,
    CMatches **matches, int &numMatches, QImage &image1Display, QImage &image2Display)
{
    numMatches = 0;

    // Compute the descriptors for each interest point.
    // You can access the descriptor for each interest point using interestPts1[i].m_Desc[j].
    // If interestPts1[i].m_DescSize = 0, it was not able to compute a descriptor for that point
    ComputeDescriptors(image1, interestPts1, numInterestsPts1);
    ComputeDescriptors(image2, interestPts2, numInterestsPts2);

    // Add your code here for finding the best matches for each point.

    // Once you uknow the number of matches allocate an array as follows:
    // *matches = new CMatches [numMatches];

    // Draw the matches
    DrawMatches(*matches, numMatches, image1Display, image2Display);
}

/**
* Project a point (x1, y1) using the homography transformation h.
* 
* @param (x1, y1) - input point
* @param (x2, y2) - returned point
* @param h - input homography used to project point
*/
void MainWindow::Project(double x1, double y1, double &x2, double &y2, double h[3][3])
{
    // Add your code here.
}

/**
* Count the number of inliers given a homography. This is a helper function for RANSAC.
*
* @param h - input homography used to project points (image1 -> image2
* @param matches - array of matching points
* @param numMatches - number of matchs in the array
* @param inlierThreshold - maximum distance between points that are considered to be inliers
* @return the total number of inliers.
*/
int MainWindow::ComputeInlierCount(double h[3][3], CMatches *matches, int numMatches, double inlierThreshold)
{
    // Add your code here.

    return 0;
}

/**
* Compute homography transformation between images using RANSAC.
*
* @param matches - set of matching points between images
* @param numMatches - number of matching points
* @param numIterations - number of iterations to run RANSAC
* @param inlierThreshold - maximum distance between points that are considered to be inliers
* @param hom - returned homography transformation (image1 -> image2)
* @param homInv - returned inverse homography transformation (image2 -> image1)
* @param image1Display - image used to display matches
* @param image2Display - image used to display matches
*/
void MainWindow::RANSAC(CMatches *matches, int numMatches, int numIterations, double inlierThreshold,
    double hom[3][3], double homInv[3][3], QImage &image1Display, QImage &image2Display)
{
    // Add your code here.

    // After you're done computing the inliers, display the corresponding matches.
    // DrawMatches(inliers, numInliers, image1Display, image2Display);
}

/**
* Bilinearly interpolate image (helper function for Stitch). You can just copy code 
* from previous assignment.
* 
* @param image - input image
* @param (x, y) - location to interpolate
* @param rgb - returned color values
*/
bool MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{
    // Add your code here.

    return true;
}

/**
* Stitch together two images using the homography transformation.
* 
* @param image1 - first input image
* @param image2 - second input image
* @param hom - homography transformation (image1 -> image2)
* @param homInv - inverse homography transformation (image2 -> image1)
* @param stitchedImage - returned stitched image
*/
void MainWindow::Stitch(QImage image1, QImage image2, double hom[3][3], double homInv[3][3], QImage &stitchedImage)
{
    // Width and height of stitchedImage
    int ws = 0;
    int hs = 0;

    // Add your code to compute ws and hs here.

    stitchedImage = QImage(ws, hs, QImage::Format_RGB32);
    stitchedImage.fill(qRgb(0,0,0));

    // Add you code to warp image1 and image2 to stitchedImage here.
}
