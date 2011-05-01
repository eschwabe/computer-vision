#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "Matrix.h"
#include <sstream>
#include <iomanip>
#include <cmath>

#ifdef _MSC_VER 
#define NOMINMAX
#include <Windows.h>
#else
#include <iostream>
#endif

// -----------------------------------------------------------------------------
// HELPER METHODS
// -----------------------------------------------------------------------------

/**
* Print string to standard output
*
* @param str - output string
*/
static void PrintString(const std::string& str)
{
#ifdef _MSC_VER 
    OutputDebugStringA(str.c_str());
#else
    std::cout << str;
    std::cout.flush();
#endif  
}

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
    int r, c, cd, rd;
    int w = image.width();
    int h = image.height();
    double *buffer = new double [w*h];
    QRgb pixel;

    // Descriptor parameters
    double sigma = 2.0;
    int rad = 4;

    // Computer descriptors from green channel
    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            buffer[r*w + c] = (double) qGreen(pixel);
        }
    }

    // Blur
    SeparableGaussianBlurImage(buffer, w, h, sigma);

    // Compute the desciptor from the difference between the point sampled at its center
    // and eight points sampled around it.
    for(int i=0;i<numInterestsPts;i++)
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

// Generate random number between 0 and max-1 that does not include the specified set of numbers.
// If a number if the not set is unused, set to -1
static int RandomNotInSet(int max, const int& not1, const int& not2, const int& not3)
{
    int r = -1;

    while(r == -1 || r == not1 || r == not2 || r == not3)
    {
        r = std::rand() % max;
    }
    return r;
}

// Copy a homogenous array from input to output
static void ArrayHomCopy(double in[3][3], double out[3][3])
{
    out[0][0] = in[0][0];
    out[0][1] = in[0][1];
    out[0][2] = in[0][2];
    out[1][0] = in[1][0];
    out[1][1] = in[1][1];
    out[1][2] = in[1][2];
    out[2][0] = in[2][0];
    out[2][1] = in[2][1];
    out[2][2] = in[2][2];
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

// Scale buffer values to a 0-max range
static void BufferSingleScale(double *buffer, const int& bWidth, const int& bHeight, const double& max)
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

    delete [] kernel;
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

    // blur dx^2, dy^2, and dxdy buffers with seperable gaussian kernel
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
    for(std::size_t pos=0; pos<iPoints.size(); pos++)
    {
        (*interestPts)[pos] = iPoints[pos];
    }

    // scale harris points for display
    BufferSingleScale(harrisBuffer, w, h, 255.0);

    // display harris buffer
    ImageConvertBuffer(&imageDisplay, harrisBuffer);

    // display the interest points
    DrawInterestPoints(*interestPts, numInterestsPts, imageDisplay);

    // cleanup
    delete [] buffer;
    delete [] horzBuffer;
    delete [] vertBuffer;
    delete [] dxdyBuffer;
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

    // compute the descriptors for each interest point.
    ComputeDescriptors(image1, interestPts1, numInterestsPts1);
    ComputeDescriptors(image2, interestPts2, numInterestsPts2);

    std::vector<CMatches> matchList;

    // find the best match for each interest point in image 1
    for(int i1 = 0; i1 < numInterestsPts1; i1++)
    {
        // unable to compute descriptor, ignore
        if(interestPts1[i1].m_DescSize == 0)
            continue;
        
        CIntPt* match = NULL;
        double matchDiff = 0.0;

        // compare descriptor with image 2 descriptors
        for(int i2 = 0; i2 < numInterestsPts2; i2++)
        {
            // unable to compute descriptor, ignore
            if(interestPts2[i2].m_DescSize == 0)
                continue;

            // compute difference
            double diff = 0.0;
            for(int desc = 0; desc < interestPts1[i1].m_DescSize && desc < interestPts2[i2].m_DescSize; desc++)
            {
                diff += std::pow( (interestPts1[i1].m_Desc[desc] - interestPts2[i2].m_Desc[desc]), 2);
            }

            // check if closer match and update
            if(match == NULL || diff < matchDiff)
            {
                match = &interestPts2[i2];
                matchDiff = diff;
            }
        }

        // add best intesest point match to list
        CMatches pointsMatch;
        pointsMatch.m_X1 = interestPts1[i1].m_X;
        pointsMatch.m_Y1 = interestPts1[i1].m_Y;
        pointsMatch.m_X2 = match->m_X;
        pointsMatch.m_Y2 = match->m_Y;
        matchList.push_back(pointsMatch);
    }

    // save match list
    numMatches = matchList.size();
    *matches = new CMatches[numMatches];
    for(std::size_t i = 0; i < matchList.size(); i++)
    {
        (*matches)[i] = matchList[i];
    }

    // print debug info
    std::stringstream s;
    s << "Image 1 Interest Points: " << numInterestsPts1 << "\n";
    s << "Image 2 Interest Points: " << numInterestsPts2 << "\n";
    s << "Matches: " << numMatches << "\n";
    PrintString(s.str());

    // draw the matches
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
    // compute resulting matrix [H][xy1] = [uvw]
    double u = h[0][0]*x1 + h[0][1]*y1 + h[0][2]*1;
    double v = h[1][0]*x1 + h[1][1]*y1 + h[1][2]*1;
    double w = h[2][0]*x1 + h[2][1]*y1 + h[2][2]*1;
    
    x2 = u / w;
    y2 = v / w;
}

/**
* Count the number of inliers given a homography. This is a helper function for RANSAC.
*
* @param h - input homography used to project points (image1 -> image2)
* @param matches - array of matching points
* @param numMatches - number of matchs in the array
* @param inlierThreshold - maximum distance between points that are considered to be inliers
* @param inlierMatches - (optional) list of matching inliers
* @return the total number of inliers.
*/
int MainWindow::ComputeInlierCount(double h[3][3], CMatches *matches, int numMatches, double inlierThreshold, 
    CMatches** inlierMatches)
{
    int inlierCount = 0;

    std::vector<CMatches> inlierMatchList;

    // check each match
    for(int i = 0; i < numMatches; i++)
    {
        double proX = 0.0;
        double proY = 0.0;

        // project image 1 coordinates
        Project(matches[i].m_X1, matches[i].m_Y1, proX, proY, h);

        // compare distance to image 2 coordinates
        // sqrt( (x2-x1)^2 + (y2-y1)^2 )
        double dist = std::sqrt( std::pow((matches[i].m_X2 - proX),2) + std::pow((matches[i].m_Y2 - proY),2) );

        // if distance within threshold, count as inlier
        if( dist <= inlierThreshold )
        {
            inlierCount++;
            inlierMatchList.push_back(matches[i]);
        }
    }

    // save matches if requested
    if(inlierMatches != NULL)
    {
        (*inlierMatches) = new CMatches[inlierCount];

        for(int i = 0; i < inlierCount; i++)
        {
            (*inlierMatches)[i] = inlierMatchList[i];
        }
    }

    return inlierCount;
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
    int inliers = 0;
    double bestHom[3][3];

    if(numMatches == 0)
        return;

    // for each iteration
    for(int iter = 0; iter < numIterations; iter++)
    {
        double selectedHom[3][3];

        // randomly select 4 pairs of points that potentially match
        int pt1 = RandomNotInSet(numMatches, -1, -1, -1);
        int pt2 = RandomNotInSet(numMatches, pt1, -1, -1);
        int pt3 = RandomNotInSet(numMatches, pt1, pt2, -1);
        int pt4 = RandomNotInSet(numMatches, pt1, pt2, pt3);

        // compute the homography relating the four selected matches
        CMatches selectedMatches[4];
        selectedMatches[0] = matches[pt1];
        selectedMatches[1] = matches[pt2];
        selectedMatches[2] = matches[pt3];
        selectedMatches[3] = matches[pt4];
        ComputeHomography(selectedMatches, 4, selectedHom, true);

        // using the computed homography, compute the number of inliers
        int selectedInliers = ComputeInlierCount(selectedHom, matches, numMatches, inlierThreshold, NULL);

        // if this homography produces the highest number of inliers, store as the best homography
        if(selectedInliers > inliers)
        {
            inliers = selectedInliers;
            ArrayHomCopy(selectedHom, bestHom);
        }
    }
    
    // using the best homography, compute the number of inliers
    CMatches* inlierMatches = NULL;
    int inlierCount = ComputeInlierCount(bestHom, matches, numMatches, inlierThreshold, &inlierMatches);

    // compute a new homography (and inverse) based on the inlier matches
    double finalHom[3][3];
    double finalHomInv[3][3];
    ComputeHomography(inlierMatches, inlierCount, finalHom, true);
    ComputeHomography(inlierMatches, inlierCount, finalHomInv, false);

    // print debug info
    std::stringstream s;
    s << "RANSAC Inlier Count: " << inlierCount << "\n";
    PrintString(s.str());

    // display the inlier matches.
    DrawMatches(inlierMatches, inlierCount, image1Display, image2Display);

    // save homographies
    ArrayHomCopy(finalHom, hom);
    ArrayHomCopy(finalHomInv, homInv);
}

/**
* Bilinearly interpolate image (helper function for Stitch). You can just copy code 
* from previous assignment.
* 
* @param image - input image
* @param (x, y) - location to interpolate
* @param rgb - returned color values
* @return true if interpolation sucessful
*/
bool MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{
    // initialize rgb to black
    rgb[0] = 0.0;
    rgb[0] = 0.0;
    rgb[0] = 0.0;
  
    // compute base pixel
    int baseX = (int)x;
    int baseY = (int)y;

    // check if pixels in range
    if( x >= 0 && (x+1) < image->width() && y >= 0 && (y+1) < image->height() )
    {
        // compute weight values
        double a = x-baseX;
        double b = y-baseY;

        // find pixels
        QRgb pixelXY = image->pixel(baseX, baseY);
        QRgb pixelX1Y = image->pixel(baseX+1, baseY);
        QRgb pixelXY1 = image->pixel(baseX, baseY+1);
        QRgb pixelX1Y1 = image->pixel(baseX+1, baseY+1);

        // compute interpolated pixel
        // f (x + a, y + b) = (1 - a)(1 - b) f (x, y) + a(1 - b) f (x + 1, y) + (1 - a)b f (x,y + 1) + ab f (x + 1, y + 1)
        rgb[0] = ((1-a)*(1-b)*qRed(pixelXY)) + (a*(1-b)*qRed(pixelX1Y)) + ((1-a)*b*qRed(pixelXY1)) + (a*b*qRed(pixelX1Y1));
        rgb[1] = ((1-a)*(1-b)*qGreen(pixelXY)) + (a*(1-b)*qGreen(pixelX1Y)) + ((1-a)*b*qGreen(pixelXY1)) + (a*b*qGreen(pixelX1Y1));
        rgb[2] = ((1-a)*(1-b)*qBlue(pixelXY)) + (a*(1-b)*qBlue(pixelX1Y)) + ((1-a)*b*qBlue(pixelXY1)) + (a*b*qBlue(pixelX1Y1));

        // cap rgb values
        rgb[0] = std::max(rgb[0],0.0);
        rgb[0] = std::min(rgb[0],255.0);
        rgb[1] = std::max(rgb[1],0.0);
        rgb[1] = std::min(rgb[1],255.0);
        rgb[2] = std::max(rgb[2],0.0);
        rgb[2] = std::min(rgb[2],255.0);

        return true;
    }

    return false;
}

/**
* Generate a center-weight map for an image with size width*height
*/
double* GenerateWeightMap(const int& height, const int& width)
{
    double max = 0.0;

    // build buffer
    double* buffer = new double[height*width];

    // compute weight for each location in the buffer
    for(int r=0;r<height;r++)
    {
        for(int c=0;c<width;c++)
        {
            // find the minimum number of pixels (distance) to an edge of the image
            double dist = std::min( std::min(r, height-r), std::min(c, width-c) ) + 1;
            buffer[(r*width) + c] = dist;

            if(dist > max)
            {
                max = dist;
            }
        }
    }

    // normalize
    for(int r=0;r<height;r++)
    {
        for(int c=0;c<width;c++)
        {
            buffer[(r*width) + c] /= max;
        }
    }

    return buffer;
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
    // width and height of stitched image
    int ws = 0;
    int hs = 0;

    // project the four corners of image 2 onto image 1
    double image2TopLeft[2] = { 0, 0 };
    double image2TopRight[2] = { image2.width()-1, 0 };
    double image2BottomLeft[2] = { 0, image2.height()-1 };
    double image2BottomRight[2] = { image2.width()-1, image2.height()-1 };
    
    Project(image2TopLeft[0], image2TopLeft[1], image2TopLeft[0], image2TopLeft[1], homInv);
    Project(image2TopRight[0], image2TopRight[1], image2TopRight[0], image2TopRight[1], homInv);
    Project(image2BottomLeft[0], image2BottomLeft[1], image2BottomLeft[0], image2BottomLeft[1], homInv);
    Project(image2BottomRight[0], image2BottomRight[1], image2BottomRight[0], image2BottomRight[1], homInv);

    // compute the size of stitched image, minimum top-left position and maximum bottom-right position
    int top = std::min(0, (int)std::min(image2TopLeft[1], image2TopRight[1]));
    int left = std::min(0, (int)std::min(image2TopLeft[0], image2BottomLeft[0]));
    int bottom = std::max(image1.height(), (int)(std::max(image2BottomRight[1], image2BottomLeft[1])+1.0));
    int right = std::max(image1.width(), (int)(std::max(image2BottomRight[0], image2TopRight[0])+1.0));

    ws = right - left + 1;
    hs = bottom - top + 1;

    // generate weight maps for images
    double* image1Weights = GenerateWeightMap(image1.height(), image1.width());
    double* image2Weights = GenerateWeightMap(image2.height(), image2.width());

    // initialize stiched image
    stitchedImage = QImage(ws, hs, QImage::Format_RGB32);
    stitchedImage.fill(qRgb(0,0,0));

    // copy image1 into stitched image at the proper location
    for(int r=0;r<image1.height();r++)
    {
        for(int c=0;c<image1.width();c++)
        {
            stitchedImage.setPixel(c+std::abs(left), r+std::abs(top), image1.pixel(c, r));
        }
    }

    // for each pixel in stitched image, 
    for(int r=top; r < bottom; r++)
    {
        for(int c=left; c < right; c++)
        {
            double x2 = 0.0;
            double y2 = 0.0;

            // project point onto image2
            Project(c, r, x2, y2, hom);

            // interpolate image2 pixel
            double rgb[3];
            if( BilinearInterpolation(&image2, x2, y2, rgb) == true )
            {
                // stiched image row and column
                int sRow = r+std::abs(top);
                int sCol = c+std::abs(left);

                // check if overlap with image1 pixel
                // combine pixels based on weight map
                if( sRow >= std::abs(top) && sRow < std::abs(top)+image1.height() &&
                    sCol >= std::abs(left) && sCol < std::abs(left)+image1.width() )
                {
                    // verify pixel is not part of black borders
                    if(qRed(stitchedImage.pixel(sCol, sRow)) != 0 &&
                        qGreen(stitchedImage.pixel(sCol, sRow)) != 0 &&
                        qBlue(stitchedImage.pixel(sCol, sRow)) != 0)
                    {
                        // compute image 1 pixel
                        int i1Row = sRow - std::abs(top);
                        int i1Col = sCol - std::abs(left);

                        // find image weights
                        double i1Weight = image1Weights[i1Row*image1.width() + i1Col];
                        double i2Weight = image2Weights[((int)y2)*image2.width() + (int)x2];

                        // normalize weights
                        double totalWeight = i1Weight + i2Weight;
                        i1Weight /= totalWeight;
                        i2Weight /= totalWeight;

                        // compute new rgb values
                        double image2rgb[3] = {rgb[0], rgb[1], rgb[2]};

                        rgb[0] = i1Weight*qRed(stitchedImage.pixel(sCol, sRow)) + i2Weight*image2rgb[0];
                        rgb[1] = i1Weight*qGreen(stitchedImage.pixel(sCol, sRow)) + i2Weight*image2rgb[1];
                        rgb[2] = i1Weight*qBlue(stitchedImage.pixel(sCol, sRow)) + i2Weight*image2rgb[2];
                    }
                }
                 
                // AVERAGE PIXELS
                //if(qRed(stitchedImage.pixel(sCol, sRow)) != 0 &&
                //   qGreen(stitchedImage.pixel(sCol, sRow)) != 0 &&
                //   qBlue(stitchedImage.pixel(sCol, sRow)) != 0)
                //{
                //    // average with existing pixel
                //    rgb[0] = (rgb[0] + qRed(stitchedImage.pixel(sCol, sRow))) / 2;
                //    rgb[1] = (rgb[1] + qGreen(stitchedImage.pixel(sCol, sRow))) / 2;
                //    rgb[2] = (rgb[2] + qBlue(stitchedImage.pixel(sCol, sRow))) / 2;
                //}

                // add image2 pixel to stitched image
                stitchedImage.setPixel(sCol, sRow, qRgb(rgb[0], rgb[1], rgb[2]));
            }
        }
    }

    delete [] image1Weights;
    delete [] image2Weights;
}
