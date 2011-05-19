#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <cmath>


/*******************************************************************************
    RGB TO CIELab CONVERSION METHODS
    Copied from: http://www.csee.wvu.edu/~xinl/source.html
*******************************************************************************/

// F
static double F(double input)
{
	if(input>0.008856)
		return (pow(input, 0.333333333));
	else
		return (7.787*input+0.137931034);
}

// RGB -> XYZ
static void RGBtoXYZ(double R, double G, double B, double &X, double &Y, double &Z)
{
	X=0.412453*R+0.357580*G+0.189423*B;
	Y=0.212671*R+0.715160*G+0.072169*B;
	Z=0.019334*R+0.119193*G+0.950227*B;
}

// XYZ -> CIELab
static void XYZtoLab(double X, double Y, double Z, double &L, double &a, double &b)
{
	const double Xo=244.66128;
	const double Yo=255.0;
	const double Zo=277.63227;
	L=116*F(Y/Yo)-16;
	a=500*(F(X/Xo)-F(Y/Yo));
	b=200*(F(Y/Yo)-F(Z/Zo));
}

// RGB -> CIELab
static void RGBtoLab(double R, double G, double B, double &L, double &a, double &b)
{
	double X, Y, Z;
	RGBtoXYZ(R, G, B, X, Y, Z);
	XYZtoLab(X, Y, Z, L, a, b);
}

/*******************************************************************************
    HELPER METHODS
*******************************************************************************/

/*******************************************************************************
    K-means segment the image

    image - Input image
    gridSize - Initial size of the segments
    numIterations - Number of iterations to run k-means
    spatialSigma - Spatial sigma for measuring distance
    colorSigma - Color sigma for measuring distance
    matchCost - The match cost for each pixel at each disparity
    numDisparities - Number of disparity levels
    segmentImage - Image showing segmentations
*******************************************************************************/
void MainWindow::Segment(QImage image, int gridSize, int numIterations, double spatialSigma, double colorSigma,
                         double *matchCost, int numDisparities, QImage *segmentImage)
{
    int w = image.width();
    int h = image.height();
    int iter;
    int numSegments = 0;

    // Stores the segment assignment for each pixel
    int *segment = new int [w*h];

    // Compute an initial segmentation
    GridSegmentation(segment, numSegments, gridSize, w, h);

    // allocate memory for storing the segments mean position and color
    double (*meanSpatial)[2] = new double [numSegments][2];
    double (*meanColor)[3] = new double [numSegments][3];

    // Iteratively update the segmentation
    for(iter=1;iter<numIterations;iter++)
    {
        // Compute new means
        ComputeSegmentMeans(image, segment, numSegments, meanSpatial, meanColor);
        // Compute new pixel assignment to pixels
        AssignPixelsToSegments(image, segment, numSegments, meanSpatial, meanColor, spatialSigma, colorSigma);
    }

    // Update means again for display
    ComputeSegmentMeans(image, segment, numSegments, meanSpatial, meanColor);
    // Display the segmentation
    DrawSegments(segmentImage, segment, meanColor);

    // Update the match cost based on the segmentation
    SegmentAverageMatchCost(segment, numSegments, w, h, numDisparities, matchCost);

    delete [] meanSpatial;
    delete [] meanColor;
    delete [] segment;
}

/*******************************************************************************
    Compute initial segmentation of the image using a grid

    segment - Segment assigned to each pixel
    numSegments - Number of segments
    gridSize - Size of the grid-based segments
    w - Image width
    h - Image height
*******************************************************************************/
void MainWindow::GridSegmentation(int *segment, int &numSegments, int gridSize, int w, int h)
{
    int r, c;
    int step = w/gridSize;

    if(step*gridSize < w)
        step += 1;

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
             int rs = r/gridSize;
             int cs = c/gridSize;

             segment[r*w + c] = rs*step + cs;

             numSegments = rs*step + cs + 1;

        }

}

/*******************************************************************************
    Draw the image segmentation

    segmentImage - Image to display the segmentation
    segment - Segment assigned to each pixel
    meanColor - The mean color of the segments
*******************************************************************************/
void MainWindow::DrawSegments(QImage *segmentImage, int *segment, double (*meanColor)[3])
{
    int w = segmentImage->width();
    int h = segmentImage->height();
    int r, c;

    for(r=0;r<h-1;r++)
        for(c=0;c<w-1;c++)
        {
            int segIdx = segment[r*w + c];
            if(segIdx != segment[r*w + c + 1] ||
               segIdx != segment[(r+1)*w + c])
            {
                segmentImage->setPixel(c, r, qRgb(255, 255, 255));
            }
            else
            {
                segmentImage->setPixel(c, r, qRgb((int) meanColor[segIdx][0],
                                                  (int) meanColor[segIdx][1], (int) meanColor[segIdx][2]));
            }
        }
}

/*******************************************************************************
    Display the computed disparities

    disparities - The disparity for each pixel
    disparityScale - The amount to scale the disparity for display
    minDisparity - Minimum disparity
    disparityImage - Image to display the disparity
    errorImage - Image to display the error
    GTImage - The ground truth disparities
    m_DisparityError - The average error
*******************************************************************************/
void MainWindow::DisplayDisparities(double *disparities, int disparityScale, int minDisparity,
                        QImage *disparityImage, QImage *errorImage, QImage GTImage, double *disparityError)
{
    int w = disparityImage->width();
    int h = disparityImage->height();
    int r, c;
    int gtw = GTImage.width();
    bool useGT = false;
    double pixelCt = 0.0;
    *disparityError = 0.0;
    double maxError = 1.0*(double) disparityScale;

    if(gtw == w)
        useGT = true;

    QRgb pixel;

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            double disparity = disparities[r*w + c];
            disparity *= (double) disparityScale;
            disparity -= minDisparity*disparityScale;

            disparityImage->setPixel(c, r, qRgb((int) disparity, (int) disparity, (int) disparity));

            if(useGT)
            {
                pixel = GTImage.pixel(c, r);

                if(qGreen(pixel) > 0)
                {
                    double dist = fabs(disparity - (double) qGreen(pixel));
                    if(dist > maxError)
                        (*disparityError)++;
                    pixelCt++;

                    if(dist > maxError)
                        errorImage->setPixel(c, r, qRgb(255,255,255));
                    else
                        errorImage->setPixel(c, r, qRgb(0,0,0));
                }


            }
        }

    if(useGT)
        *disparityError /= pixelCt;
}

/*******************************************************************************
    Render warped views between the images

    image - Image to be warped
    disparities - The disparities for each pixel
    disparityScale - The amount to warp the image, usually between 0 and 1
    renderImage - The final rendered image
*******************************************************************************/
void MainWindow::Render(QImage image, double *disparities, double disparityScale, QImage *renderImage)
{
    int r, c;
    int w = image.width();
    int h = image.height();
    double *projDisparity = new double [w*h];
    double *projDisparityCt = new double [w*h];
    QRgb pixel0;
    QRgb pixel1;

    memset(projDisparity, 0, w*h*sizeof(double));
    memset(projDisparityCt, 0, w*h*sizeof(double));

    // First forward project the disparity values
    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            double disparity =  -disparities[r*w + c]*disparityScale;
            double x = (double) c + disparity;
            int cp = (int) x;
            double del = x - (double) cp;

            if(cp >= 0 && cp < w-1)
            {
                // Make sure we get the depth ordering correct.
                if(projDisparityCt[r*w + cp] == 0)
                {
                    projDisparity[r*w + cp] = (1.0 - del)*disparity;
                    projDisparityCt[r*w + cp] = (1.0 - del);
                }
                else
                {
                    // Make sure the depth ordering is correct
                    if(fabs(disparity) > fabs(2.0 + projDisparity[r*w + cp]/projDisparityCt[r*w + cp]))
                    {
                        projDisparity[r*w + cp] = (1.0 - del)*disparity;
                        projDisparityCt[r*w + cp] = (1.0 - del);
                    }
                    else
                    {
                        projDisparity[r*w + cp] += (1.0 - del)*disparity;
                        projDisparityCt[r*w + cp] += (1.0 - del);
                    }
                }

                if(projDisparityCt[r*w + cp + 1] == 0)
                {
                    projDisparity[r*w + cp + 1] = (del)*disparity;
                    projDisparityCt[r*w + cp + 1] = (del);
                }
                else
                {
                    // Make sure the depth ordering is correct
                    if(fabs(disparity) > fabs(2.0 + projDisparity[r*w + cp + 1]/projDisparityCt[r*w + cp + 1]))
                    {
                        projDisparity[r*w + cp + 1] = (del)*disparity;
                        projDisparityCt[r*w + cp + 1] = (del);
                    }
                    else
                    {
                        projDisparity[r*w + cp + 1] += (del)*disparity;
                        projDisparityCt[r*w + cp + 1] += (del);
                    }
                }
            }
        }

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            if(projDisparityCt[r*w + c] > 0.0)
            {
                projDisparity[r*w + c] /= projDisparityCt[r*w + c];
            }
        }

    // Fill in small holes after the forward projection
    FillHoles(projDisparity, projDisparityCt, w, h);

    renderImage->fill(qRgb(0,0,0));

    // Backward project to find the color values for each pixel
    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
            if(projDisparityCt[r*w + c] > 0.0)
        {
            double disparity =  projDisparity[r*w + c];
            double x = (double) c - disparity;
            int cp = (int) x;
            double del = x - (double) cp;

            if(cp >= 0 && cp < w-1)
            {
                pixel0 = image.pixel(cp, r);
                pixel1 = image.pixel(cp+1, r);

                int red = (int) ((1.0 - del)*(double)qRed(pixel0) + del*(double)qRed(pixel1));
                int green = (int) ((1.0 - del)*(double)qGreen(pixel0) + del*(double)qGreen(pixel1));
                int blue = (int) ((1.0 - del)*(double)qBlue(pixel0) + del*(double)qBlue(pixel1));

                // Uncomment if you want to see the disparities
            //    red = (int) disparity*4.0;
            //    green = (int) disparity*4.0;
            //    blue = (int) disparity*4.0;

                renderImage->setPixel(c, r, qRgb(red, green, blue));
            }
        }


    delete [] projDisparity;
    delete [] projDisparityCt;
}

/*******************************************************************************
    Fill holes in the projected disparities (Render helper function)

    projDisparity - Projected disparity
    projDisparityCt - The weight of each projected disparity.  A value of 0 means the pixel doesn't have a disparity
    w, h - The width and height of the image
*******************************************************************************/
void MainWindow::FillHoles(double *projDisparity, double *projDisparityCt, int w, int h)
{
    int r, c, cd, rd;
    double *bufferCt = new double [w*h];

    memcpy(bufferCt, projDisparityCt, w*h*sizeof(double));

    for(r=1;r<h-1;r++)
        for(c=1;c<w-1;c++)
            if(bufferCt[r*w + c] == 0)
        {
            double avgDisparity = 0.0;
            double avgCt = 0.0;

            for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                {
                    int idx = (r + rd)*w + c + cd;
                   avgDisparity += projDisparity[idx]*bufferCt[idx];
                   avgCt += bufferCt[idx];
                }

            if(avgCt > 0.0)
            {
                projDisparity[r*w + c] = avgDisparity/avgCt;
                projDisparityCt[r*w + c] = avgCt;

            }
        }

    delete [] bufferCt;
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

    delete [] copyBuffer;
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

// -----------------------------------------------------------------------------
// PIXEL METHODS
// -----------------------------------------------------------------------------

double PixelMagnitude(const QRgb& p)
{
    double mag = (qRed(p) + qGreen(p) + qBlue(p)) / 3.0;
    return mag;
}

/*******************************************************************************
    IMAGE PROCESSING METHODS
*******************************************************************************/

/*******************************************************************************
    Compute match cost using Squared Distance

    image1 - Input image 1
    image2 - Input image 2
    minDisparity - Minimum disparity between image 1 and image 2
    maxDisparity - Maximum disparity between image 1 and image 2
    matchCost - The match cost (squared distance) between pixels

    To access the match cost at pixel (c, r) at disparity d use
    matchCost[d*w*h + r*w + c]
*******************************************************************************/
void MainWindow::SSD(QImage image1, QImage image2, int minDisparity, int maxDisparity, double *matchCost)
{
    int w = image1.width();
    int h = image1.height();

    // for each pixel
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            for(int d = minDisparity; d<maxDisparity; d++)
            {
                // find image 1 and image 2 pixels
                QRgb p1 = image1.pixel(c, r);
                QRgb p2 = qRgb(0, 0, 0);
                int dCol = c-d;
                if(dCol >= 0 && dCol < w)
                {
                    p2 = image2.pixel(dCol, r);
                }

                // compute squared distance on each color channel
                double rdist = std::pow(qRed(p1)-qRed(p2),2.0);
                double gdist = std::pow(qGreen(p1)-qGreen(p2),2.0);
                double bdist = std::pow(qBlue(p1)-qBlue(p2),2.0);

                // combine channels
                double dist = std::sqrt(rdist + gdist + bdist);

                matchCost[(d-minDisparity)*w*h + r*w + c] = dist;
            }
        }
    }
}

/*******************************************************************************
    Compute match cost using Absolute Distance

    image1 - Input image 1
    image2 - Input image 2
    minDisparity - Minimum disparity between image 1 and image 2
    maxDisparity - Minimum disparity between image 1 and image 2
    matchCost - The match cost (absolute distance) between pixels

    To access the match cost at pixel (c, r) at disparity d use
    matchCost[d*w*h + r*w + c]
*******************************************************************************/
void MainWindow::SAD(QImage image1, QImage image2, int minDisparity, int maxDisparity, double *matchCost)
{
    int w = image1.width();
    int h = image1.height();

    // for each pixel
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            for(int d = minDisparity; d<maxDisparity; d++)
            {
                // find image 1 and image 2 pixels
                QRgb p1 = image1.pixel(c, r);
                QRgb p2 = qRgb(0, 0, 0);
                int dCol = c-d;
                if(dCol >= 0 && dCol < w)
                {
                    p2 = image2.pixel(dCol, r);
                }

                // compute absolute distance on each channel
                double rdist = std::abs(qRed(p1)-qRed(p2));
                double gdist = std::abs(qGreen(p1)-qGreen(p2));
                double bdist = std::abs(qBlue(p1)-qBlue(p2));

                // combine distances
                double dist = std::sqrt(std::pow(rdist,2.0) + std::pow(gdist,2.0) + std::pow(bdist,2.0));

                matchCost[(d-minDisparity)*w*h + r*w + c] = dist;
            }
        }
    }
}

/*******************************************************************************
    Compute match cost using Normalized Cross Correlation

    image1 - Input image 1
    image2 - Input image 2
    minDisparity - Minimum disparity between image 1 and image 2
    maxDisparity - Minimum disparity between image 1 and image 2
    radius - Radius of window to compute the NCC score
    matchCost - The match cost (1 - NCC) between pixels

    To access the match cost at pixel (c, r) at disparity d use
    matchCost[d*w*h + r*w + c]
*******************************************************************************/
void MainWindow::NCC(QImage image1, QImage image2, int minDisparity, int maxDisparity, 
    int radius, double *matchCost)
{
    int w = image1.width();
    int h = image1.height();

    // compute 
    double* image1sqred = new double[w*h];
    double* image1sqgreen = new double[w*h];
    double* image1sqblue = new double[w*h];
    double* image2sqred = new double[w*h];
    double* image2sqgreen = new double[w*h];
    double* image2sqblue = new double[w*h];

    // square each pixel in image 1 and image 2
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            QRgb p1 = image1.pixel(c, r);
            QRgb p2 = image2.pixel(c, r);

            image1sqred[r*w + c] = std::pow(qRed(p1),2.0);
            image1sqgreen[r*w + c] = std::pow(qGreen(p1),2.0);
            image1sqblue[r*w + c] = std::pow(qBlue(p1),2.0);

            image2sqred[r*w + c] = std::pow(qRed(p2),2.0);
            image2sqgreen[r*w + c] = std::pow(qGreen(p2),2.0);
            image2sqblue[r*w + c] = std::pow(qBlue(p2),2.0);
        }
    }

    // for each pixel
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            for(int d = minDisparity; d<maxDisparity; d++)
            {
                double distrm = 0.0;
                double distgm = 0.0;
                double distbm = 0.0;

                double distr1s = 0.0;
                double distg1s = 0.0;
                double distb1s = 0.0;

                double distr2s = 0.0;
                double distg2s = 0.0;
                double distb2s = 0.0;

                // scan window based on radius
                for(int rd = -radius; rd <= radius; rd++)
                {
                    for(int cd = -radius; cd <= radius; cd++)
                    {
                        int row = r+rd;
                        int col = c+cd;

                        // find image 1 and image 2 pixels
                        QRgb p1 = qRgb(0, 0, 0);
                        QRgb p2 = qRgb(0, 0, 0);

                        if(row >= 0 && row < h && col >= 0 && col < w)
                        {
                            p1 = image1.pixel(col, row);

                            distr1s += image1sqred[row*w+col];
                            distg1s += image1sqgreen[row*w+col];
                            distb1s += image1sqblue[row*w+col];
                        }

                        if(row >= 0 && row < h && col-d >= 0 && col-d < w)
                        {
                            p2 = image2.pixel(col-d, row);

                            distr2s += image2sqred[row*w+(col-d)];
                            distg2s += image2sqgreen[row*w+(col-d)];
                            distb2s += image2sqblue[row*w+(col-d)];
                        }

                        // sum components of ncc
                        distrm += qRed(p1)*qRed(p2);
                        distgm += qGreen(p1)*qGreen(p2); 
                        distbm += qBlue(p1)*qBlue(p2);  
                    }
                }

                double cost = 1.0;

                // compute ncc on each channel
                if(distr1s != 0.0 && distg1s != 0.0 && distb1s != 0.0 && 
                    distr2s != 0.0 && distg2s != 0.0 && distb2s != 0.0)
                {
                    double nccr = distrm / (std::sqrt(distr1s*distr2s));
                    double nccg = distgm / (std::sqrt(distg1s*distg2s));
                    double nccb = distbm / (std::sqrt(distb1s*distb2s));

                    double ncc = (nccr+nccg+nccb)/3.0;

                    // normalize and compute cost
                    cost = 1 - ncc;
                }

                matchCost[(d-minDisparity)*w*h + r*w + c] = cost;
            }
        }
    }
}

/*******************************************************************************
    Gaussian blur the match score.

    matchCost - The match cost between pixels
    w, h - The width and height of the image
    numDisparities - The number of disparity levels
    sigma - The standard deviation of the blur kernel

    I would recommend using SeparableGaussianBlurImage as a helper function.
*******************************************************************************/
void MainWindow::GaussianBlurMatchScore(double *matchCost, int w, int h, int numDisparities, double sigma)
{
    // run each disparity through gaussian blur
    for(int d = 0; d < numDisparities; d++)
    {
        double* dispImage = &(matchCost[d*w*h + 0*w + 0]);
        SeparableGaussianBlurImage(dispImage, w, h, sigma);
    }
}

/*******************************************************************************
    Blur a floating piont image using Gaussian kernel (helper function for 
    GaussianBlurMatchScore.)

    image - Floating point image
    w, h - The width and height of the image
    sigma - The standard deviation of the blur kernel

    You may just cut and paste code from previous assignment
*******************************************************************************/
void MainWindow::SeparableGaussianBlurImage(double *image, int w, int h, double sigma)
{
    int radius = 3*sigma;
    int size = radius*2 + 1;

    // create kernel
    double* kernel = KernelBuildSeperableGaussian(sigma, radius, size);

    // apply kernel (horizontal and vertical)
    BufferSingleApplyKernel(image, w, h, kernel, 1, size);
    BufferSingleApplyKernel(image, w, h, kernel, size, 1);

    delete [] kernel;
}

/*******************************************************************************
    Bilaterally blur the match score using the colorImage to compute kernel weights

    matchCost - The match cost between pixels
    w, h - The width and height of the image
    numDisparities - The number of disparity levels
    sigmaS, sigmaI - The standard deviation of the blur kernel for spatial and intensity
    colorImage - The color image
*******************************************************************************/
void MainWindow::BilateralBlurMatchScore(
    double *matchCost, int w, int h, int numDisparities,
    double sigmaS, double sigmaI, QImage colorImage)
{
    int radius = 3*sigmaS;
    int size = 2*radius+1;

    // create kernel
    double* kernel = new double[size*size];

    // create copy of match costs
    double* matchCostCopy = new double[numDisparities*h*w];
    for(int i = 0; i<(numDisparities*h*w); i++)
    {
        matchCostCopy[i] = matchCost[i];
    }

    // compute each match cost
    for(int r = 0; r < h; r++)
    {
        for(int c = 0; c < w; c++)
        {          
            double norm = 0.0;

            // get original pixel
            QRgb orgPixel = colorImage.pixel(c, r);

            // generate kernel for pixel
            // convolve around the pixel
            for(int rd=-radius;rd<=radius;rd++)
            {
                for(int cd=-radius;cd<=radius;cd++)
                {
                    int row = r+rd;
                    int col = c+cd;

                    // find convolution pixel
                    QRgb convPixel = qRgb(0,0,0);
                    if(row >= 0 && row < h && col >= 0 && col < w)
                    {
                        convPixel = colorImage.pixel(c+cd, r+rd);
                    }
                    
                    // compute range filter (intensity)
                    double range = std::exp( -std::pow( PixelMagnitude(orgPixel)-PixelMagnitude(convPixel), 2) / (2*std::pow(sigmaI,2)) );

                    // domain filter (guassian)
                    double domain = std::exp( -(std::pow((double)rd,2) + std::pow((double)cd,2)) / (2*std::pow(sigmaS,2)) );

                    // compute weight, norm, rgb values
                    double weight = range*domain;
                    norm += weight;

                    // save weight in kernel
                    kernel[(rd+radius)*size + (cd+radius)] = weight;
                }
            }

            // normalize kernel
            for(int i = 0; i<size*size; i++)
            {
                kernel[i] /= norm;
            }

            // apply kernel to each cost in each disparity
            for(int d = 0; d < numDisparities; d++)
            {
                double cost = 0.0;

                // convolve around the pixel
                for(int rd=-radius;rd<=radius;rd++)
                {
                    for(int cd=-radius;cd<=radius;cd++)
                    {
                        double cCost = 0.0;
                        int row = r+rd;
                        int col = c+cd;

                        // get weight
                        double weight = kernel[(rd+radius)*size + (cd+radius)];
                        
                        // get current cost from copy
                        if(row >= 0 && row < h && col >= 0 && col < w)
                        {
                            cCost = matchCostCopy[d*w*h + row*w + col];
                        }

                        // add weighted cost to total
                        cost += weight*cCost;
                    }
                }

                // save new cost
                matchCost[d*w*h + r*w + c] = cost;
            }
        }
    }

    // cleanup
    delete [] kernel;
    delete [] matchCostCopy;
}

/*******************************************************************************
    Compute the mean color and position for each segment (helper function for Segment.)

    image - Color image
    segment - Image segmentation
    numSegments - Number of segments
    meanSpatial - Mean position of segments
    meanColor - Mean color of segments
*******************************************************************************/
void MainWindow::ComputeSegmentMeans(QImage image, int *segment, int numSegments, 
    double (*meanSpatial)[2], double (*meanColor)[3])
{
    int w = image.width();
    int h = image.height();

    int* numPixels = new int[numSegments];

    // initialize means and pixel count
    for(int i = 0; i < numSegments; i++)
    {
        meanSpatial[i][0] = 0.0;
        meanSpatial[i][1] = 0.0;
        meanColor[i][0] = 0.0;
        meanColor[i][1] = 0.0;
        meanColor[i][2] = 0.0;
        numPixels[i] = 0;
    }

    // compute each match cost
    for(int r = 0; r < h; r++)
    {
        for(int c = 0; c < w; c++)
        {  
            // find segment number for pixel
            int seg = segment[r*w+c];

            QRgb pix = image.pixel(c, r);

            // add means and count
            meanSpatial[seg][0] += r;
            meanSpatial[seg][1] += c;
            meanColor[seg][0] += qRed(pix);
            meanColor[seg][1] += qGreen(pix);
            meanColor[seg][2] += qBlue(pix);
            numPixels[seg]++;
        }
    }

    // compute means
    for(int i = 0; i < numSegments; i++)
    {
        meanSpatial[i][0] /= numPixels[i];
        meanSpatial[i][1] /= numPixels[i];
        meanColor[i][0] /= numPixels[i];
        meanColor[i][1] /= numPixels[i];
        meanColor[i][2] /= numPixels[i];
    }

    delete [] numPixels;
}

/*******************************************************************************
    Assign each pixel to the closest segment using position and color

    image - Color image
    segment - Image segmentation
    numSegments - Number of segments
    meanSpatial - Mean position of segments
    meanColor - Mean color of segments
    spatialSigma - Assumed standard deviation of the spatial distribution of pixels in segment
    colorSigma - Assumed standard deviation of the color distribution of pixels in segment
*******************************************************************************/
void MainWindow::AssignPixelsToSegments(QImage image, int *segment, int numSegments, 
    double (*meanSpatial)[2], double (*meanColor)[3], double spatialSigma, double colorSigma)
{
    int w = image.width();
    int h = image.height();

    double sqSpatialSigma = std::pow(spatialSigma,2.0);
    double sqColorSigma = std::pow(colorSigma,2.0);

    // compute the closest segment for each pixel
    for(int r = 0; r < h; r++)
    {
        for(int c = 0; c < w; c++)
        {
            int seg = -1;
            double segDPos = 0.0;
            double segDColor = 0.0;

            QRgb pixel = image.pixel(c, r);

            // check each segment
            for(int i = 0; i < numSegments; i++)
            {
                // position
                double dy = std::pow( (r-meanSpatial[i][0])/sqSpatialSigma, 2.0);
                double dx = std::pow( (c-meanSpatial[i][1])/sqSpatialSigma, 2.0);
                double dPos = std::sqrt(dy+dx);

                // color
                double dr = std::pow( (qRed(pixel)-meanColor[i][0])/sqColorSigma, 2.0);
                double dg = std::pow( (qGreen(pixel)-meanColor[i][1])/sqColorSigma, 2.0);
                double db = std::pow( (qBlue(pixel)-meanColor[i][2])/sqColorSigma, 2.0);
                double dColor = std::sqrt(dr+dg+db);

                // compare and update
                if( seg == -1 || (dPos+dColor) < (segDPos+segDColor) )
                {
                    seg = i;
                    segDPos = dPos;
                    segDColor = dColor;
                }
            }

            // assign segment
            segment[r*w+c] = seg;
        }
    }
}

/*******************************************************************************
    Update the match cost based on the segmentation. That is, average the match cost
    for each pixel in a segment.

    segment - Image segmentation
    numSegments - Number of segments
    width, height - Width and height of image
    numDisparities - Number of disparities
    matchCost - The match cost between pixels
*******************************************************************************/
void MainWindow::SegmentAverageMatchCost(int *segment, int numSegments,
    int w, int h, int numDisparities, double *matchCost)
{
    // for each segment
    for(int seg = 0; seg < numSegments; seg++)
    {
        // for each disparity 
        for(int d = 0; d < numDisparities; d++)
        {
            double newMatchCost = 0.0;
            int pixelCount = 0;

            // compute new match cost from pixels in the segment
            for(int r = 0; r < h; r++)
            {
                for(int c = 0; c < w; c++)
                {
                    // check if in segment
                    if(segment[r*w+c] == seg)
                    {
                        newMatchCost += matchCost[(d)*w*h + r*w + c];
                        pixelCount++;
                    }
                }
            }

            // average match cost
            newMatchCost /= pixelCount;

            // set match cost for each pixel in the segment
            for(int r = 0; r < h; r++)
            {
                for(int c = 0; c < w; c++)
                {
                    // check if in segment
                    if(segment[r*w+c] == seg)
                    {
                        // set cost
                        matchCost[(d)*w*h + r*w + c] = newMatchCost;
                    }
                }
            }
        }
    }
}

/*******************************************************************************
    For each pixel find the disparity with minimum match cost

    matchCost - The match cost between pixels
    disparities - The disparity for each pixel (use disparity[r*w + c])
    width, height - Width and height of image
    minDisparity - The minimum disparity
    numDisparities - Number of disparities
*******************************************************************************/
void MainWindow::FindBestDisparity(double *matchCost, double *disparities, int w, int h, 
    int minDisparity, int numDisparities)
{
    // for each pixel
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            // find the lowest disparity
            int min = minDisparity;

            for(int d = 0; d<numDisparities; d++)
            {
                // check if cost is lower at this disparity
                if(matchCost[(min-minDisparity)*w*h + r*w + c] > matchCost[(d)*w*h + r*w + c])
                {
                    min = d+minDisparity;
                }
            }

            // save lowest cost disparity
            disparities[r*w + c] = min;
        }
    }
}

/*******************************************************************************
    CIELab Color Space
*******************************************************************************/
struct CIELab {
    double L;
    double a;
    double b;
};

/*******************************************************************************
    Convert each pixel in the image from RGB to CIELab color space
*******************************************************************************/
static CIELab* ConvertImageToLab(QImage image)
{
    int w = image.width();
    int h = image.height();

    CIELab* lab = new CIELab[w*h];

    // for each pixel
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            double L;
            double a;
            double b;

            // get pixel
            QRgb pixel = image.pixel(c, r);

            // convert
            RGBtoLab(qRed(pixel), qGreen(pixel), qBlue(pixel), L, a, b);

            // store
            lab[r*w + c].L = L;
            lab[r*w + c].a = a;
            lab[r*w + c].b = b;
        }
    }

    return lab;
}

/*******************************************************************************
    Compute the support weight for the adaptive support-weight approach.
*******************************************************************************/
static double ComputeSupportWeight(CIELab p, int px, int py, CIELab q, int qx, int qy, double colorWeight, double spatialWeight)
{
    double colorDist = std::sqrt( std::pow(p.L-q.L, 2.0) + std::pow(p.a-q.a, 2.0) + std::pow(p.b-q.b, 2.0) );
    
    //double colorDist = std::sqrt( std::pow(qRed(p)-qRed(q), 2.0) + std::pow(qGreen(p)-qGreen(q), 2.0) + std::pow(qBlue(p)-qBlue(q), 2.0) );

    double spatialDist = std::sqrt( std::pow(px-qx, 2.0) + std::pow(py-qy, 2.0) );

    double result = std::exp( -((colorDist/colorWeight) + (spatialDist/spatialWeight)) );

    return result;
}

/*******************************************************************************
    Convert each pixel in the image into an array of support weights
*******************************************************************************/
static double* ConvertImageToWeights(QImage image, const int radius, const double colorWeight, const double spatialWeight)
{
    int w = image.width();
    int h = image.height();

    int size = radius*2 + 1;

    double* weights = new double[w*h*size*size];

    // convert image to CIELab color space
    CIELab* imageLab = ConvertImageToLab(image);

    // for each pixel
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            // compute entire window
            for(int rd = -radius; rd <= radius; rd++)
            {
                for(int cd = -radius; cd <= radius; cd++)
                {
                    // compute pixel rows and cols
                    int pRow = r;
                    int pCol = c;
                    int qRow = r+rd;
                    int qCol = c+cd;

                    CIELab pLab = {0.0, 0.0, 0.0};
                    CIELab qLab = {0.0, 0.0, 0.0};

                    // get pixel values
                    if(pRow >= 0 && pRow < h && pCol >= 0 && pCol < w)
                        pLab = imageLab[pRow*w + pCol];

                    if(qRow >= 0 && qRow < h && qCol >= 0 && qCol < w)
                        qLab = imageLab[qRow*w + qCol];

                    // compute support weight
                    double sWeight = ComputeSupportWeight(pLab, pCol, pRow, qLab, qCol, qRow, colorWeight, spatialWeight);

                    int windowPos = (rd+radius)*size + (cd+radius);
                    weights[r*w + c + windowPos] = sWeight;
                }
            }
        }
    }

    delete [] imageLab;

    return weights;
}

/*******************************************************************************
    Compute the raw matching cost between image 1 and image 2 window pixels
*******************************************************************************/
static double ComputeRawCost(QRgb q1, QRgb q2, double trunc)
{
    double redCost = std::abs(qRed(q1)-qRed(q2));
    double greenCost = std::abs(qGreen(q1)-qGreen(q2));
    double blueCost = std::abs(qBlue(q1)-qBlue(q2));

    double cost = std::min( (redCost+greenCost+blueCost), trunc );

    return cost;
}

/*******************************************************************************
    Create your own "magic" stereo algorithm

    image1 - Input image 1
    image2 - Input image 2
    minDisparity - Minimum disparity between image 1 and image 2
    maxDisparity - Minimum disparity between image 1 and image 2
    param1 - The first parameter to your algorithm
    param2 - The second paramater to your algorithm
    matchCost - The match cost (squared distance) between pixels
*******************************************************************************/
void MainWindow::MagicStereo(QImage image1, QImage image2, int minDisparity, int maxDisparity, 
    double param1, double param2, double *matchCost)
{
    int w = image1.width();
    int h = image1.height();

    int radius = 10;        // optimal window = 35x35
    int size = radius*2+1;  // window size
    double trunc = 40;      // matching cost truncation

    // convert images to support weights
    double* image1SupportWeights = ConvertImageToWeights(image1, radius, param1, param2);
    double* image2SupportWeights = ConvertImageToWeights(image2, radius, param1, param2);

    // for each pixel
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            for(int d = minDisparity; d<maxDisparity; d++)
            {
                double sumNumerator = 0.0;
                double sumDenominator = 0.0;

                // scan window based on radius
                for(int rd = -radius; rd <= radius; rd++)
                {
                    for(int cd = -radius; cd <= radius; cd++)
                    {
                        // compute pixel rows and cols
                        int image1pRow = r;
                        int image1pCol = c;
                        int image1qRow = r+rd;
                        int image1qCol = c+cd;

                        int image2pRow = r;
                        int image2pCol = c-d;
                        int image2qRow = r+rd;
                        int image2qCol = c+cd-d;

                        // get image 1 and image 2 weights
                        double refWeight = 0.0;
                        double tarWeight = 0.0;
                        
                        int windowPos = (rd+radius)*size + (cd+radius);
                        
                        // reference weight
                        if(image1pRow >= 0 && image1pRow < h && image1pCol >= 0 && image1pCol < w)
                            refWeight = image1SupportWeights[(image1pRow*w + image1pCol) + windowPos];

                        // target weight
                        if(image2pRow >= 0 && image2pRow < h && image2pCol >= 0 && image2pCol < w)
                            tarWeight = image2SupportWeights[(image2pRow*w + image2pCol) + windowPos];

                        // find image 1 and image 2 q pixels
                        QRgb image1q = qRgb(0, 0, 0);
                        QRgb image2q = qRgb(0, 0, 0);

                        // reference window q pixel
                        if(image1qRow >= 0 && image1qRow < h && image1qCol >= 0 && image1qCol < w)
                            image1q = image1.pixel(image1qCol, image1qRow);

                        // target window q pixel
                        if(image2qRow >= 0 && image2qRow < h && image2qCol >= 0 && image2qCol < w)
                            image2q = image2.pixel(image2qCol, image2qRow);

                        // compute raw matching cost between q pixels
                        double redCost = std::abs(qRed(image1q)-qRed(image2q));
                        double greenCost = std::abs(qGreen(image1q)-qGreen(image2q));
                        double blueCost = std::abs(qBlue(image1q)-qBlue(image2q));

                        double rawCost = std::min( (redCost+greenCost+blueCost), trunc );

                        // add costs
                        double combinedWeights = refWeight*tarWeight;
                        sumNumerator += (combinedWeights*rawCost);
                        sumDenominator += combinedWeights;
                    }
                }

                // compute final cost
                double cost = 0.0;
                if(sumDenominator != 0.0)
                    cost = sumNumerator / sumDenominator;

                // save cost
                matchCost[(d-minDisparity)*w*h + r*w + c] = cost;
            }
        }
    }
    
    delete [] image1SupportWeights;
    delete [] image2SupportWeights;
}
