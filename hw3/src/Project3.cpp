#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>

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
                if(c-d >= 0)
                {
                    p2 = image2.pixel(c-d, r);
                }

                // compute squared distance on each color channel
                double rdist = std::pow(qRed(p1)-qRed(p2),2.0);
                double gdist = std::pow(qGreen(p1)-qGreen(p2),2.0);
                double bdist = std::pow(qBlue(p1)-qBlue(p2),2.0);

                // combine channels
                double dist = rdist+gdist+bdist;

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
                if(c-d >= 0)
                {
                    p2 = image2.pixel(c-d, r);
                }

                // compute absolute distance on each channel
                double rdist = std::abs(qRed(p1)-qRed(p2));
                double gdist = std::abs(qGreen(p1)-qGreen(p2));
                double bdist = std::abs(qBlue(p1)-qBlue(p2));

                // combine distances
                double dist = rdist+gdist+bdist;

                matchCost[(d-minDisparity)*w*h + r*w + c] = dist;
            }
        }
    }
}

double ComputeNCC(double v1, double v2)
{
    if(v1 == 0 || v2 == 0)
    {
        return 0.0;
    }
    else
    {
        double out = (v1*v2) / (std::sqrt(std::pow(v1,2)*std::pow(v2,2)));
        return out;
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

                // compute ncc on each channel
                double nccr = distrm / (std::sqrt(distr1s*distr2s));
                double nccg = distgm / (std::sqrt(distg1s*distg2s));
                double nccb = distbm / (std::sqrt(distb1s*distb2s));

                double ncc = (nccr+nccg+nccb)/3.0;

                // normalize and compute cost
                double cost = 1 - ncc;

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
    // Add your code here
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
    // Add your code here
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
    // Add your code here
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
    // Add your code here
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
    // Add your code here
}

/*******************************************************************************
    Update the match cost based ont eh segmentation.  That is, average the match cost
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
    // Add your code here
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
                if(matchCost[(min)*w*h + r*w + c] > matchCost[(d)*w*h + r*w + c])
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
    // Add your code here

}
