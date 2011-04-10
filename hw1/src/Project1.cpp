#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <cmath>
#include <iostream>

// ----------------------------------------------------------------------------
// This is the only file you need to change for your assignment.  The
// other files control the UI (in case you want to make changes.)
// ----------------------------------------------------------------------------

// Pixel
struct Pixel {
    double r;
    double g;
    double b;
};

// ----------------------------------------------------------------------------
// HELPER METHODS
// ----------------------------------------------------------------------------

// Get the pixel at the specified row and column using image width and padding
// Buffer width is calculated as image width plus 2x padding 
static Pixel* BufferGetPixel(Pixel* buffer, const int& imgWidth, const int& padding, const int& row, const int& col)
{
    int bWidth = imgWidth + (2*padding);
    Pixel* p = &buffer[ (row*bWidth) + col ];
    return p;
}

// Creates a copy of a pixel buffer.
// Buffer copy must be freed by caller.
static Pixel* BufferCreateCopy(Pixel *buffer, const int& imgWidth, const int& imgHeight, const int& padding)
{
    // initialize new buffer
    int bWidth = imgWidth+(padding*2);
    int bHeight = imgHeight+(padding*2);

    Pixel *newBuffer = new Pixel[ bWidth * bHeight ];

    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            Pixel *newPixel = BufferGetPixel(newBuffer, imgWidth, padding, r, c);
            Pixel *orgPixel = BufferGetPixel(buffer, imgWidth, padding, r, c);
            *newPixel = *orgPixel;
        }
    }

    return newBuffer;
}

// Applies a kernel filter to a pixel buffer. Values are updated in the buffer.
// The kernel can be any odd numbered height and width in size.
static void BufferApplyKernel(
    Pixel *buffer, const int& imgWidth, const int& imgHeight, const int& padding, 
    const double* kernel, const int kHeight, const int kWidth)
{
    // compute horizontal and vertical kernel radius
    int kHeightRadius = kHeight/2;
    int kWidthRadius = kWidth/2;

    int kHeightPaddingOffset = padding - kHeightRadius;
    int kWidthPaddingOffset = padding - kWidthRadius;

    // create copy of original buffer
    Pixel* copyBuffer = BufferCreateCopy(buffer, imgWidth, imgHeight, padding);

    // for each pixel in the image
    for(int r=0;r<imgHeight;r++)
    {
        for(int c=0;c<imgWidth;c++)
        {
            double rgb[3];

            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;

            // convolve the kernel at each pixel
            for(int rd=-kHeightRadius; rd<=kHeightRadius; rd++)
            {
                for(int cd=-kWidthRadius; cd<=kWidthRadius; cd++)
                {
                    // get the original pixel value from copy buffer
                    int pRow = r + (rd + kHeightRadius) + kHeightPaddingOffset;
                    int pCol = c + (cd + kWidthRadius) + kWidthPaddingOffset;

                    Pixel* p = BufferGetPixel( copyBuffer, imgWidth, padding, pRow, pCol );

                    // get the value of the kernel
                    double weight = kernel[(rd + kHeightRadius)*kWidth + cd + kWidthRadius];

                    // apply weights
                    rgb[0] += weight*p->r;
                    rgb[1] += weight*p->g;
                    rgb[2] += weight*p->b;
                }
            }

            // store mean pixel in the buffer
            Pixel* pOut = BufferGetPixel( buffer, imgWidth, padding, r+padding, c+padding);
            pOut->r = rgb[0];
            pOut->g = rgb[1];
            pOut->b = rgb[2];
        }
    }
}

// Add a color offset value to every pixel int eh buffer
static void BufferApplyOffset(Pixel *buffer, const int& imgWidth, const int& imgHeight, const int& padding, QRgb offset)
{
    // compute size
    int bWidth = imgWidth+(padding*2);
    int bHeight = imgHeight+(padding*2);

    // add offsets
    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            Pixel* p = BufferGetPixel(buffer, imgWidth, padding, r, c);
            p->r += qRed(offset);
            p->g += qGreen(offset);
            p->b += qBlue(offset);
        }
    }
}

// Adds a buffer of pixel values to an existing buffer
// Buffers must be the exact same size
static void BufferAddBuffer(Pixel *buffer, const int& imgWidth, const int& imgHeight, const int& padding, Pixel *vBuffer)
{
    // compute size
    int bWidth = imgWidth+(padding*2);
    int bHeight = imgHeight+(padding*2);

    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            Pixel* p = BufferGetPixel(buffer, imgWidth, padding, r, c);
            Pixel* vp = BufferGetPixel(buffer, imgWidth, padding, r, c);
            p->r += vp->r;
            p->g += vp->g;
            p->b += vp->b;
        }
    }
}

// Subtracts a buffer of pixel values from an existing buffer
// Buffers must be the exact same size
static void BufferSubtractBuffer(Pixel *buffer, const int& imgWidth, const int& imgHeight, const int& padding, Pixel *vBuffer)
{
    // compute size
    int bWidth = imgWidth+(padding*2);
    int bHeight = imgHeight+(padding*2);

    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            Pixel* p = BufferGetPixel(buffer, imgWidth, padding, r, c);
            Pixel* vp = BufferGetPixel(buffer, imgWidth, padding, r, c);
            p->r -= vp->r;
            p->g -= vp->g;
            p->b -= vp->b;
        }
    }
}

// Creates an image buffer with the specified amount of padding on the borders
// Border padding uses reflected pixels
static Pixel* ImageCreateBuffer(QImage *image, const int& padding)
{
    // initialize buffer
    int imgHeight = image->height();
    int imgWidth = image->width();
    int bHeight = imgHeight+(padding*2);
    int bWidth = imgWidth+(padding*2);

    Pixel *buffer = new Pixel[ bWidth * bHeight ];

    // set each pixel
    for(int r = -padding; r < (imgHeight + padding); r++)
    {
        for(int c = -padding; c < (imgWidth + padding); c++)
        {
            // find pixel in buffer that corresponds to the image
            Pixel *p = BufferGetPixel(buffer, imgWidth, padding, r+padding, c+padding);

            int row = r;
            int col = c;

            // reflect row and column entires that outside the image
            if(row < 0) row = -row;
            if(row >= imgHeight) row = imgHeight - (row-imgHeight) - 1;
            if(col < 0) col = -col;
            if(col >= imgWidth) col = imgWidth - (col-imgWidth) - 1;

            // copy rgb values
            p->r = qRed(image->pixel(col, row));
            p->g = qGreen(image->pixel(col, row));
            p->b = qBlue(image->pixel(col, row));
        }
    }

    return buffer;
}

// Convert the pixel buffer back into the original image
// Pixels outside the 0-255 range are truncated
static void ImageConvertBuffer(QImage *image, Pixel *buffer, const int& padding)
{
    for(int r = 0; r < image->height(); r++)
    {
        for(int c = 0; c < image->width(); c++)
        {
            // find pixel in buffer that corresponds to the image
            Pixel *p = BufferGetPixel(buffer, image->width(), padding, r+padding, c+padding);

            // convert to integer value and truncate
            int red = (int)floor(p->r + 0.5);
            int green = (int)floor(p->g + 0.5);
            int blue = (int)floor(p->b + 0.5);
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

// ----------------------------------------------------------------------------
// KERNEL METHODS
// ----------------------------------------------------------------------------

// Convolves two kernels together creating a new kernel
// Both kernels must be square and have an odd size. The output kernel will be the same size as kernel 1.
static double* KernelConvolve(const double const* k1, const int& k1size, const double const* k2, const int& k2size)
{
    // allocate output kernel
    double* outKernel = new double[k1size*k1size];

    int k2radius = k2size / 2;

    // for each weight in k1
    for(int r =0; r<k1size; r++)
    {
        for(int c = 0; c<k1size; c++)
        {
            double value = 0.0;

            // convolve with k2 weights
            for(int rd = -k2radius; rd<=k2radius; rd++)
            {
                for(int cd = -k2radius; cd<=k2radius; cd++)
                {
                    // find value in k1
                    int vRow = r + rd;
                    int vCol = c + cd;

                    // if outside k1, default to value of 0.0
                    double orgValue = 0.0;
                    if(vRow >= 0 && vRow < k1size && vCol >= 0 && vCol < k1size)
                    {
                        orgValue = k1[vRow*k1size + vCol];
                    }

                    // get weight in k2
                    double weight = k2[ (rd+k2radius)*k2size + (cd+k2radius) ];

                    // add weighted value to total new value
                    value += weight*orgValue;
                }
            }

            // set new value in output kernel
            outKernel[r*k1size + c] = value;
        }
    }

    return outKernel;
}

// Build a guassian kernel with the specified sigma, radius and size
// Size must be 2*radius+1
static double* KernelBuildGaussian(const double& sigma, const int& radius, const int& size)
{
    // create kernel to convolve with the image
    double *kernel = new double [size*size];

    // compute kernel weights and z normalization
    double znorm = 0.0;
    for(int x=-radius; x<=radius; x++)
    {
        for(int y=-radius; y<=radius; y++)
        {
            double value = std::exp( -(std::pow((double)x,2) + std::pow((double)y,2)) / (2*std::pow(sigma,2)) );
            kernel[(x+radius)*size+(y+radius)] = value;
            znorm += value;
        }
    }

    // normalize kernel
    for(int x=-radius; x<=radius; x++)
    {
        for(int y=-radius; y<=radius; y++)
        {
            kernel[(x+radius)*size+(y+radius)] /= znorm;
        }
    }

    return kernel;
}

// Build a horizontal/vertical gaussian kernel with the specified sigma, radius, and size
// Size must be 2*radius+1
static double* KernelBuildSeperableGaussian(const double& sigma, const int& radius, const int& size) 
{
    // create kernel
    double *kernel = new double [size];

    // compute kernel weights and z normalization
    double znorm = 0.0;
    for(int x=-radius; x<=radius; x++)
    {
        double value = std::exp( -std::pow((double)x,2) / (2*std::pow(sigma,2)) );
        kernel[x+radius] = value;
        znorm += value;
    }

    // normalize kernel
    for(int x=-radius; x<=radius; x++)
    {
        kernel[x+radius] /= znorm;
    }

    return kernel;
}

// Build gaussian second derivative kernel
// size must be 2*radius+1
static double* KernelBuildSecDervGaussian(const double& sigma, const int& radius, const int& size) 
{
    // create standard guassian kernel
    double *gKernel = KernelBuildGaussian(sigma, radius, size);

    // build mexican hat (second derivative) kernel
    int sdKernelSize = 5;
    double *sdKernel = new double[sdKernelSize*sdKernelSize];

    // initialize second derivative kernel
    for(int i = 0; i < sdKernelSize*sdKernelSize; i++)
    {
        sdKernel[i] = 0.0;
    }

    // fixed values
    sdKernel[7] = 1;
    sdKernel[11] = 1;
    sdKernel[12] = -4;
    sdKernel[13] = 1;
    sdKernel[17] = 1;

    // convolve kernels
    double* finalKernel = KernelConvolve(gKernel, size, sdKernel, sdKernelSize);

    // clean up
    delete [] gKernel;
    delete [] sdKernel;

    return finalKernel;
}

// ----------------------------------------------------------------------------
// IMAGE METHODS
// ----------------------------------------------------------------------------

// Convert an image to grey-scale
void MainWindow::BlackWhiteImage(QImage *image)
{
    int r, c;
    QRgb pixel;

    for(r=0;r<image->height();r++)
    {
        for(c=0;c<image->width();c++)
        {
            pixel = image->pixel(c, r);
            double red = (double) qRed(pixel);
            double green = (double) qGreen(pixel);
            double blue = (double) qBlue(pixel);

            // Compute intensity from colors - these are common weights
            double intensity = 0.3*red + 0.6*green + 0.1*blue;

            image->setPixel(c, r, qRgb( (int) intensity, (int) intensity, (int) intensity));
        }
    }
}

// Add random noise to the image
void MainWindow::AddNoise(QImage *image, double mag, bool colorNoise)
{
    int r, c;
    QRgb pixel;
    int noiseMag = mag;
    noiseMag *= 2;

    for(r=0;r<image->height();r++)
    {
        for(c=0;c<image->width();c++)
        {
            pixel = image->pixel(c, r);
            int red = qRed(pixel);
            int green = qGreen(pixel);
            int blue = qBlue(pixel);

            // If colorNoise, add color independently to each channel
            if(colorNoise)
            {
                red += rand()%noiseMag - noiseMag/2;
                green += rand()%noiseMag - noiseMag/2;
                blue += rand()%noiseMag - noiseMag/2;
            }
            // otherwise add the same amount of noise to each channel
            else
            {
                int noise = rand()%noiseMag - noiseMag/2;

                red += noise;
                green += noise;
                blue += noise;
            }

            // Make sure we don't over or under saturate
            red = min(255, max(0, red));
            green = min(255, max(0, green));
            blue = min(255, max(0, blue));

            image->setPixel(c, r, qRgb( red, green, blue));
        }
    }
}

// Here is an example of blurring an image using a mean or box filter with the specified radius.
// This could be implemented using separable filters to make it much more efficient, but it is not.
void MainWindow::MeanBlurImage(QImage *image, int radius)
{
    if(radius == 0)
        return;

    int r, c, rd, cd, i;
    QRgb pixel;

    // This is the size of the kernel
    int size = 2*radius + 1;

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    QImage buffer;
    int w = image->width();
    int h = image->height();

    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders.
    // This could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);

    // Compute kernel to convolve with the image.
    double *kernel = new double [size*size];

    for(i=0;i<size*size;i++)
    {
        kernel[i] = 1.0;
    }

    // Make sure kernel sums to 1
    double denom = 0.000001;
    for(i=0;i<size*size;i++)
        denom += kernel[i];
    for(i=0;i<size*size;i++)
        kernel[i] /= denom;

    // For each pixel in the image...
    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double rgb[3];

            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;

            // Convolve the kernel at each pixel
            for(rd=-radius;rd<=radius;rd++)
                for(cd=-radius;cd<=radius;cd++)
                {
                     // Get the pixel value
                     pixel = buffer.pixel(c + cd + radius, r + rd + radius);

                     // Get the value of the kernel
                     double weight = kernel[(rd + radius)*size + cd + radius];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }

            // Store mean pixel in the image to be returned.
            image->setPixel(c, r, qRgb((int) floor(rgb[0] + 0.5), (int) floor(rgb[1] + 0.5), (int) floor(rgb[2] + 0.5)));
        }
    }

    // Clean up.
    delete [] kernel;
}

// Downsample the image by 1/2
void MainWindow::HalfImage(QImage &image)
{
    QImage buffer;
    int w = image.width();
    int h = image.height();
    int r, c;

    buffer = image.copy();

    // Reduce the image size.
    image = QImage(w/2, h/2, QImage::Format_RGB32);

    // Copy every other pixel
    for(r=0;r<h/2;r++)
        for(c=0;c<w/2;c++)
        {
             image.setPixel(c, r, buffer.pixel(c*2, r*2));
        }
}

// Gaussian Blur Image
// Create kernel based on gaussian equation and interate through each pixel in the image
// blurring it with other pixels nearby.
void MainWindow::GaussianBlurImage(QImage *image, double sigma)
{
    int x, y;

    // kernel radius and size
    int radius = 3*sigma;
    int size = 2*radius + 1;

    // create kernel to convolve with the image
    double *kernel = KernelBuildGaussian(sigma, radius, size);

    // create pixel buffer
    Pixel* buffer = ImageCreateBuffer(image, radius);

    // apply kernel to buffer
    BufferApplyKernel(buffer, image->width(), image->height(), radius, kernel, size, size);

    // store buffer to image
    ImageConvertBuffer(image, buffer, radius);

    // clean up
    delete [] buffer;
    delete [] kernel;
}

// Seperable Gaussian Blur Image
// Seperates Gaussian algorithm into horizonal and vertical components. Apply each component
// to the image to create the same blurring effect.
void MainWindow::SeparableGaussianBlurImage(QImage *image, double sigma)
{
    int x, y;

    // kernel radius and size
    int radius = 3*sigma;
    int size = 2*radius + 1;

    // create kernels to convolve with the image
    // the same kernel is used for horizontal and vertical convolution
    double *kernel = KernelBuildSeperableGaussian(sigma, radius, size);

    // create pixel buffer
    Pixel* buffer = ImageCreateBuffer(image, radius);

    // apply kernel to buffer in both horizonal and vertical directions
    BufferApplyKernel(buffer, image->width(), image->height(), radius, kernel, 1, size);
    BufferApplyKernel(buffer, image->width(), image->height(), radius, kernel, size, 1);

    // store buffer to image
    ImageConvertBuffer(image, buffer, radius);

    // clean up
    delete [] buffer;
    delete [] kernel;
}

// FirstDerivImage
// Compute first derivative guassian on the image. The first derivative is created by
// computing and applying a horizontal kernel and then blurring with the guassian.
void MainWindow::FirstDerivImage(QImage *image, double sigma)
{
    // kernel radius and size
    int radius = 3*sigma;
    int size = 2*radius + 1;

    // create the seperable guassian kernel
    double* gKernel = KernelBuildSeperableGaussian(sigma, radius, size);

    // create first derivative horizontal to convolve with the image
    double* dKernel = new double[size];

    // compute horizontal first derivative kernel weights
    for(int x=-radius; x<=radius; x++)
    {
        // first derivate equation
        // ( x / sigma^2 ) * ( e ^ (-x^2 / 2*sigma^2) )
        double value = (x / (std::pow(sigma,2))) * (std::exp(-std::pow((double)x,2) / (2*std::pow(sigma,2))));
        dKernel[x+radius] = value;
    }

    // create pixel buffer
    Pixel* buffer = ImageCreateBuffer(image, radius);

    // apply first derivative kernel in horizonal direction
    BufferApplyKernel(buffer, image->width(), image->height(), radius, dKernel, 1, size);

    // apply seperable gaussian kernel to buffer in both horizonal and vertical directions
    BufferApplyKernel(buffer, image->width(), image->height(), radius, gKernel, 1, size);
    BufferApplyKernel(buffer, image->width(), image->height(), radius, gKernel, size, 1);

    // add image offset for negative values
    BufferApplyOffset(buffer, image->width(), image->height(), radius, qRgb(128, 128, 128));

    // store buffer to image
    ImageConvertBuffer(image, buffer, radius);
   
    // clean up
    delete [] buffer;
    delete [] gKernel;
    delete [] dKernel;
}

// SecondDerivImage
// Compute guassian second derivative on the image. The second derivative kernel is computed
// in both directions and blurred with the guassian.
void MainWindow::SecondDerivImage(QImage *image, double sigma)
{
    // kernel radius and size
    int radius = 3*sigma;
    int size = 2*radius + 1;

    // create the seperable guassian kernel
    double* gKernel = KernelBuildSeperableGaussian(sigma, radius, size);

    // create second derivative kernel to convolve with the image
    double *sdKernel = new double[5*5];

    // compute second derivative kernel weights
    //for(int x=-radius; x<=radius; x++)
    //{
    //    // second derivate equation
    //    // ( ( (x^2 - sigma^2) / 2*sigma^2 ) * ( e ^ (-x^2 / 2*sigma^2) )
    //    double value = ((std::pow((double)x,2) - std::pow(sigma,2)) / (2*std::pow(sigma,2))) * (std::exp(-std::pow((double)x,2) / (2*std::pow(sigma,2))));
    //    sdKernel[x+radius] = value;
    //}

    //for(int x=-radius; x<=radius; x++)
    //{
    //    for(int y=-radius; y<=radius; y++)
    //    {
    //        double value = (1/std::pow(sigma,3))*(2-((std::pow((double)x,2)+std::pow((double)y,2))/(2*std::pow(sigma,2))))*
    //            std::exp( -(std::pow((double)x,2) + std::pow((double)y,2)) / (2*std::pow(sigma,2)) );
    //        sdKernel[(x+radius)*size+(y+radius)] = value;
    //    }
    //}

    double *kernel = KernelBuildSecDervGaussian(sigma, radius, size);

    // create pixel buffer
    Pixel* buffer = ImageCreateBuffer(image, radius);

    // apply second derivative kernel
    BufferApplyKernel(buffer, image->width(), image->height(), radius, kernel, size, size);
    //BufferApplyKernel(buffer, image->width(), image->height(), radius, sdKernel, size, 1);

    // apply seperable gaussian kernel to buffer in both horizonal and vertical directions
    //BufferApplyKernel(buffer, image->width(), image->height(), radius, gKernel, 1, size);
    //BufferApplyKernel(buffer, image->width(), image->height(), radius, gKernel, size, 1);

    // add image offset for negative values
    BufferApplyOffset(buffer, image->width(), image->height(), radius, qRgb(128, 128, 128));

    // store buffer to image
    ImageConvertBuffer(image, buffer, radius);

    // clean up
    delete [] buffer;
    delete [] gKernel;
    delete [] sdKernel;
}

// SharpenImage
// Sharpen image by subtracting second derivative from original image
void MainWindow::SharpenImage(QImage *image, double sigma, double alpha)
{
    // kernel radius and size
    int radius = 3*sigma;
    int size = 2*radius + 1;

    // create second derivative pixel buffer
    Pixel* sdBuffer = ImageCreateBuffer(image, radius);

    // create second derivative kernel
    double *kernel = KernelBuildSecDervGaussian(sigma, radius, size);

    // apply second derivative kernel
    BufferApplyKernel(sdBuffer, image->width(), image->height(), radius, kernel, size, size);


    // create new pixel buffer
    Pixel* buffer = ImageCreateBuffer(image, radius);

    // subtract second derivative from buffer
    BufferAddBuffer(buffer, image->width(), image->height(), radius, sdBuffer);

    // store buffer to image
    ImageConvertBuffer(image, buffer, radius);

    delete [] kernel;
    delete [] sdBuffer;
    delete [] buffer;
}

void MainWindow::BilateralImage(QImage *image, double sigmaS, double sigmaI)
{
    // Add your code here.  Should be similar to GaussianBlurImage.
}

void MainWindow::SobelImage(QImage *image)
{
    // Add your code here.

    /***********************************************************************
      When displaying the orientation image I
      recommend the following:

    double mag; // magnitude of the gradient
    double orien; // orientation of the gradient

    double red = (sin(orien) + 1.0)/2.0;
    double green = (cos(orien) + 1.0)/2.0;
    double blue = 1.0 - red - green;

    red *= mag*4.0;
    green *= mag*4.0;
    blue *= mag*4.0;

    // Make sure the pixel values range from 0 to 255
    red = min(255.0, max(0.0, red));
    green = min(255.0, max(0.0, green));
    blue = min(255.0, max(0.0, blue));

    image->setPixel(c, r, qRgb( (int) (red), (int) (green), (int) (blue)));

    ************************************************************************/
}


void MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{
    // Add your code here.  Return the RGB values for the pixel at location (x,y) in double rgb[3].
}

// Here is some sample code for rotating an image.  I assume orien is in degrees.

void MainWindow::RotateImage(QImage *image, double orien)
{
    int r, c;
    QRgb pixel;
    QImage buffer;
    int w = image->width();
    int h = image->height();
    double radians = -2.0*3.141*orien/360.0;

    buffer = image->copy();

    pixel = qRgb(0, 0, 0);
    image->fill(pixel);

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double rgb[3];
            double x0, y0;
            double x1, y1;

            // Rotate around the center of the image.
            x0 = (double) (c - w/2);
            y0 = (double) (r - h/2);

            // Rotate using rotation matrix
            x1 = x0*cos(radians) - y0*sin(radians);
            y1 = x0*sin(radians) + y0*cos(radians);

            x1 += (double) (w/2);
            y1 += (double) (h/2);

            BilinearInterpolation(&buffer, x1, y1, rgb);

            image->setPixel(c, r, qRgb((int) floor(rgb[0] + 0.5), (int) floor(rgb[1] + 0.5), (int) floor(rgb[2] + 0.5)));
        }
    }

}

void MainWindow::FindPeaksImage(QImage *image, double thres)
{
    // Add your code here.
}


void MainWindow::MedianImage(QImage *image, int radius)
{
    // Add your code here
}

void MainWindow::HoughImage(QImage *image)
{
    // Add your code here
}

void MainWindow::CrazyImage(QImage *image)
{
    // Add your code here
}
