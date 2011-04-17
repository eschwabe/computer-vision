#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

#define _DEBUG_ 1

#ifdef _DEBUG_
#define NOMINMAX
#include <Windows.h>
#endif

// ----------------------------------------------------------------------------
// This is the only file you need to change for your assignment.  The
// other files control the UI (in case you want to make changes.)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// HELPER METHODS
// ----------------------------------------------------------------------------

// Compute pixel magnitude
static double PixelMagnitude(const Pixel *p)
{
    double mag = (p->r + p->g + p->b) / (3);
    return mag;
}

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

// Adds a buffer of pixel values to an existing buffer multiplied by alpha
// Buffers must be the exact same size
static void BufferAddBuffer(Pixel *buffer, const int& imgWidth, const int& imgHeight, const int& padding, 
    Pixel *addBuffer, const double& alpha)
{
    // compute size
    int bWidth = imgWidth+(padding*2);
    int bHeight = imgHeight+(padding*2);

    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            Pixel* p = BufferGetPixel(buffer, imgWidth, padding, r, c);
            Pixel* vp = BufferGetPixel(addBuffer, imgWidth, padding, r, c);
            p->r += vp->r*alpha;
            p->g += vp->g*alpha;
            p->b += vp->b*alpha;
        }
    }
}

// Subtracts a buffer of pixel values from an existing buffer multiplied by alpha
// Buffers must be the exact same size
static void BufferSubtractBuffer(Pixel *buffer, const int& imgWidth, const int& imgHeight, const int& padding, 
    Pixel *subBuffer, const double& alpha)
{
    // compute size
    int bWidth = imgWidth+(padding*2);
    int bHeight = imgHeight+(padding*2);

    for(int r = 0; r < bHeight; r++)
    {
        for(int c = 0; c < bWidth; c++)
        {
            Pixel* p = BufferGetPixel(buffer, imgWidth, padding, r, c);
            Pixel* vp = BufferGetPixel(subBuffer, imgWidth, padding, r, c);
            p->r -= vp->r*alpha;
            p->g -= vp->g*alpha;
            p->b -= vp->b*alpha;
        }
    }
}

// Creates an image buffer with the specified amount of padding on the borders
// Border padding uses reflected pixels
static Pixel* ImageCreateBuffer(QImage *image, const int& padding, const bool& reflect=true)
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

            // initialize pixel
            p->r = 0.0;
            p->g = 0.0;
            p->b = 0.0;

            int row = r;
            int col = c;

            // reflect row and column entires that outside the image
            if(reflect == true)
            {
                if(row < 0) row = -row;
                if(row >= imgHeight) row = imgHeight - (row-imgHeight) - 1;
                if(col < 0) col = -col;
                if(col >= imgWidth) col = imgWidth - (col-imgWidth) - 1;
            }

            // check if pixels in range
            if(row >= 0 && row < imgHeight && col >= 0 && col < imgWidth)
            {
                // copy rgb values
                p->r = qRed(image->pixel(col, row));
                p->g = qGreen(image->pixel(col, row));
                p->b = qBlue(image->pixel(col, row));
            }
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

// Convert a kernel into an image
// Kernel weights are multipled to make more visable. Image is assumed to be large enough.
static void ImageConvertKernel(QImage* image, double* kernel, const int& size)
{
    // clear image
    image->fill(qRgb(255, 255, 255));

    // output kernel
    for(int r = 0; r < size; r++)
    {
        for(int c = 0; c < size; c++)
        {
            // convert to integer value and truncate
            int color = (int)floor( (kernel[(r*size)+c]*2000) + 128);
            color = std::max(color,0);
            color = std::min(color,255);

            // set pixel
            image->setPixel(c+10, r+10, qRgb(color, color, color));
        }
    }
}

// ----------------------------------------------------------------------------
// KERNEL METHODS
// ----------------------------------------------------------------------------

// Convolves two kernels together creating a new kernel
// Both kernels must be square and have an odd size. The output kernel will be the same size as kernel 1.
static double* KernelConvolve(const double * k1, const int& k1size, const double * k2, const int& k2size)
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

// Normalize kernel values so the sum of the absolute values is 1
// Kernel must be square in shape
static void KernelNormalize(double* kernel, const int& size)
{
    double norm = 0.0;

    // compute normal
    for(int i=0; i<size*size; i++)
    {
        norm += std::abs(kernel[i]);
    }

    // normalize kernel
    double net = 0.0;
    for(int i=0; i<size*size; i++)
    {
        kernel[i] /= norm;
        net += kernel[i];
    }

#ifdef _DEBUG_
    std::stringstream s;
    s << "Net: " << net << std::endl;
    OutputDebugStringA(s.str().c_str());
#endif
}

// Prints a kernel to the debug output
// Kernel is assumed to be square
static void KernelPrint(const double * kernel, const int& size)
{
#ifdef _DEBUG_
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            std::stringstream s;
            s.width(8);
            s.precision(5);
            s << std::fixed << kernel[(size*i)+j] << " ";
            OutputDebugStringA(s.str().c_str());
        }
        OutputDebugStringA("  \n");
    }
#endif
}

// Build a guassian kernel with the specified sigma, radius and size
// Size must be 2*radius+1
static double* KernelBuildGaussian(const double& sigma, const int& radius, const int& size)
{
    // create kernel to convolve with the image
    double *kernel = new double [size*size];

    // compute kernel weights
    for(int x=-radius; x<=radius; x++)
    {
        for(int y=-radius; y<=radius; y++)
        {
            double value = std::exp( -(std::pow((double)x,2) + std::pow((double)y,2)) / (2*std::pow(sigma,2)) );
            kernel[(x+radius)*size+(y+radius)] = value;
        }
    }

    // normalize
    KernelNormalize(kernel, size);

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

// Build guassian first derivative kernal
// Size must be 2*radius+1
static double* KernelBuildFirstDervGuassian(const double& sigma, const int& radius, const int& size) 
{
    // create standard guassian kernel
    double *gKernel = KernelBuildGaussian(sigma, radius, size);

    // create first derivative horizontal to convolve with the image
    double* fdKernel = new double[size*size];

    // initialize kernel
    for(int i = 0; i < size*size; i++)
    {
        fdKernel[i] = 0.0;
    }

    // compute horizontal first derivative kernel weights
    for(int x=-radius; x<=radius; x++)
    {
        // first derivate equation
        // ( x / sigma^2 ) * ( e ^ (-x^2 / 2*sigma^2) )
        double value = (-x / (std::pow(sigma,2))) * (std::exp(-std::pow((double)x,2) / (2*std::pow(sigma,2))));
        fdKernel[(size*radius)+x+radius] = value;
    }

    // convolve kernels
    double* finalKernel = KernelConvolve(gKernel, size, fdKernel, size);

    // normalize kernel
    KernelNormalize(finalKernel, size);
    
    // cleanup
    delete [] gKernel;
    delete [] fdKernel;

    //KernelPrint(finalKernel, size);

    return finalKernel;
}

// Build gaussian second derivative kernel approximation (convolves guassian with 2nd derivative approx.)
// size must be 2*radius+1
static double* KernelBuildSecDervGaussianApprox(const double& sigma, const int& radius, const int& size) 
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

    // fixed 2nd derivate guassian approximation
    sdKernel[7] = 1;
    sdKernel[11] = 1;
    sdKernel[12] = -4;
    sdKernel[13] = 1;
    sdKernel[17] = 1;

    // convolve kernels
    double* finalKernel = KernelConvolve(gKernel, size, sdKernel, sdKernelSize);

    // normalize kernel
    KernelNormalize(finalKernel, size);

    //KernelPrint(finalKernel, size);

    // clean up
    delete [] gKernel;
    delete [] sdKernel;

    return finalKernel;
}

// Build guassian second derivative kernel
// size must be 2*radius+1
static double* KernelBuildSecDervGaussian(const double& sigma, const int& radius, const int& size) 
{
    // create kernel
    double *kernel = new double[size*size];

    // compute kernel weights
    for(int x=-radius; x<=radius; x++)
    {
        for(int y=-radius; y<=radius; y++)
        {
            double value = ( (std::pow((double)x,2) + std::pow((double)y,2) - (2*std::pow(sigma,2))) / (std::pow(sigma,4)) ) *
                
                std::exp( -((std::pow((double)x,2) + std::pow((double)y,2)) / (2*std::pow(sigma,2))) );

            kernel[(x+radius)*size+(y+radius)] = value;
        }
    }

    return kernel;
}

// Build sobel magnitude kernel
// Size is ignored. 3x3 returned.
static double* KernelBuildSobelMagnitude() 
{
    // create kernel
    double *kernel = new double[3*3];

    // set kernel values
    kernel[0] = -1.0;
    kernel[1] = 0.0;
    kernel[2] = 1.0;
    kernel[3] = -2.0;
    kernel[4] = 0.0;
    kernel[5] = 2.0;
    kernel[6] = -1.0;
    kernel[7] = 0.0;
    kernel[8] = 1.0;

    KernelNormalize(kernel,3);

    return kernel;
}

// Build sobel orientation kernel
// Size is ignored. 3x3 returned.
static double* KernelBuildSobelOrientation()
{
    double *kernel = new double[3*3];

    kernel[0] = -1.0;
    kernel[1] = -2.0;
    kernel[2] = -1.0;
    kernel[3] = 0.0;
    kernel[4] = 0.0;
    kernel[5] = 0.0;
    kernel[6] = 1.0;
    kernel[7] = 2.0;
    kernel[8] = 1.0;

    KernelNormalize(kernel,3);

    return kernel;
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

    // create the first derivative guassian kernel
    double* kernel = KernelBuildFirstDervGuassian(sigma, radius, size);

    // create pixel buffer
    Pixel* buffer = ImageCreateBuffer(image, radius);

    // apply first derivative kernel in horizonal direction
    BufferApplyKernel(buffer, image->width(), image->height(), radius, kernel, size, size);

    // add image offset for negative values
    BufferApplyOffset(buffer, image->width(), image->height(), radius, qRgb(128, 128, 128));

    // store buffer to image
    ImageConvertBuffer(image, buffer, radius);
   
    // clean up
    delete [] buffer;
    delete [] kernel;
}

// SecondDerivImage
// Compute guassian second derivative on the image. The second derivative kernel is computed
// in both directions and blurred with the guassian.
void MainWindow::SecondDerivImage(QImage *image, double sigma)
{
    // kernel radius and size
    int radius = 3*sigma;
    int size = 2*radius + 1;

    // create guassian second derivative kernel
    double *kernel = KernelBuildSecDervGaussianApprox(sigma, radius, size);
    
    // create pixel buffer
    Pixel* buffer = ImageCreateBuffer(image, radius);

    // apply second derivative kernel
    BufferApplyKernel(buffer, image->width(), image->height(), radius, kernel, size, size);

    // add image offset for negative values
    BufferApplyOffset(buffer, image->width(), image->height(), radius, qRgb(128, 128, 128));

    // store buffer to image
    ImageConvertBuffer(image, buffer, radius);
    //ImageConvertKernel(image, kernel, size);

    // clean up
    delete [] buffer;
    delete [] kernel;
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
    double *kernel = KernelBuildSecDervGaussianApprox(sigma, radius, size);

    // apply second derivative kernel
    BufferApplyKernel(sdBuffer, image->width(), image->height(), radius, kernel, size, size);

    // create new pixel buffer
    Pixel* buffer = ImageCreateBuffer(image, radius);

    // subtract second derivative from buffer
    BufferSubtractBuffer(buffer, image->width(), image->height(), radius, sdBuffer, alpha);

    // store buffer to image
    ImageConvertBuffer(image, buffer, radius);

    delete [] kernel;
    delete [] sdBuffer;
    delete [] buffer;
}

// Bilateral filter an image using spatial sigma and intensity sigma.
void MainWindow::BilateralImage(QImage *image, double sigmaS, double sigmaI)
{
    int radius = 3*sigmaS;

    // create pixel buffer copy
    Pixel* copyBuffer = ImageCreateBuffer(image, radius);
    Pixel* outBuffer = ImageCreateBuffer(image, radius);

    // compute each pixel
    for(int r = 0; r <image->height(); r++)
    {
        for(int c = 0; c <image->width(); c++)
        {          
            double norm = 0.0;
            double rgb[3] = {0.0, 0.0, 0.0};

            // find pixel
            Pixel* orgPixel = BufferGetPixel(copyBuffer, image->width(), radius, r+radius, c+radius);

            // convolve around the pixel
            for(int rd=-radius;rd<=radius;rd++)
            {
                for(int cd=-radius;cd<=radius;cd++)
                {
                    // find convolution pixel
                    Pixel* convPixel = BufferGetPixel(copyBuffer, image->width(), radius, r+rd+radius, c+cd+radius);

                    // compute range filter (intensity)
                    // pixel intensity calculation may not be correct
                    double range = std::exp( -std::pow( PixelMagnitude(orgPixel)-PixelMagnitude(convPixel), 2) / (2*std::pow(sigmaI,2)) );

                    // domain filter (guassian)
                    double domain = std::exp( -(std::pow((double)rd,2) + std::pow((double)cd,2)) / (2*std::pow(sigmaS,2)) );

                    // compute weight, norm, rgb values
                    double weight = range*domain;
                    norm += weight;

                    rgb[0] += convPixel->r*weight;
                    rgb[1] += convPixel->g*weight;
                    rgb[2] += convPixel->b*weight;
                }
            }

            // find output pixel
            Pixel *outPixel = BufferGetPixel(outBuffer, image->width(), radius, r+radius, c+radius);
            outPixel->r = rgb[0] / norm;
            outPixel->g = rgb[1] / norm;
            outPixel->b = rgb[2] / norm;
        }
    }

    // set output image
    ImageConvertBuffer(image, outBuffer, radius);

    delete [] copyBuffer;
    delete [] outBuffer;
}

// Run the sobel operator on the image. Detects the rate of change in an image
// and displays the magnitude and orientation. The sobel operator is effectively a
// simple edge detector.
//
// When displaying the orientation image I recommend the following:
//
// double mag;   // magnitude of the gradient
// double orien; // orientation of the gradient
//
// double red = (sin(orien) + 1.0)/2.0;
// double green = (cos(orien) + 1.0)/2.0;
// double blue = 1.0 - red - green;
//
// red *= mag*4.0;
// green *= mag*4.0;
// blue *= mag*4.0;
//
// Make sure the pixel values range from 0 to 255
// red = min(255.0, max(0.0, red));
// green = min(255.0, max(0.0, green));
// blue = min(255.0, max(0.0, blue));
//
// image->setPixel(c, r, qRgb( (int) (red), (int) (green), (int) (blue)));
//
void MainWindow::SobelImage(QImage *image)
{
    int radius = 1;
    int size = 2*radius + 1;

    // create buffers
    Pixel *magBuffer = ImageCreateBuffer(image, radius);
    Pixel *orienBuffer = ImageCreateBuffer(image, radius);

    // create kernels
    double *magKernel = KernelBuildSobelMagnitude();
    double *orienKernel = KernelBuildSobelOrientation();

    // apply kernels
    BufferApplyKernel(magBuffer, image->width(), image->height(), radius, magKernel, size, size);
    BufferApplyKernel(orienBuffer, image->width(), image->height(), radius, orienKernel, size, size);

    // create output buffer
    Pixel *outBuffer = ImageCreateBuffer(image, radius);

    // compute pixel values based on buffers
    for(int r = 0; r <image->height(); r++)
    {
        for(int c = 0; c <image->width(); c++)
        {
            Pixel *xp = BufferGetPixel(magBuffer, image->width(), radius, r+radius, c+radius);
            Pixel *yp = BufferGetPixel(orienBuffer, image->width(), radius, r+radius, c+radius);
            
            double xpm = PixelMagnitude(xp);
            double ypm = PixelMagnitude(yp);

            double magnitude = std::sqrt(std::pow(xpm,2) + std::pow(ypm,2));
            double orientation = std::atan(ypm / xpm);

            Pixel *outPixel = BufferGetPixel(outBuffer, image->width(), radius, r+radius, c+radius);
            outPixel->r = (std::sin(orientation) + 1.0) / 2.0;
            outPixel->g = (std::cos(orientation) + 1.0) / 2.0;
            outPixel->b = 1.0 - outPixel->r - outPixel->g;
            
            /*outPixel->r *= 100.0;
            outPixel->g *= 100.0;
            outPixel->b *= 100.0;*/

            outPixel->r *= magnitude*4.0;
            outPixel->g *= magnitude*4.0;
            outPixel->b *= magnitude*4.0;
        }
    }

    // update image
    ImageConvertBuffer(image, outBuffer, radius);

    delete [] magBuffer;
    delete [] orienBuffer;
    delete [] magKernel;
    delete [] orienKernel;
    delete [] outBuffer;
}

// Bilinear interpolation
// Computes the value of a pixel at a fractional position. Interpolates the pixel values
// based on the surrounding pixels.
// Return the RGB values for the pixel at location (x,y) in double rgb[3].
void MainWindow::BilinearInterpolation(Pixel *buffer, const int& width, const int& height, const int& padding,
    const double& x, const double& y, Pixel* p)
{
    // initialize rgb to black
    p->r = 0;
    p->g = 0;
    p->b = 0;

    // compute buffer x,y location
    double bufX = (x+padding);
    double bufY = (y+padding);
   
    // compute base pixel
    int baseX = (int)bufX;
    int baseY = (int)bufY;

    // check if pixels in range (including padding)
    if( bufX >= 0 && (bufX+1) < (width+(padding*2)) && bufY >= 0 && (bufY+1) < (height+(padding*2)) )
    {
        // compute weight values
        double a = bufX-baseX;
        double b = bufY-baseY;

        // find pixels
        Pixel* pixelXY = BufferGetPixel(buffer, width, padding, bufY, bufX);
        Pixel* pixelX1Y = BufferGetPixel(buffer, width, padding, bufY, bufX+1);
        Pixel* pixelXY1 = BufferGetPixel(buffer, width, padding, bufY+1, bufX);
        Pixel* pixelX1Y1 = BufferGetPixel(buffer, width, padding, bufY+1, bufX+1);

        // compute interpolated pixel
        // f (x + a, y + b) = (1 - a)(1 - b) f (x, y) + a(1 - b) f (x + 1, y) + (1 - a)b f (x,y + 1) + ab f (x + 1, y + 1)
        p->r = ((1-a)*(1-b)*pixelXY->r) + (a*(1-b)*pixelX1Y->r) + ((1-a)*b*pixelXY1->r) + (a*b*pixelX1Y1->r);
        p->g = ((1-a)*(1-b)*pixelXY->g) + (a*(1-b)*pixelX1Y->g) + ((1-a)*b*pixelXY1->g) + (a*b*pixelX1Y1->g);
        p->b = ((1-a)*(1-b)*pixelXY->b) + (a*(1-b)*pixelX1Y->b) + ((1-a)*b*pixelXY1->b) + (a*b*pixelX1Y1->b);

        // cap rgb values
        p->r = std::max(p->r,0.0);
        p->r = std::min(p->r,255.0);
        p->g = std::max(p->g,0.0);
        p->g = std::min(p->g,255.0);
        p->b = std::max(p->b,0.0);
        p->b = std::min(p->b,255.0);
    }
}

// Rotates the image by the specified angle. The orientation is specified in degrees.
void MainWindow::RotateImage(QImage *image, double orien)
{
    int r, c;
    QRgb pixel;
    int w = image->width();
    int h = image->height();
    int padding = 1;
    double radians = -2.0*3.141*orien/360.0;

    // create copy buffer
    Pixel* buffer = ImageCreateBuffer(image, padding, false);
    
    // reset image
    pixel = qRgb(0, 0, 0);
    image->fill(pixel);

    // rotate each pixel
    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            Pixel p;
            double x0, y0;
            double x1, y1;

            // rotate around the center of the image.
            x0 = (double) (c - w/2);
            y0 = (double) (r - h/2);

            // rotate using rotation matrix
            x1 = x0*cos(radians) - y0*sin(radians);
            y1 = x0*sin(radians) + y0*cos(radians);

            x1 += (double) (w/2);
            y1 += (double) (h/2);

            // interpolate
            BilinearInterpolation(buffer, w, h, padding, x1, y1, &p);

            // set pixel
            image->setPixel(c, r, qRgb((int) floor(p.r + 0.5), (int) floor(p.g + 0.5), (int) floor(p.b + 0.5)));
        }
    }

    // cleanup
    delete [] buffer;
}

// Find peaks image
// A peak response is found by comparing a pixel’s edge magnitude to two samples perpendicular to an edge at a distance 
// of one pixel (slide 11 in EdgeDetection), call these two samples e0 and e1.  Compute e0 and e1 using BilinearInterpolation.  
// A pixel is a peak response if it is larger than the threshold (“thres”), e0, and e1
void MainWindow::FindPeaksImage(QImage *image, double thres)
{
    // fixed radius
    int radius = 1;
    int size = 2*radius + 1;

    // create buffers
    Pixel *magBuffer = ImageCreateBuffer(image, radius);
    Pixel *orienBuffer = ImageCreateBuffer(image, radius);

    // create kernels
    double *magKernel = KernelBuildSobelMagnitude();
    double *orienKernel = KernelBuildSobelOrientation();

    // apply kernels
    BufferApplyKernel(magBuffer, image->width(), image->height(), radius, magKernel, size, size);
    BufferApplyKernel(orienBuffer, image->width(), image->height(), radius, orienKernel, size, size);

    // reset image
    QRgb clearPixel = qRgb(0, 0, 0);
    image->fill(clearPixel);

    // check each pixel
    for(int r = 0; r < image->height(); r++)
    {
        for(int c = 0; c < image->width(); c++)
        {
            // get magnitidue and orientation
            Pixel *xp = BufferGetPixel(magBuffer, image->width(), radius, r+radius, c+radius);
            Pixel *yp = BufferGetPixel(orienBuffer, image->width(), radius, r+radius, c+radius);
            
            double xpm = PixelMagnitude(xp);
            double ypm = PixelMagnitude(yp);

            double magnitude = std::sqrt(std::pow(xpm,2) + std::pow(ypm,2));
            double orientation = std::atan(ypm / xpm);

            // compute perpendicular pixels
            // distance is the number of pixels from the original pixel
            // x`=-sin(), y`=cos()
            double distance = 1.0;
            double e0x = c + (-distance*std::sin(orientation+(M_PI/2)));
            double e0y = r + (distance*std::cos(orientation+(M_PI/2)));

            double e1x = c + (-distance*std::sin(orientation-(M_PI/2)));
            double e1y = r + (distance*std::cos(orientation-(M_PI/2)));
                
            // bilinear interpolate calculated pixels
            Pixel e0xp;
            Pixel e0yp;
            Pixel e1xp;
            Pixel e1yp;
            BilinearInterpolation(magBuffer, image->width(), image->height(), radius, e0x, e0y, &e0xp);
            BilinearInterpolation(orienBuffer, image->width(), image->height(), radius, e0x, e0y, &e0yp);
            BilinearInterpolation(magBuffer, image->width(), image->height(), radius, e1x, e1y, &e1xp);
            BilinearInterpolation(orienBuffer, image->width(), image->height(), radius, e1x, e1y, &e1yp);

            // compute magnitudes
            double e0xm = PixelMagnitude(&e0xp);
            double e0ym = PixelMagnitude(&e0yp);
            double e1xm = PixelMagnitude(&e1xp);
            double e1ym = PixelMagnitude(&e1yp);

            double e0mag = std::sqrt(std::pow(e0xm,2) + std::pow(e0ym,2));
            double e1mag = std::sqrt(std::pow(e1xm,2) + std::pow(e1ym,2));

            // set output pixel if magnitude is greater than thresholds
            if( magnitude >= e0mag && magnitude >= e1mag && magnitude >= thres )
            {
               QRgb outPixel = qRgb(255,255,255);
               image->setPixel(c, r, outPixel);

               // testing: display perpendicular pixels
               /*QRgb outPerpPixel = qRgb(128, 128, 128);
               image->setPixel(e0x, e0y, outPerpPixel);
               image->setPixel(e1x, e1y, outPerpPixel);*/
            } 
        }
    }
}

void MainWindow::MedianImage(QImage *image, int radius)
{
    // Add your code here
}

void MainWindow::HoughImage(QImage *image)
{
    // Add your code here
}

// Crazy filter
// Modification of the bilaterial blurring filter
void MainWindow::CrazyImage(QImage *image)
{
    double sigmaS = 1.0;
    double sigmaI = 20.0;
    int radius = 3*sigmaS;

    // create pixel buffer copy
    Pixel* copyBuffer = ImageCreateBuffer(image, radius);
    Pixel* outBuffer = ImageCreateBuffer(image, radius);

    // compute each pixel
    for(int r = 0; r <image->height(); r++)
    {
        for(int c = 0; c <image->width(); c++)
        {          
            double norm = 0.0;
            double rgb[3] = {0.0, 0.0, 0.0};

            // find pixel
            Pixel* orgPixel = BufferGetPixel(copyBuffer, image->width(), radius, r+radius, c+radius);

            // convolve around the pixel
            for(int rd=-radius;rd<=radius;rd++)
            {
                for(int cd=-radius;cd<=radius;cd++)
                {
                    // find convolution pixel
                    Pixel* convPixel = BufferGetPixel(copyBuffer, image->width(), radius, r+rd+radius, c+cd+radius);

                    // compute range filter (intensity)
                    // pixel intensity calculation may not be correct
                    double range = std::exp( std::pow( PixelMagnitude(orgPixel)-PixelMagnitude(convPixel), 2) / (2*std::pow(sigmaI,2)) );

                    // domain filter (guassian)
                    double domain = std::exp( -(std::pow((double)rd,2) + std::pow((double)cd,2)) / (2*std::pow(sigmaS,2)) );

                    // compute weight, norm, rgb values
                    double weight = range*domain;
                    norm += weight;

                    rgb[0] += convPixel->r*weight;
                    rgb[1] += convPixel->g*weight;
                    rgb[2] += convPixel->b*weight;
                }
            }

            // find output pixel
            Pixel *outPixel = BufferGetPixel(outBuffer, image->width(), radius, r+radius, c+radius);
            outPixel->r = rgb[0] / norm;
            outPixel->g = rgb[1] / norm;
            outPixel->b = rgb[2] / norm;
        }
    }

    // set output image
    ImageConvertBuffer(image, outBuffer, radius);

    delete [] copyBuffer;
    delete [] outBuffer;
}
