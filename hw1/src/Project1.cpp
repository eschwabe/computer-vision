#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <cmath>
#include <iostream>

/***********************************************************************
  This is the only file you need to change for your assignment.  The
  other files control the UI (in case you want to make changes.)
************************************************************************/

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

// ApplyKernel
// Applies a kernel filter to an image. The kernel can be any height and width in size.
static void ApplyKernel(QImage *image, const double* kernel, const int kh, const int kw)
{
    int r, rd;      // image and kernel row
    int c, cd;      // image and kernel col
    QRgb pixel;     // temporary pixel

    // create a buffer image so we're not reading and writing to the same image during filtering.
    QImage buffer;
    int width = image->width();
    int height = image->height();

    // compute horizontal and vertical kernel radius
    int khr = kh/2;
    int kwr = kw/2;

    // create an image of size (w + kernel width, h + kernel height) with black borders.
    // could be improved by filling the pixels using a different padding technique (reflected, fixed, etc.)
    buffer = image->copy(-kwr, -khr, width + kw, height + kh);

    // for each pixel in the image
    for(r=0;r<height;r++)
    {
        for(c=0;c<width;c++)
        {
            double rgb[3];

            rgb[0] = 0.0;
            rgb[1] = 0.0;
            rgb[2] = 0.0;

            // convolve the kernel at each pixel
            for(rd=-khr;rd<=khr;rd++)
            {
                for(cd=-kwr;cd<=kwr;cd++)
                {
                     // get the pixel value
                     pixel = buffer.pixel(c + cd + kwr, r + rd + khr);

                     // get the value of the kernel
                     double weight = kernel[(rd + khr)*kw + cd + kwr];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }
            }

            // store mean pixel in the image to be returned.
            image->setPixel(c, r, qRgb((int) floor(rgb[0] + 0.5), (int) floor(rgb[1] + 0.5), (int) floor(rgb[2] + 0.5)));
        }
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
    double *kernel = new double [size*size];

    // compute kernel weights and z normalization
    double znorm = 0.0;
    for(x=-radius; x<=radius; x++)
    {
        for(y=-radius; y<=radius; y++)
        {
            double value = std::exp( -(std::pow((double)x,2) + std::pow((double)y,2)) / (2*std::pow(sigma,2)) );
            kernel[(x+radius)*size+(y+radius)] = value;
            znorm += value;
        }
    }

    // normalize kernel
    for(x=-radius; x<=radius; x++)
    {
        for(y=-radius; y<=radius; y++)
        {
            kernel[(x+radius)*size+(y+radius)] /= znorm;
        }
    }

    // apply kernel
    ApplyKernel(image, kernel, size, size);

    // clean up
    delete [] kernel;
}

// Seperable Gaussian Blur Image
// Seperate Gaussian algorithm into horizonal and vertical components. Apply each component
// to the image to create the same effect.
void MainWindow::SeparableGaussianBlurImage(QImage *image, double sigma)
{
    int x, y;

    // kernel radius and size
    int radius = 3*sigma;
    int size = 2*radius + 1;

    // create horizontal and vertical kernels to convolve with the image
    double *x_kernel = new double [size];
    double *y_kernel = new double [size];

    // compute horizontal kernel weights and z normalization
    double znorm = 0.0;
    for(x=-radius; x<=radius; x++)
    {
        double value = std::exp( -std::pow((double)x,2) / (2*std::pow(sigma,2)) );
        x_kernel[x+radius] = value;
        znorm += value;
    }

    // normalize horizontal kernel
    for(x=-radius; x<=radius; x++)
    {
        x_kernel[x+radius] /= znorm;
    }

    // compute vertical kernel weights and z normalization
    znorm = 0.0;
    for(y=-radius; y<=radius; y++)
    {
        double value = std::exp( -std::pow((double)y,2) / (2*std::pow(sigma,2)) );
        y_kernel[y+radius] = value;
        znorm += value;
    }

    // normalize vertical kernel
    for(y=-radius; y<=radius; y++)
    {
        y_kernel[y+radius] /= znorm;
    }

    // apply horizonal and then vertical kernels
    ApplyKernel(image, x_kernel, 1, size);
    ApplyKernel(image, y_kernel, size, 1);

    // clean up
    delete [] x_kernel;
    delete [] y_kernel;
}

void MainWindow::FirstDerivImage(QImage *image, double sigma)
{
    // Add your code here.
}

void MainWindow::SecondDerivImage(QImage *image, double sigma)
{
    // Add your code here.
}

void MainWindow::SharpenImage(QImage *image, double sigma, double alpha)
{
    // Add your code here.  It's probably easiest to call SecondDerivImage as a helper function.
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
