#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <cmath>

/*******************************************************************************
The following are helper routines with code already written.
The routines you'll need to write for the assignment are below.
*******************************************************************************/

/*******************************************************************************
Open the training dataset

posdirectory - Directory containing face images
negdirectory - Directory containing non-face images
trainingData - Array used to store the data
trainingLabel - Label assigned to training data (1 = face, 0 = non-face)
numTrainingExamples - Number of training examples
patchSize - Size of training patches
*******************************************************************************/
void MainWindow::OpenDataSet(QDir posdirectory, QDir negdirectory, double *trainingData, int *trainingLabel, int numTrainingExamples, int patchSize)
{
    int i, c, r;
    QStringList imgNames;
    QImage inImage;
    QRgb pixel;

    imgNames = posdirectory.entryList();

    int idx = 0;

    for(i=0;i<imgNames.length();i++)
        if(idx < numTrainingExamples/2)
        {
            inImage.load(posdirectory.absolutePath() + QDir::separator() + imgNames.at(i));

            if(!(inImage.isNull()))
            {
                for(r=0;r<patchSize;r++)
                    for(c=0;c<patchSize;c++)
                    {
                        pixel = inImage.pixel(c, r);
                        trainingData[idx*patchSize*patchSize + r*patchSize + c] = (double) qGreen(pixel);
                    }

                    trainingLabel[idx] = 1;

                    idx++;
            }
        }

        imgNames = negdirectory.entryList();

        for(i=0;i<imgNames.length();i++)
            if(idx < numTrainingExamples)
            {
                inImage.load(negdirectory.absolutePath() + QDir::separator() + imgNames.at(i));

                if(!(inImage.isNull()))
                {
                    for(r=0;r<patchSize;r++)
                        for(c=0;c<patchSize;c++)
                        {
                            pixel = inImage.pixel(c, r);
                            trainingData[idx*patchSize*patchSize + r*patchSize + c] = (double) qGreen(pixel);
                        }

                        trainingLabel[idx] = 0;

                        idx++;
                }
            }
}

/*******************************************************************************
DisplayTrainingDataset - Display example patches from training dataset

displayImage - Display image
trainingData - Array used to store the data
trainingLabel - Label assigned to training data (1 = face, 0 = non-face)
numTrainingExamples - Number of training examples
patchSize - Size of training patches
*******************************************************************************/
void MainWindow::DisplayTrainingDataset(QImage *displayImage, double *trainingData, int *trainingLabel, int numTrainingExamples, int patchSize)
{
    int w = displayImage->width();
    int h = displayImage->height();
    int r, c;
    int rOffset = 0;
    int cOffset = 0;
    bool inBounds = true;
    int ct = 0;

    while(inBounds)
    {
        int idx = rand()%numTrainingExamples;

        for(r=0;r<patchSize;r++)
            for(c=0;c<patchSize;c++)
            {
                if(trainingLabel[idx] == 1)
                {
                    int val = (int) trainingData[idx*patchSize*patchSize + r*patchSize + c];
                    displayImage->setPixel(c + cOffset, r + rOffset, qRgb(val, val, val));

                }
                else
                {
                    int val = (int) trainingData[idx*patchSize*patchSize + r*patchSize + c];
                    displayImage->setPixel(c + cOffset, r + rOffset, qRgb(val, val, val));
                }
            }

            cOffset += patchSize;

            if(cOffset + patchSize >= w)
            {
                cOffset = 0;
                rOffset += patchSize;

                if(rOffset + patchSize >= h)
                    inBounds = false;
            }

            ct++;
    }
}

/*******************************************************************************
SaveClassifier - Save the computed AdaBoost classifier

fileName - Name of file
*******************************************************************************/
void MainWindow::SaveClassifier(QString fileName)
{
    int i, j;
    FILE *out;

    out = fopen(fileName.toAscii(), "w");

    fprintf(out, "%d\n", m_NumWeakClassifiers);

    for(i=0;i<m_NumWeakClassifiers;i++)
    {
        fprintf(out, "%d\n", m_WeakClassifiers[i].m_NumBoxes);

        for(j=0;j<m_WeakClassifiers[i].m_NumBoxes;j++)
            fprintf(out, "%lf\t%lf\t%lf\t%lf\t%lf\n", m_WeakClassifiers[i].m_BoxSign[j], m_WeakClassifiers[i].m_Box[j][0][0], m_WeakClassifiers[i].m_Box[j][0][1],
            m_WeakClassifiers[i].m_Box[j][1][0], m_WeakClassifiers[i].m_Box[j][1][1]);

        fprintf(out, "%lf\n", m_WeakClassifiers[i].m_Area);
        fprintf(out, "%lf\n", m_WeakClassifiers[i].m_Polarity);
        fprintf(out, "%lf\n", m_WeakClassifiers[i].m_Threshold);
        fprintf(out, "%lf\n", m_WeakClassifiers[i].m_Weight);
    }

    fclose(out);
}

/*******************************************************************************
OpenClassifier - Open the computed AdaBoost classifier

fileName - Name of file
*******************************************************************************/
void MainWindow::OpenClassifier(QString fileName)
{
    int i, j;
    FILE *in;

    in = fopen(fileName.toAscii(), "r");

    fscanf(in, "%d\n", &m_NumWeakClassifiers);
    m_WeakClassifiers = new CWeakClassifiers [m_NumWeakClassifiers];

    for(i=0;i<m_NumWeakClassifiers;i++)
    {
        fscanf(in, "%d\n", &(m_WeakClassifiers[i].m_NumBoxes));
        m_WeakClassifiers[i].m_Box = new double [m_WeakClassifiers[i].m_NumBoxes][2][2];
        m_WeakClassifiers[i].m_BoxSign = new double [m_WeakClassifiers[i].m_NumBoxes];

        for(j=0;j<m_WeakClassifiers[i].m_NumBoxes;j++)
            fscanf(in, "%lf\t%lf\t%lf\t%lf\t%lf\n", &(m_WeakClassifiers[i].m_BoxSign[j]), &(m_WeakClassifiers[i].m_Box[j][0][0]), &(m_WeakClassifiers[i].m_Box[j][0][1]),
            &(m_WeakClassifiers[i].m_Box[j][1][0]), &(m_WeakClassifiers[i].m_Box[j][1][1]));

        fscanf(in, "%lf\n", &(m_WeakClassifiers[i].m_Area));
        fscanf(in, "%lf\n", &(m_WeakClassifiers[i].m_Polarity));
        fscanf(in, "%lf\n", &(m_WeakClassifiers[i].m_Threshold));
        fscanf(in, "%lf\n", &(m_WeakClassifiers[i].m_Weight));
    }

    fclose(in);

}

/*******************************************************************************
DisplayClassifiers - Display the Haar wavelets for the classifier

displayImage - Display image
weakClassifiers - The weak classifiers used in AdaBoost
numWeakClassifiers - Number of weak classifiers
*******************************************************************************/
void MainWindow::DisplayClassifiers(QImage *displayImage, CWeakClassifiers *weakClassifiers, int numWeakClassifiers)
{
    int w = displayImage->width();
    int h = displayImage->height();
    int i, j, r, c;
    int rOffset = 0;
    int cOffset = 0;
    int size = 50;
    bool inBounds = true;

    displayImage->fill(qRgb(0,0,0));

    for(i=0;(i<numWeakClassifiers) && inBounds;i++)
    {
        for(r=0;r<size;r++)
            for(c=0;c<size;c++)
            {
                displayImage->setPixel(c + cOffset, r + rOffset, qRgb(128, 128, 128));
            }

            for(j=0;j<weakClassifiers[i].m_NumBoxes;j++)
                for(r=(int) ((double) size*weakClassifiers[i].m_Box[j][0][1]);r<(int) ((double) size*weakClassifiers[i].m_Box[j][1][1]);r++)
                    for(c=(int) ((double) size*weakClassifiers[i].m_Box[j][0][0]);c<(int) ((double) size*weakClassifiers[i].m_Box[j][1][0]);c++)
                    {
                        if(weakClassifiers[i].m_BoxSign[j] > 0.0)
                            displayImage->setPixel(c + cOffset, r + rOffset, qRgb(255, 255, 255));
                        else
                            displayImage->setPixel(c + cOffset, r + rOffset, qRgb(0, 0, 0));
                    }

                    cOffset += size+1;

                    if(cOffset + size >= w)
                    {
                        cOffset = 0;
                        rOffset += size + 1;

                        if(rOffset + size >= h)
                            inBounds = false;
                    }
    }
}

/*******************************************************************************
DisplayIntegralImage - Display the integral image

displayImage - Display image
integralImage - Output integral image
w, h - Width and height of image
*******************************************************************************/
void MainWindow::DisplayIntegralImage(QImage *displayImage, double *integralImage, int w, int h)
{
    int r, c;
    double maxVal = integralImage[(h-1)*w + w-1];

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            int val = (int) (255.0*integralImage[r*w + c]/maxVal);

            displayImage->setPixel(c, r, qRgb(val, val, val));
        }
}

/*******************************************************************************
InitializeFeatures - Randomly initialize the candidate weak classifiers

weakClassifiers - Candidate weak classifiers
numWeakClassifiers - Number of candidate weak classifiers
*******************************************************************************/
void MainWindow::InitializeFeatures(CWeakClassifiers *weakClassifiers, int numWeakClassifiers)
{
    int i;

    for(i=0;i<numWeakClassifiers;i++)
    {
        double x, y, w, h;

        // We don't know these values yet, so just initialize to 0
        weakClassifiers[i].m_Polarity = 0.0;
        weakClassifiers[i].m_Threshold = 0.0;
        weakClassifiers[i].m_Weight = 0.0;

        // The Haar wavelet's corners can range in the area of 0.02 to 0.98, with a minimum size of 0.25
        // We limit the range to [0.2, 0.98], instead of [0, 1] so we don't need to worry about checking
        // out of bounds errors later on, i.e. in the BilinearInterpolation function.

        // x position of box and width
        w = 0.25 + 0.71*(double) rand()/(double) RAND_MAX;
        x = 0.02 + (0.96 - w)*(double) rand()/(double) RAND_MAX;

        // y position of box and height
        h = 0.25 + 0.71*(double) rand()/(double) RAND_MAX;
        y = 0.02 + (0.96 - h)*(double) rand()/(double) RAND_MAX;

        int boxType = rand()%4;

        if(boxType == 0)
        {
            // Vertical boxes
            weakClassifiers[i].m_NumBoxes = 2;
            weakClassifiers[i].m_Box = new double [weakClassifiers[i].m_NumBoxes][2][2];
            weakClassifiers[i].m_BoxSign = new double [weakClassifiers[i].m_NumBoxes];

            weakClassifiers[i].m_BoxSign[0] = 1.0;
            weakClassifiers[i].m_Box[0][0][0] = x;
            weakClassifiers[i].m_Box[0][0][1] = y;
            weakClassifiers[i].m_Box[0][1][0] = x + w/2;
            weakClassifiers[i].m_Box[0][1][1] = y + h;

            weakClassifiers[i].m_BoxSign[1] = -1.0;
            weakClassifiers[i].m_Box[1][0][0] = x + w/2;
            weakClassifiers[i].m_Box[1][0][1] = y;
            weakClassifiers[i].m_Box[1][1][0] = x + w;
            weakClassifiers[i].m_Box[1][1][1] = y + h;
        }

        if(boxType == 1)
        {
            // 2 Horizontal boxes
            weakClassifiers[i].m_NumBoxes = 2;
            weakClassifiers[i].m_Box = new double [weakClassifiers[i].m_NumBoxes][2][2];
            weakClassifiers[i].m_BoxSign = new double [weakClassifiers[i].m_NumBoxes];

            weakClassifiers[i].m_BoxSign[0] = 1.0;
            weakClassifiers[i].m_Box[0][0][0] = x;
            weakClassifiers[i].m_Box[0][0][1] = y;
            weakClassifiers[i].m_Box[0][1][0] = x + w;
            weakClassifiers[i].m_Box[0][1][1] = y + h/2;

            weakClassifiers[i].m_BoxSign[1] = -1.0;
            weakClassifiers[i].m_Box[1][0][0] = x;
            weakClassifiers[i].m_Box[1][0][1] = y + h/2;
            weakClassifiers[i].m_Box[1][1][0] = x + w;
            weakClassifiers[i].m_Box[1][1][1] = y + h;
        }

        if(boxType == 2)
        {
            // 3 Vertical boxes
            weakClassifiers[i].m_NumBoxes = 3;
            weakClassifiers[i].m_Box = new double [weakClassifiers[i].m_NumBoxes][2][2];
            weakClassifiers[i].m_BoxSign = new double [weakClassifiers[i].m_NumBoxes];

            weakClassifiers[i].m_BoxSign[0] = 1.0;
            weakClassifiers[i].m_Box[0][0][0] = x;
            weakClassifiers[i].m_Box[0][0][1] = y;
            weakClassifiers[i].m_Box[0][1][0] = x + w/3;
            weakClassifiers[i].m_Box[0][1][1] = y + h;

            weakClassifiers[i].m_BoxSign[1] = -2.0;
            weakClassifiers[i].m_Box[1][0][0] = x + w/3;
            weakClassifiers[i].m_Box[1][0][1] = y;
            weakClassifiers[i].m_Box[1][1][0] = x + 2*w/3;
            weakClassifiers[i].m_Box[1][1][1] = y + h;

            weakClassifiers[i].m_BoxSign[2] = 1.0;
            weakClassifiers[i].m_Box[2][0][0] = x + 2*w/3;
            weakClassifiers[i].m_Box[2][0][1] = y;
            weakClassifiers[i].m_Box[2][1][0] = x + w;
            weakClassifiers[i].m_Box[2][1][1] = y + h;
        }

        // Eric Schwabe
        // ADD 4 BOX FEATURE 
        if(boxType == 3)
        {
            // 4 box grid
            weakClassifiers[i].m_NumBoxes = 4;
            weakClassifiers[i].m_Box = new double [weakClassifiers[i].m_NumBoxes][2][2];
            weakClassifiers[i].m_BoxSign = new double [weakClassifiers[i].m_NumBoxes];

            weakClassifiers[i].m_BoxSign[0] = 1.0;
            weakClassifiers[i].m_Box[0][0][0] = x;
            weakClassifiers[i].m_Box[0][0][1] = y;
            weakClassifiers[i].m_Box[0][1][0] = x + w/2;
            weakClassifiers[i].m_Box[0][1][1] = y + h/2;

            weakClassifiers[i].m_BoxSign[1] = -1.0;
            weakClassifiers[i].m_Box[1][0][0] = x + w/2;
            weakClassifiers[i].m_Box[1][0][1] = y;
            weakClassifiers[i].m_Box[1][1][0] = x + w;
            weakClassifiers[i].m_Box[1][1][1] = y + h/2;

            weakClassifiers[i].m_BoxSign[2] = -1.0;
            weakClassifiers[i].m_Box[2][0][0] = x;
            weakClassifiers[i].m_Box[2][0][1] = y + h/2;
            weakClassifiers[i].m_Box[2][1][0] = x + w/2;
            weakClassifiers[i].m_Box[2][1][1] = y + h;

            weakClassifiers[i].m_BoxSign[3] = 1.0;
            weakClassifiers[i].m_Box[3][0][0] = x + w/2;
            weakClassifiers[i].m_Box[3][0][1] = y + h/2;
            weakClassifiers[i].m_Box[3][1][0] = x + w;
            weakClassifiers[i].m_Box[3][1][1] = y + h;
        }

        weakClassifiers[i].m_Area = w*h;
    }
}

/*******************************************************************************
ConvertColorToDouble - Simple helper function to convert from RGB to double

image - Input image
dImage - Output double image
w, h - Image width and height
*******************************************************************************/
void MainWindow::ConvertColorToDouble(QImage image, double *dImage, int w, int h)
{
    QRgb pixel;
    int r, c;

    for(r=0;r<h;r++)
        for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            dImage[r*w + c] = qGreen(pixel);
        }
}

/*******************************************************************************
ComputeTrainingSetFeatures - Compute all of the features for the training dataset

trainingData - Array used to store the data
features - Array holding feature values
numTrainingExamples - Number of training examples
patchSize - Size of training patches
weakClassifiers - Candidate weak classifiers
numWeakClassifiers - Number of candidate weak classifiers
*******************************************************************************/
void MainWindow::ComputeTrainingSetFeatures(double *trainingData, double *features,
    int numTrainingExamples, int patchSize, CWeakClassifiers *weakClassifiers, int numWeakClassifiers)
{
    int i;
    double *integralImage = new double [patchSize*patchSize];

    for(i=0;i<numTrainingExamples;i++)
    {
        // Compute features for training examples

        // First compute the integral image for each patch
        IntegralImage(&(trainingData[i*patchSize*patchSize]), integralImage, patchSize, patchSize);

        // Compute the Haar wavelets
        ComputeFeatures(integralImage, 0, 0, patchSize, &(features[i*numWeakClassifiers]), weakClassifiers, numWeakClassifiers, patchSize);
    }


    // We shouldn't need the training data anymore so let's delete it.
    delete [] trainingData;

    delete [] integralImage;
}

/*******************************************************************************
DisplayFeatures - Display the computed features (green = faces, red = background)

displayImage - Display image
features - Array holding feature values
trainingLabel - Label assigned to training data (1 = face, 0 = non-face)
numFeatures - Number of features
numTrainingExamples - Number of training examples
*******************************************************************************/
void MainWindow::DisplayFeatures(QImage *displayImage, double *features, int *trainingLabel, int numFeatures, int numTrainingExamples)
{
    int r, c;
    int w = displayImage->width();
    int h = displayImage->height();
    int posCt = 0;
    int negCt = 0;

    double mean = 0.0;
    double meanCt = 0.0;

    displayImage->fill(qRgb(230,230,230));

    for(r=0;r<numTrainingExamples;r+=10)
    {
        for(c=0;c<numFeatures;c++)
        {
            mean += fabs(features[r*numFeatures + c]);
            meanCt++;
        }
    }

    mean /= meanCt;

    for(r=0;r<numTrainingExamples;r++)
    {
        if(trainingLabel[r] == 1 && posCt < h/2)
        {
            for(c=0;c<numFeatures;c++)
                if(c < w)
                {
                    int val = 255.0*(features[r*numFeatures + c]/(4.0*mean)) + 128.0;
                    val = min(255, max(0, val));

                    displayImage->setPixel(c, posCt, qRgb(0, val, 0));
                }

                posCt++;
        }

        if(trainingLabel[r] == 0 && negCt < h/2)
        {
            for(c=0;c<numFeatures;c++)
                if(c < w)
                {
                    int val = 255.0*(features[r*numFeatures + c]/(4.0*mean)) + 128.0;
                    val = min(255, max(0, val));

                    displayImage->setPixel(c, negCt + h/2, qRgb(val, 0, 0));
                }

                negCt++;
        }
    }

}

/*******************************************************************************
AdaBoost - Computes and AdaBoost classifier using a set of candidate weak classifiers

features - Array of feature values pre-computed for the training dataset
trainingLabel - Ground truth labels for the training examples (1 = face, 0 = background)
numTrainingExamples - Number of training examples
candidateWeakClassifiers - Set of candidate weak classifiers
numCandidateWeakClassifiers - Number of candidate weak classifiers
weakClassifiers - Set of weak classifiers selected by AdaBoost
numWeakClassifiers - Number of selected weak classifiers
*******************************************************************************/
void MainWindow::AdaBoost(double *features, int *trainingLabel, int numTrainingExamples,
    CWeakClassifiers *candidateWeakClassifiers, int numCandidateWeakClassifiers, CWeakClassifiers *weakClassifiers, int numWeakClassifiers)
{
    FILE *out;
    out = fopen("AdaBoost.txt", "w");
    double *scores = new double [numTrainingExamples];
    double weightSum = 0.0;
    int *featureSortIdx = new int [numTrainingExamples*numCandidateWeakClassifiers];
    double *featureTranspose = new double [numTrainingExamples*numCandidateWeakClassifiers];

    // Record the classification socres for each training example
    memset(scores, 0, numTrainingExamples*sizeof(double));

    int i, j;
    // The weighting for each training example
    double *dataWeights = new double [numTrainingExamples];

    // Begin with uniform weighting
    for(i=0;i<numTrainingExamples;i++)
        dataWeights[i] = 1.0/(double) (numTrainingExamples);


    // Let's sort the feature values for each candidate weak classifier
    for(i=0;i<numCandidateWeakClassifiers;i++)
    {
        QMap<double, int> featureSort;
        QMap<double, int>::const_iterator iterator;


        for(j=0;j<numTrainingExamples;j++)
        {
            featureSort.insertMulti(features[j*numCandidateWeakClassifiers + i], j);

            // For ease later on we'll store a transposed version of the feature array
            featureTranspose[i*numTrainingExamples + j] = features[j*numCandidateWeakClassifiers + i];
        }

        j = 0;
        iterator = featureSort.constBegin();
        // Let's remember the indices of the sorted features for later.
        while (iterator != featureSort.constEnd())
        {
            featureSortIdx[i*numTrainingExamples + j] = iterator.value();
            iterator++;
            j++;
        }
    }

    // We shouldn't need the features anymore so let's delete it.
    delete [] features;


    // Find a set of weak classifiers using AdaBoost
    for(i=0;i<numWeakClassifiers;i++)
    {
        double bestError = 99999.0;
        int bestIdx = 0;

        // For each potential weak classifier
        for(j=0;j<numCandidateWeakClassifiers;j++)
        {
            CWeakClassifiers bestClassifier;

            // Find the best threshold, polarity and weight for the candidate weak classifier
            double error = FindBestClassifier(&(featureSortIdx[j*numTrainingExamples]),
                &(featureTranspose[j*numTrainingExamples]),
                trainingLabel, dataWeights, numTrainingExamples,
                candidateWeakClassifiers[j], &bestClassifier);

            // Is this the best classifier found so far?
            if(error < bestError)
            {
                bestError = error;
                bestIdx = j;

                // Remember the best classifier
                bestClassifier.copy(&(weakClassifiers[i]));
            }
        }

        // Given the best weak classifier found, update the weighting of the training data.
        UpdateDataWeights(&(featureTranspose[bestIdx*numTrainingExamples]), trainingLabel, weakClassifiers[i], dataWeights, numTrainingExamples);

        // Let's compute the current error for the training dataset
        weightSum += weakClassifiers[i].m_Weight;
        double error = 0.0;
        for(j=0;j<numTrainingExamples;j++)
        {
            if(featureTranspose[bestIdx*numTrainingExamples + j] > weakClassifiers[i].m_Threshold)
            {
                scores[j] += weakClassifiers[i].m_Weight*weakClassifiers[i].m_Polarity;
            }
            else
            {
                scores[j] += weakClassifiers[i].m_Weight*(1.0 - weakClassifiers[i].m_Polarity);
            }

            if((scores[j] > 0.5*weightSum && trainingLabel[j] == 0) ||
                (scores[j] < 0.5*weightSum && trainingLabel[j] == 1))
                error++;
        }

        // Output information that you might find useful for debugging
        fprintf(out, "Count: %d\tIdx: %d\tWeight: %lf\tError: %lf\n", i, bestIdx,
            weakClassifiers[i].m_Weight, error/(double) numTrainingExamples);
        fflush(out);
    }

    delete [] dataWeights;
    delete [] scores;
    delete [] featureSortIdx;
    delete [] featureTranspose;

    fclose(out);
}

/*******************************************************************************
FindFaces - Find faces in an image

weakClassifiers - Set of weak classifiers
numWeakClassifiers - Number of weak classifiers
threshold - Classifier must be above Threshold to return detected face.
minScale, maxScale - Minimum and maximum scale to search for faces.
faceDetections - Set of face detections
displayImage - Display image showing detected faces.
*******************************************************************************/
void MainWindow::FindFaces(QImage inImage, CWeakClassifiers *weakClassifiers, int numWeakClassifiers, double threshold, double minScale, double maxScale,
    QMap<double, CDetection> *faceDetections, QImage *displayImage)
{
    int w = inImage.width();
    int h = inImage.height();
    double *integralImage = new double [w*h];
    double *dImage = new double [w*h];
    double scaleMulti = 1.26;
    double scale;
    int r, c;

    ConvertColorToDouble(inImage, dImage, w, h);
    // Compute the integral image
    IntegralImage(dImage, integralImage, w, h);

    // Serach in scale space
    for(scale=minScale;scale<maxScale;scale*=scaleMulti)
    {
        // Find size of bounding box, and the step size between neighboring bounding boxes.
        int faceSize = (int) scale;
        int stepSize = max(2, faceSize/8);

        // For every possible position
        for(r=0;r<h-faceSize;r+=stepSize)
            for(c=0;c<w-faceSize;c+=stepSize)
            {
                // Compute the score of the classifier
                double score = ClassifyBox(integralImage, c, r, faceSize, weakClassifiers, numWeakClassifiers, w);

                // Is the score above threshold?
                if(score > threshold)
                {
                    CDetection detection;
                    detection.m_Score = score;
                    detection.m_Scale = scale;
                    detection.m_X = (double) c;
                    detection.m_Y = (double) r;

                    // Remember the detection
                    faceDetections->insertMulti(score, detection);
                }

            }
    }

    // Draw face bounding boxes
    DrawFace(displayImage, faceDetections);

    delete [] dImage;
    delete [] integralImage;
}

/*******************************************************************************
DrawFace - Draw the detected faces.

displayImage - Display image
faceDetections - Set of face detections
*******************************************************************************/
void MainWindow::DrawFace(QImage *displayImage, QMap<double, CDetection> *faceDetections)
{
    int r, c;
    QMap<double, CDetection>::const_iterator iterator = faceDetections->constBegin();

    while(iterator != faceDetections->constEnd())
    {
        CDetection detection = iterator.value();
        int c0 = (int) detection.m_X;
        int r0 = (int) detection.m_Y;
        int size = (int) detection.m_Scale;

        for(r=r0;r<r0+size;r++)
            displayImage->setPixel(c0, r, qRgb(255, 0, 0));

        for(r=r0;r<r0+size;r++)
            displayImage->setPixel(c0 + size, r, qRgb(255, 0, 0));

        for(c=c0;c<c0+size;c++)
            displayImage->setPixel(c, r0, qRgb(255, 0, 0));

        for(c=c0;c<c0+size;c++)
            displayImage->setPixel(c, r0 + size, qRgb(255, 0, 0));

        iterator++;
    }

}

/*******************************************************************************
METHODS
*******************************************************************************/

/*******************************************************************************
DisplayAverageFace - Display the average face and non-face image

displayImage - Display image, draw the average images on this image
trainingData - Array used to store the data
trainingLabel - Label assigned to training data (1 = face, 0 = non-face)
numTrainingExamples - Number of training examples
patchSize - Size of training patches in one dimension (patches have patchSize*patchSize pixels)
*******************************************************************************/
void MainWindow::DisplayAverageFace(QImage *displayImage, double *trainingData, int *trainingLabel, 
    int numTrainingExamples, int patchSize)
{
    double *averageFace = new double[patchSize*patchSize];
    double *averageNon = new double[patchSize*patchSize];
    int faces = 0;
    int nonfaces = 0;

    // initialize average data
    for(int i=0; i<patchSize*patchSize; i++)
    {
        averageFace[i] = 0.0;
        averageNon[i] = 0.0;
    }

    // for each training image
    for(int idx=0; idx<numTrainingExamples; idx++)
    {
        double* averageData = NULL;

        // check training data type
        if(trainingLabel[idx] == 1)
        {
            // face
            averageData = averageFace;
            faces++;
        } 
        else 
        {
            // non-face
            averageData = averageNon;
            nonfaces++;
        }

        // for each row/col in the image
        for(int r=0;r<patchSize;r++)
        {
            for(int c=0;c<patchSize;c++)
            {
                // add each pixel to average data
                double data = trainingData[idx*patchSize*patchSize + r*patchSize + c];
                averageData[r*patchSize + c] += data;
            }
        }
    }

    // compute final averages
    for(int i=0; i<patchSize*patchSize; i++)
    {
        averageFace[i] /= faces;
        averageNon[i] /= nonfaces;
    }

    // clear image
    displayImage->fill(qRgb(230, 230, 230));

    // store averages in display image
    for(int r=0;r<patchSize;r++)
    {
        for(int c=0;c<patchSize;c++)
        {
            int pFace = (int)averageFace[r*patchSize + c];
            int pNonFace = (int)averageNon[r*patchSize + c];

            displayImage->setPixel(c, r, qRgb(pFace,pFace,pFace));
            displayImage->setPixel(c+patchSize, r, qRgb(pNonFace,pNonFace,pNonFace));
        }
    }
}

/*******************************************************************************
IntegralImage - Compute the integral image

image - Input double image
integralImage - Output integral image
w, h - Width and height of image
*******************************************************************************/
void MainWindow::IntegralImage(double *image, double *integralImage, int w, int h)
{
    // sums of rectangular regions

    // for each pixel
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            // find each pixel in nearby grid
            // if out of range, pixel contributes 0
            double pXY = image[r*w + c];
            double pX1Y = 0.0;
            double pXY1 = 0.0;
            double pX1Y1 = 0.0;

            if(c>0) pX1Y = integralImage[r*w + (c-1)];
            if(r>0) pXY1 = integralImage[(r-1)*w + (c)];
            if(r>0 && c>0) pX1Y1 = integralImage[(r-1)*w + (c-1)];

            // compute sum for current pixel
            integralImage[r*w+c] = pXY + pX1Y + pXY1 - pX1Y1;
        }
    }
}

/*******************************************************************************
Standard bilinear interpolation (helper function for SumBox)

image - image
x, y - Position to interpolate
w - Width of image (integralImage)
*******************************************************************************/
double MainWindow::BilinearInterpolation(double *image, double x, double y, int w)
{
    double value = 0.0;

    // compute base pixel
    int baseX = (int)x;
    int baseY = (int)y;

    // check if pixels in range
    //if( x >= 0 && (x+1) < w && y >= 0 )

    // compute weight values
    double a = x-baseX;
    double b = y-baseY;

    // find pixels
    double pixelXY = image[(baseY)*w+(baseX)];
    double pixelX1Y = image[(baseY)*w+(baseX+1)];
    double pixelXY1 = image[(baseY+1)*w+(baseX)];
    double pixelX1Y1 = image[(baseY+1)*w+(baseX+1)];

    // compute interpolated pixel
    // f (x + a, y + b) = (1 - a)(1 - b) f (x, y) + a(1 - b) f (x + 1, y) + (1 - a)b f (x,y + 1) + ab f (x + 1, y + 1)
    value = ((1.0-a)*(1.0-b)*pixelXY) + (a*(1.0-b)*pixelX1Y) + ((1.0-a)*b*pixelXY1) + (a*b*pixelX1Y1);

    return value;
}

/*******************************************************************************
SumBox - Helper function for ComputeFeatures - compute the sum of the pixels within a box.

integralImage - integral image
x0, y0 - Upper lefthand corner of box
x1, y1 - Lower righthand corner of box
w - Width of image (integralImage)
*******************************************************************************/
double MainWindow::SumBox(double *integralImage, double x0, double y0, double x1, double y1, int w)
{
    double sum = 0.0;

    // x0,y0
    //      A ----- B
    //      |       |
    //      |       |
    //      C ----- D
    //              x1,y1

    // compute A (bottom left), B (bottom right), C (top left), D (top right) sum pixels
    double A = BilinearInterpolation(integralImage, x0,   y0,   w);
    double B = BilinearInterpolation(integralImage, x1,   y0,   w);
    double C = BilinearInterpolation(integralImage, x0,   y1,   w);
    double D = BilinearInterpolation(integralImage, x1,   y1,   w);

    // compute D+A-B-C
    sum = D + A - B - C;

    return sum;
}

/*******************************************************************************
ComputeFeatures - Compute all of the features for a specific bounding box

integralImage - integral image
c0, r0 - position of upper lefthand corner of bounding box
size - Size of bounding box
features - Array for storing computed feature values, access using features[i] for all i less than numWeakClassifiers.
weakClassifiers - Weak classifiers
numWeakClassifiers - Number of weak classifiers
w - Width of image (integralImage)
*******************************************************************************/
void MainWindow::ComputeFeatures(double *integralImage, int c0, int r0, int size, double *features, 
    CWeakClassifiers *weakClassifiers, int numWeakClassifiers, int w)
{
    int i, j;

    // compute each classifier
    for(i=0;i<numWeakClassifiers;i++)
    {
        features[i] = 0.0;

        // sum pixels within each box
        for(j=0;j<weakClassifiers[i].m_NumBoxes;j++)
        {
            // compute box coordinates
            double x0 = weakClassifiers[i].m_Box[j][0][0]*size + c0;
            double y0 = weakClassifiers[i].m_Box[j][0][1]*size + r0;

            double x1 = weakClassifiers[i].m_Box[j][1][0]*size + c0;
            double y1 = weakClassifiers[i].m_Box[j][1][1]*size + r0;

            // compute sum of pixels in box
            double sum = SumBox(integralImage, x0, y0, x1, y1, w); 

            // store the final feature value
            features[i] += weakClassifiers[i].m_BoxSign[j]*sum / ((double)(size*size));
        } 
    }
}

/*******************************************************************************
FindBestClassifier - AdaBoost helper function.  Find the best threshold for the candidate classifier

featureSortIdx - Indexes of the training examples sorted based on the feature responses (lowest to highest)
Use these indices to index into the other arrays, i.e. features, trainingLabel, dataWeights.
features - Array of feature values for the candidate classifier
trainingLabel - Ground truth labels for the training examples (1 = face, 0 = background)
dataWeights - Weights used to weight each training example
numTrainingExamples - Number of training examples
candidateWeakClassifier - Candidate classifier
bestClassifier - Returned best classifier (updated threshold, weight and polarity)
*******************************************************************************/
double MainWindow::FindBestClassifier(int *featureSortIdx, double *features, int *trainingLabel, double *dataWeights,
    int numTrainingExamples, CWeakClassifiers candidateWeakClassifier, CWeakClassifiers *bestClassifier)
{
    double bestError = 99999999.0;

    // Copy the weak classifiers params
    candidateWeakClassifier.copy(bestClassifier);

    // train a classifier using a single feature
    double sPos = 0.0;
    double sNeg = 0.0;
    double tPos = 0.0;
    double tNeg = 0.0;

    // compute totals
    for(int i = 0; i < numTrainingExamples; i++)
    {
        // face (positive result)
        if(trainingLabel[i] == 1) { 
            tPos += dataWeights[i]; 
        }
        
        // background (negative result)
        else { 
            tNeg += dataWeights[i]; 
        }
    }

    // run through each training example
    for(int i = 0; i < numTrainingExamples; i++)
    {
        // find sorted feature index
        int idx = featureSortIdx[i];

        // update sums

        // face (positive result)
        if(trainingLabel[idx] == 1) { 
            sPos += dataWeights[idx]; 
        }
            
        // background (negative result)
        else { 
            sNeg += dataWeights[idx]; 
        }

        // compute positive and negative error
        double errorFaceAboveThres = sPos + (tNeg - sNeg);
        double errorFaceBelowThres = sNeg + (tPos - sPos);

        double error = std::min( errorFaceBelowThres, errorFaceAboveThres );

        // check if error decreased
        if(error < bestError)
        {
            // update best classifier error
            bestError = error;
            
            // determine polarity
            if(errorFaceAboveThres < errorFaceBelowThres) {
                // face above threshold
                bestClassifier->m_Polarity = 1;
            }
            else {
                // face below threshold
                bestClassifier->m_Polarity = 0;
            }

            // set threshold
            bestClassifier->m_Threshold = features[idx];

            // compute error weight
            bestClassifier->m_Weight = std::log((1.0-bestError)/bestError);
        }
    }

    return bestError;
}

/*******************************************************************************
UpdateDataWeights - AdaBoost helper function.  Updates the weighting of the training examples

features - Array of feature values for the candidate classifier
trainingLabel - Ground truth labels for the training examples (1 = face, 0 = background)
weakClassifier - A weak classifier
dataWeights - Weights used to weight each training example.  These are teh weights updated.
numTrainingExamples - Number of training examples
*******************************************************************************/
void MainWindow::UpdateDataWeights(double *features, int *trainingLabel, CWeakClassifiers weakClassifier, double *dataWeights, int numTrainingExamples)
{
    double total = 0.0;

    // compute beta from alpha weight
    double beta = 1.0 / std::exp(weakClassifier.m_Weight);
    bool ec = 0;

    for(int i = 0; i < numTrainingExamples; i++)
    {
        // initialize as not classified correctly
        ec = 1;

        // faces above threshold
        if( weakClassifier.m_Polarity == 1 )
        {
            // feature lower than threshold and not a face
            if(features[i] < weakClassifier.m_Threshold && trainingLabel[i] == 0) {
                ec = 0;
            }
            // feature higher than threshold and a face
            else if(features[i] >= weakClassifier.m_Threshold && trainingLabel[i] == 1) {
                ec = 0;
            }
        }

        // faces below threshold
        else if( weakClassifier.m_Polarity == 0 )
        {
            // feature lower than threshold and a face
            if(features[i] < weakClassifier.m_Threshold && trainingLabel[i] == 1) {
                ec = 0;
            }
            // feature higher than threshold and not a face
            else if(features[i] >= weakClassifier.m_Threshold && trainingLabel[i] == 0) {
                ec = 0;
            }
        }

        // compute new weight
        dataWeights[i] = dataWeights[i]*std::pow(beta, 1-ec);
        total += dataWeights[i];
    }

    // normalize weights
    for(int i = 0; i < numTrainingExamples; i++)
    {
        dataWeights[i] /= total;
    }
}

/*******************************************************************************
ClassifyBox - FindFaces helper function.  Return classification score for bounding box

integralImage - integral image
c0, r0 - position of upper lefthand corner of bounding box
size - Size of bounding box
weakClassifiers - Weak classifiers
numWeakClassifiers - Number of weak classifiers
w - Width of image (integralImage)
*******************************************************************************/
double MainWindow::ClassifyBox(double *integralImage, int c0, int r0, int size, CWeakClassifiers *weakClassifiers, int numWeakClassifiers, int w)
{
    // compute features for the classifier list
    double* features = new double[numWeakClassifiers];
    ComputeFeatures(integralImage, c0, r0, size, features, weakClassifiers, numWeakClassifiers, w);

    // compute score sums
    double weightedFeatureScore = 0.0;
    double weightScore = 0.0;

    // compute classification score
    for(int i = 0; i < numWeakClassifiers; i++)
    {
        if(features[i] > weakClassifiers[i].m_Threshold)
        {
            weightedFeatureScore += weakClassifiers[i].m_Weight*weakClassifiers[i].m_Polarity;
        }
        else
        {
            weightedFeatureScore += weakClassifiers[i].m_Weight*(1.0 - weakClassifiers[i].m_Polarity);
        }

        weightScore += weakClassifiers[i].m_Weight;
    }

    // sum_t alpha_t*h_t(x) - 0.5*sum_t alpha_t
    double score = weightedFeatureScore - (0.5*weightScore);

    delete [] features;

    return score;
}

/*******************************************************************************
NMS - Non-maximal suppression of face detections. 
Neighboring face detections must be beyond xyThreshold AND scaleThreshold in 
position and scale respectivitely.

faceDetections - Set of face detections
xyThreshold - Minimum distance in position between neighboring detections
scaleThreshold - Minimum distance in scale between neighboring detections
displayImage - Display image
*******************************************************************************/
void MainWindow::NMS(QMap<double, CDetection> *faceDetections, double xyThreshold, double scaleThreshold, QImage *displayImage)
{  
    // store the final set of face detections
    QMap<double, CDetection> finalFaceDetections;

    // iterate through all the faces detections (lowest face detection score first)
    QMap<double, CDetection>::const_iterator iterCurrent = faceDetections->constBegin();
    while(iterCurrent != faceDetections->constEnd())
    {
        // scan through higher scoring detections
        QMap<double, CDetection>::const_iterator iterScan = iterCurrent;
        iterScan++;

        while(iterScan != faceDetections->constEnd())
        {
            // compute distance
            double distance = std::sqrt( 
                std::pow((iterCurrent->m_X - iterScan->m_X), 2.0) + 
                std::pow((iterCurrent->m_Y - iterScan->m_Y), 2.0) );

            // compute scale
            double scale = std::sqrt( std::pow(iterCurrent->m_Scale - iterScan->m_Scale, 2.0) );
            
            // if detection found within position and scale thresholds, stop scan
            if( distance < xyThreshold && scale < scaleThreshold ) {
                break;
            }

            // increment scan
            iterScan++;
        }
     
        // if scan did not find a neighboring detection
        if(iterScan == faceDetections->constEnd()) {
            // add current face detection to finalFaceDetections
            finalFaceDetections.insertMulti(iterCurrent.key(), iterCurrent.value());
        }

        // increment current detection
        iterCurrent++;
    }

    // draw detections
    DrawFace(displayImage, &finalFaceDetections);
}
