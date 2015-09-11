#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace std;
using namespace cv;

static void help()
{
    cout << "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
               "   [filename]\n\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( UMat& img, Mat& canvas, CascadeClassifier& cascade, double scale );

string cascadeName = "../../data/haarcascades/haarcascade_frontalface_alt.xml";

int main( int argc, const char** argv )
{
    VideoCapture capture;
    UMat frame, image;
    Mat canvas;
    const string scaleOpt = "--scale=";
    size_t scaleOptLen = scaleOpt.length();
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    String inputName;

    CascadeClassifier cascade;
    double scale = 1;

    for( int i = 1; i < argc; i++ )
    {
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "cascadeName = " << cascadeName << endl;
        }
        else if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
        {
            if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale > 1 )
                scale = 1;
            cout << "scale = " << scale << endl;
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        }
        else
            inputName = argv[i];
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help();
        return -1;
    }

    cout << "old cascade: " << (cascade.isOldFormatCascade() ? "TRUE" : "FALSE") << endl;

    image = imread( inputName, 1 ).getUMat(ACCESS_READ);
    if( image.empty() )
    {
		cout << "Could not read " << inputName << endl;
		help();
		return -1;
    }

    namedWindow( "result", 1 );

	cout << "Detecting face(s) in " << inputName << endl;
    if( !image.empty() )
    {
        detectAndDraw( image, canvas, cascade, scale );
        waitKey(0);
    }

    return 0;
}

void detectAndDraw( UMat& img, Mat& canvas, CascadeClassifier& cascade, double scale0 )
{
    int i = 0;
    double t = 0, scale=1;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(0,0,255),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,255,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(255,0,0),
        Scalar(255,0,255)
    };
    static UMat gray, smallImg;

    t = (double)getTickCount();

    resize( img, smallImg, Size(), scale0, scale0, INTER_LINEAR );
    cvtColor( smallImg, gray, COLOR_BGR2GRAY );
    equalizeHist( gray, gray );

    cascade.detectMultiScale( gray, faces,
        1.1, 3, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(30, 30) );
    t = (double)getTickCount() - t;
    smallImg.copyTo(canvas);

    putText(canvas, format("OpenCL: %s", ocl::useOpenCL() ? "ON" : "OFF"), Point(50, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0,255,0), 2);

    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r->width/r->height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            circle( canvas, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( canvas, Point(cvRound(r->x*scale), cvRound(r->y*scale)),
                       Point(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                       color, 3, 8, 0);
    }
    imshow( "result", canvas );
}
