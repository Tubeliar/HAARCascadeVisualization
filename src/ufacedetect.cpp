#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

#include "VisualCascade.hpp"

using namespace std;
using namespace cv;

static void help()
{
    cout << "Usage:\n"
            "./facedetect [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
               "   [--detectscale=<image scale to perform the detection. Smaller scales detect faster and coarser>]\n"
               "   [--showscale=<image scale to do the visualisation at. This will not affect the detection>]\n"
               "   [--scalefactor=<factor greater than 1. Bigger numbers cause a coarser bnt faster search>]\n"
               "   [filename]\n\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( UMat& img, Mat& canvas, VisualCascade& cascade, double detectScale, double showScale, double factor );

string cascadeName = "../../data/haarcascades/haarcascade_frontalface_alt.xml";

int main( int argc, const char** argv )
{
    VideoCapture capture;
    UMat frame, image;
    Mat canvas;
	const string detectScaleOpt = "--detectscale=";
	size_t detectScaleOptLen = detectScaleOpt.length();
	const string showScaleOpt = "--showscale=";
	size_t showScaleOptLen = showScaleOpt.length();
	const string scaleFactorOpt = "--scalefactor=";
	size_t scaleFactorOptLen = scaleFactorOpt.length();
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    String inputName;

    VisualCascade cascade;
	double detectScale = 1;
	double showScale = 1;
	double factor = 1.5;

    for( int i = 1; i < argc; i++ )
    {
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
            cout << "cascadeName = " << cascadeName << endl;
        }
		else if (detectScaleOpt.compare(0, detectScaleOptLen, argv[i], detectScaleOptLen) == 0)
		{
			if (!sscanf(argv[i] + detectScaleOptLen, "%lf", &detectScale) || detectScale > 1)
				detectScale = 1;
			cout << "detection scale = " << detectScale << endl;
		}
		else if (showScaleOpt.compare(0, showScaleOptLen, argv[i], showScaleOptLen) == 0)
		{
			if (!sscanf(argv[i] + showScaleOptLen, "%lf", &showScale))
				showScale = 1;
			cout << "detection scale = " << showScale << endl;
		}
		else if (scaleFactorOpt.compare(0, scaleFactorOptLen, argv[i], scaleFactorOptLen) == 0)
		{
			if (!sscanf(argv[i] + scaleFactorOptLen, "%lf", &factor) || factor <= 1)
				factor = 1.5;
			cout << "scale factor = " << factor << endl;
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

    image = imread( inputName, 1 ).getUMat(ACCESS_READ);
    if( image.empty() )
    {
		cout << "Could not read " << inputName << endl;
		help();
		return -1;
    }

	cout << "Detecting face(s) in " << inputName << endl;
    if( !image.empty() )
    {
        detectAndDraw( image, canvas, cascade, detectScale, showScale, factor );
        waitKey(0);
    }

    return 0;
}

void detectAndDraw( UMat& img, Mat& canvas, VisualCascade& cascade, double detectScale, double showScale, double factor )
{
    int i = 0;
    double scale=1;
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
    static UMat gray, smallImg, visualisationImage;

    resize( img, smallImg, Size(), detectScale, detectScale, INTER_LINEAR );
	resize( img, visualisationImage, Size(), showScale, showScale, INTER_LINEAR);
    cvtColor( smallImg, gray, COLOR_BGR2GRAY );
    equalizeHist( gray, gray );

    cascade.detectMultiScale(visualisationImage, gray, faces,
        showScale / detectScale, factor, 3, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(30, 30) );
    smallImg.copyTo(canvas);

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
	destroyWindow(VisualCascade::mWindowName);
	namedWindow("result", 1);
    imshow( "result", canvas );
}
