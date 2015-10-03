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

static void help(const char * executableName)
{
    cout << "Usage:\n"
	           "./" << executableName << " <image filename>\n"
	           "   <--cascade=<cascade_path> this is the primary trained classifier such as frontal face>\n"
               "   [--detectscale=<image scale to perform the detection. Since the detection is multiscale starting at a smaller scale just means stopping sooner>]\n"
               "   [--showscale=<image scale to do the visualisation at. This will not affect the detection>]\n"
               "   [--scalefactor=<Multiscale step. Bigger than 1. Bigger numbers cause a coarser but faster search>]\n"
               "   [--depth=<cascade depth to visualise. Deeper levels will still be performed but not shown>]\n"
		       "   [--images=<folder name in which the individual frames will be written. The path must exist.>]\n"
		       "   [--video=<file name for an avi video that will be created>]\n"
               "\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( UMat& img, VisualCascade& cascade, double detectScale, double showScale, double factor, int depth );

string cascadeName = "../../data/haarcascades/haarcascade_frontalface_alt.xml";

bool findOpt(const char * command, const string option, const char *& argument)
{
	size_t opLen = option.length();
	if (option.compare(0, opLen, command, opLen) == 0)
	{
		argument = command + opLen;
		return true;
	}
	argument = command;
	return false;
}

int main( int argc, const char** argv )
{
    VideoCapture capture;
    UMat frame, image;
    String inputName;

    VisualCascade cascade;
	double detectScale = 1;
	double showScale = 1;
	double factor = 1.3;
	int depth = 4;
	string outputPath;
	string videoPath;

	const char * argument = 0;
	double doubleValue;
	int intValue;

    for( int i = 1; i < argc; i++ )
    {
        if( findOpt(argv[i], "--cascade=", argument) )
        {
            cascadeName.assign( argument );
            cout << "cascadeName = " << cascadeName << endl;
        }
		else if ( findOpt(argv[i], "--detectscale=", argument) )
		{
			if (sscanf(argument, "%lf", &doubleValue)) detectScale = doubleValue;
			cout << "detection scale = " << detectScale << endl;
		}
		else if ( findOpt(argv[i], "--showscale=", argument) )
		{
			if (sscanf(argument, "%lf", &doubleValue)) showScale = doubleValue;
			cout << "detection scale = " << showScale << endl;
		}
		else if (findOpt(argv[i], "--scalefactor=", argument))
		{
			if (sscanf(argument, "%lf", &doubleValue) && doubleValue > 1) factor = doubleValue;
			cout << "scale factor = " << factor << endl;
		}
		else if (findOpt(argv[i], "--depth=", argument))
		{
			if (sscanf(argument, "%d", &intValue)) depth = intValue;
			cout << "visualise cascade depth = " << depth << endl;
		}
		else if (findOpt(argv[i], "--images=", argument))
		{
			outputPath = argument;
			cout << "write image series in " << outputPath << endl;
		}
		else if (findOpt(argv[i], "--video=", argument))
		{
			videoPath = argument;
			cout << "write video " << videoPath << endl;
		}
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        }
        else inputName = argv[i];
    }

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help(argv[0]);
        return -1;
    }

    image = imread( inputName, 1 ).getUMat(ACCESS_READ);
    if( image.empty() )
    {
		cout << "Could not read " << inputName << endl;
		help(argv[0]);
		return -1;
    }

	if (!outputPath.empty()) cascade.setImagePath(outputPath);
	if (!videoPath.empty()) cascade.setVideo(outputPath);

	cout << "Detecting face(s) in " << inputName << endl;
    if( !image.empty() )
    {
        detectAndDraw( image, cascade, detectScale, showScale, factor, depth );
        waitKey(0);
    }

    return 0;
}

void detectAndDraw( UMat& img, VisualCascade& cascade, double detectScale, double showScale, double factor , int depth)
{
    int i = 0;
	double scale = showScale / detectScale;
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

	int minNeighbours = 3;
	int flags = 0
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| CASCADE_SCALE_IMAGE
		;
    cascade.detectMultiScale(visualisationImage, gray, faces,
        showScale / detectScale, depth, factor, minNeighbours, 
        flags, Size(30, 30) );
	Mat canvas = cascade.getProgressImage();

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
	if (cascade.isRecording())
	{
		// Show the result for longer than one frame
		for (int i = 0; i < 90; i++)
		{
			cascade.recordImage(canvas);
		}
	}
    imshow(VisualCascade::mWindowName, canvas );
}
