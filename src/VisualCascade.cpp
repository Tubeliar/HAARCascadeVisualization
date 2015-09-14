#include "VisualCascade.hpp"

#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "VisualHaar.hpp"

using namespace cv;
using namespace std;

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };
struct getNeighbors { int operator ()(const CvAvgComp& e) const { return e.neighbors; } };

string VisualCascade::mWindowName = "Cascade Visualiser";

void VisualCascade::detectMultiScale(InputArray showImage, InputArray _image, std::vector<Rect>& objects,
	double showScale, double scaleFactor, int minNeighbors,
	int flags, Size minObjectSize, Size maxObjectSize)
{
	mShowScale = showScale;
	std::vector<int> rejectLevels;
	std::vector<double> levelWeights;
	mProgress = showImage.getMat();
	Mat image = _image.getMat();
	CV_Assert(scaleFactor > 1 && image.depth() == CV_8U);

	if (empty()) return;

	std::vector<int> numDetections;
	if (isOldFormatCascade())
	{
		std::vector<CvAvgComp> vecAvgComp;

		MemStorage storage(cvCreateMemStorage(0));
		CvMat _image = image;
		CvSeq* _objects = viscasHaarDetectObjectsForROC(&_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
			minNeighbors, flags, minObjectSize, maxObjectSize, false, this);
		Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
		objects.resize(vecAvgComp.size());
		std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());

		numDetections.resize(vecAvgComp.size());
		std::transform(vecAvgComp.begin(), vecAvgComp.end(), numDetections.begin(), getNeighbors());
	}
	else
	{
		cout << "New format cascade not supported for visualisation" << endl;
		detectMultiScaleNoGrouping(image, objects, rejectLevels, levelWeights, scaleFactor, minObjectSize, maxObjectSize);
		const double GROUP_EPS = 0.2;
		groupRectangles(objects, numDetections, minNeighbors, GROUP_EPS);
	}
}

void VisualCascade::show(int x, int y, Size windowSize, Size ssz, bool keep)
{
	Mat result;
	Size showWindow(static_cast<int>(mShowScale * windowSize.width), static_cast<int>(mShowScale * windowSize.height));
	int xOffset = (mProgress.cols - showWindow.width)  * x / ssz.width;
	int yOffset = (mProgress.rows - showWindow.height) * y / ssz.height;
	//cout << ssz << " " << x << " " << xOffset << endl;
	if (keep) rectangle(mProgress, Rect(Point(xOffset, yOffset), showWindow), Scalar(0, 255, 0));
	mProgress.copyTo(result);
	rectangle(result, Rect(Point(xOffset, yOffset), showWindow), Scalar(0, 0, 255), 2);
	imshow(mWindowName, result);
	waitKey(1);
}