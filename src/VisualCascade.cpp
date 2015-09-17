#include "VisualCascade.hpp"

#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include "objdetect/VisualHaar.hpp"

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

void VisualCascade::setIntegral(cv::Size integralSize, cv::Mat sum, cv::Mat sqsum)
{
	mIntegralSize = integralSize;
	mSum = sum;
	mSqsum = sqsum;
}

void VisualCascade::setWindow(int x, int y, Size detectWindowSize, Size ssz)
{
	Size showWindowSize(static_cast<int>(mShowScale * detectWindowSize.width), static_cast<int>(mShowScale * detectWindowSize.height));
	int xOffset = (mProgress.cols - showWindowSize.width)  * x / ssz.width;
	int yOffset = (mProgress.rows - showWindowSize.height) * y / ssz.height;
	mWindow = Rect(Point(xOffset, yOffset), showWindowSize);
}

void VisualCascade::show(const vector<int>& branches, int featureIndex, int nFeatures, CvHidHaarFeature& feature, int offset)
{
	Mat result;
	mProgress.copyTo(result);
	rectangle(result, mWindow, Scalar(0, 0, 255), 2);
	drawFeature(result, feature, offset);

	stringstream description;
	description << "Branch: ";
	for (unsigned index = 0; index < branches.size(); index++)
	{
		if (index > 0) description << "-";
		description << branches[index];
	}
	putText(result, description.str(), Point(mWindow.x, mWindow.y + mWindow.height + 12), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
	description.str("");
	description << "Feature: " << featureIndex << " of " << nFeatures;
	putText(result, description.str(), Point(mWindow.x, mWindow.y + mWindow.height + 24), CV_FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
	imshow(mWindowName, result);
	waitKey(1);
}

void VisualCascade::drawFeature(cv::Mat image, CvHidHaarFeature& feature, int offset)
{
	//cout << mIntegralSize << " " << mSum.size() << " " << mSqsum.size() << endl;
	for (int rectIndex = 0; rectIndex < CV_HAAR_FEATURE_MAX; rectIndex++)
	{
		HaarFeatureRect hfr = feature.rect[rectIndex];
		if (!hfr.p0) break;
		int stride = mIntegralSize.width + 1;

		int topLIndex = hfr.p0 - reinterpret_cast<sumtype*>(mSum.data); // Use to draw
		int topRIndex = hfr.p1 - reinterpret_cast<sumtype*>(mSum.data); // Use to check
		int botLIndex = hfr.p2 - reinterpret_cast<sumtype*>(mSum.data); // Use to check
		int botRIndex = hfr.p3 - reinterpret_cast<sumtype*>(mSum.data); // Use to draw

		// Perform checks so make sure we have the right stride
		if (topRIndex % stride != botRIndex % stride) cout << "p1 misaligned x" << endl;
		if (topRIndex / stride != topLIndex / stride) cout << "p1 misaligned y" << endl;
		if (botLIndex % stride != topLIndex % stride) cout << "p2 misaligned x" << endl;
		if (botLIndex / stride != botRIndex / stride) cout << "p1 misaligned y" << endl;
		Point topL((topLIndex % stride) * mWindow.width / mIntegralSize.width, (topLIndex / stride) * mWindow.height / mIntegralSize.height);
		Point botR((botRIndex % stride) * mWindow.width / mIntegralSize.width, (botRIndex / stride) * mWindow.height / mIntegralSize.height);
		topL += mWindow.tl();
		botR += mWindow.tl();
		Scalar color = hfr.weight > 0 ? Scalar(255, 255, 255) : Scalar(0, 0, 0);
		rectangle(image, Rect(topL, botR), color, CV_FILLED);
	}
}

void VisualCascade::keepWindow()
{
	rectangle(mProgress, mWindow, Scalar(0, 255, 0));
}