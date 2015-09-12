#include "VisualCascade.hpp"

#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };
struct getNeighbors { int operator ()(const CvAvgComp& e) const { return e.neighbors; } };

void VisualCascade::detectMultiScale(InputArray _image, std::vector<Rect>& objects,
	double scaleFactor, int minNeighbors,
	int flags, Size minObjectSize, Size maxObjectSize)
{
	std::cout << "Subclassed!" << std::endl;
	std::vector<int> rejectLevels;
	std::vector<double> levelWeights;
	Mat image = _image.getMat();
	CV_Assert(scaleFactor > 1 && image.depth() == CV_8U);

	if (empty()) return;

	std::vector<int> numDetections;
	if (isOldFormatCascade())
	{
		std::vector<CvAvgComp> vecAvgComp;

		MemStorage storage(cvCreateMemStorage(0));
		CvMat _image = image;
		CvSeq* _objects = cvHaarDetectObjectsForROC(&_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
			minNeighbors, flags, minObjectSize, maxObjectSize, false);
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