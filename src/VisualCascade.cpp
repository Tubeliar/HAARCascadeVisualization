#include "VisualCascade.hpp"
#include <iostream>

using namespace cv;

void VisualCascade::detectMultiScale(InputArray _image, std::vector<Rect>& objects,
	double scaleFactor, int minNeighbors,
	int flags, Size minObjectSize, Size maxObjectSize)
{
	std::cout << "Subclassed!" << std::endl;
	std::vector<int> fakeLevels;
	std::vector<double> fakeWeights;
	CascadeClassifierImpl::detectMultiScale(_image, objects, fakeLevels, fakeWeights, scaleFactor,
		minNeighbors, flags, minObjectSize, maxObjectSize);
}