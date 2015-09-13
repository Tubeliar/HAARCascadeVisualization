#ifndef VISUAL_CASCADE_HPP
#define VISUAL_CASCADE_HPP

#include "objdetect/cascadedetect.hpp"

class VisualCascade : public cv::CascadeClassifierImpl
{
public:
	void detectMultiScale(cv::InputArray image,
		CV_OUT std::vector<cv::Rect>& objects,
		double scaleFactor = 1.1,
		int minNeighbors = 3, int flags = 0,
		cv::Size minSize = cv::Size(),
		cv::Size maxSize = cv::Size());

	void show(int x, int y, cv::Size windowSize, cv::Size ssz);

protected:
	cv::Mat mOriginal;
	cv::Mat mProgress;
};

#endif