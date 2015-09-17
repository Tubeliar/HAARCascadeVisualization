#ifndef VISUAL_CASCADE_HPP
#define VISUAL_CASCADE_HPP

#include "objdetect/cascadedetect.hpp"
#include "objdetect/VisualHaar.hpp"
#include <vector>

class VisualCascade : public cv::CascadeClassifierImpl
{
public:
	void detectMultiScale(cv::InputArray show,
		cv::InputArray image,
		CV_OUT std::vector<cv::Rect>& objects,
		double showScale = 1,
		double scaleFactor = 1.1,
		int minNeighbors = 3, int flags = 0,
		cv::Size minSize = cv::Size(),
		cv::Size maxSize = cv::Size());

	void setIntegral(cv::Size integralSize, cv::Mat sum, cv::Mat sqsum);
	void setWindow(int x, int y, cv::Size windowSize, cv::Size ssz);
	void keepWindow();
	void show(const std::vector<int>& branches, int featureIndex, int nFeatures, CvHidHaarFeature& feature, int offset);
	void drawFeature(cv::Mat image, CvHidHaarFeature& feature, int offset);

	static std::string mWindowName;

protected:
	cv::Mat mProgress;
	double mShowScale;
	cv::Rect mWindow;
	cv::Size mIntegralSize;
	cv::Mat mSum;
	cv::Mat mSqsum;
};

#endif