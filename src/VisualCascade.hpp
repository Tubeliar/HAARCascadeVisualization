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
		int depth = 3,
		double scaleFactor = 1.1,
		int minNeighbors = 3, int flags = 0,
		cv::Size minSize = cv::Size(),
		cv::Size maxSize = cv::Size());

	int getDepth() const;
	void setIntegral(cv::Size integralSize, cv::Mat sum, cv::Mat sqsum);
	void setWindow(int x, int y, cv::Size windowSize, cv::Size ssz);
	void keepWindow();
	void show(const std::vector<int>& branches, int featureIndex, int nFeatures, CvHidHaarFeature& feature);
	void show(int stage, int featureIndex, int nFeatures, CvHidHaarFeature& feature);
	void show(std::string caption, int featureIndex, int nFeatures, CvHidHaarFeature& feature);
	void drawFeature(cv::Mat image, CvHidHaarFeature& feature);

	static std::string mWindowName;

protected:
	cv::Mat mProgress;
	double mShowScale;
	cv::Rect mWindow;
	cv::Size mIntegralSize;
	cv::Mat mSum;
	cv::Mat mSqsum;
	cv::Size mOriginalWindowSize;
	int mVisualisationDepth;
};

#endif