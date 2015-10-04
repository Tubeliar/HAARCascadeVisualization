#ifndef VISUAL_CASCADE_HPP
#define VISUAL_CASCADE_HPP

#include "objdetect/cascadedetect.hpp"
#include "objdetect/VisualHaar.hpp"
#include <vector>
#include <opencv2/highgui.hpp>

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
		cv::Size maxSize = cv::Size(),
		unsigned steps = 1);

	int getDepth() const;
	void setIntegral(cv::Size integralSize, cv::Mat sum, cv::Mat sqsum);
	void setWindow(int x, int y, cv::Size windowSize, cv::Size ssz);
	void keepWindow();
	void show(const std::vector<int>& branches, int featureIndex, int nFeatures, const CvHidHaarFeature& feature);
	void show(int stage, int featureIndex, int nFeatures, const CvHidHaarFeature& feature);
	void show(std::string caption, int featureIndex, int nFeatures, const CvHidHaarFeature& feature);
	void drawFeature(cv::Mat image, const CvHidHaarFeature& feature);
	void setVideo(std::string videoFilename);
	void setImagePath(std::string imagePath);
	cv::Mat getProgressImage() const;;
	bool isRecording() const;
	void recordImage(const cv::Mat image);

	static std::string mWindowName;

protected:
	void borderText(cv::Mat& image, std::string text, cv::Point origin, int font, double scale, cv::Scalar colour, cv::Scalar borderColour = cv::Scalar(0, 0, 0));
	cv::Mat mProgress;
	double mShowScale;
	cv::Rect mWindow;
	cv::Size mIntegralSize;
	cv::Mat mSum;
	cv::Mat mSqsum;
	cv::Size mOriginalWindowSize;
	int mVisualisationDepth;
	std::string mImagePath;
	std::string mVideoPath;
	cv::VideoWriter mOutVideo;
	unsigned mFrameCounter;
	unsigned mSteps;
	unsigned mStepCounter;
};

#endif