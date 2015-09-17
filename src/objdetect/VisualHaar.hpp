#ifndef VISUAL_HAAR_HPP
#define VISUAL_HAAR_HPP

#include <opencv2/core/core_c.h>
#include <opencv2/objdetect/objdetect_c.h>

class VisualCascade;

// We have some function signatures that differ from the original ones in objectdetect, so this
// header file can be included for forward declarations to those
CvSeq*
viscasHaarDetectObjectsForROC(const CvArr* _img,
	CvHaarClassifierCascade* cascade, CvMemStorage* storage,
	std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
	double scaleFactor, int minNeighbors, int flags,
	CvSize minSize, CvSize maxSize, bool outputRejectLevels, VisualCascade* pVisCas);

typedef int sumtype;

typedef struct HaarFeatureRect
{
	sumtype *p0, *p1, *p2, *p3;
	float weight;
} HaarFeatureRect;

typedef struct CvHidHaarFeature
{
	HaarFeatureRect rect[CV_HAAR_FEATURE_MAX];
} CvHidHaarFeature;

#endif