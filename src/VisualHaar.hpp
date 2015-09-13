#ifndef VISUAL_HAAR_HPP
#define VISUAL_HAAR_HPP

#include <opencv2/core/core_c.h>
#include "VisualCascade.hpp"

// We have some function signatures that differ from the original ones in objectdetect, so this
// header file can be included for forward declarations to those
CvSeq*
viscasHaarDetectObjectsForROC(const CvArr* _img,
	CvHaarClassifierCascade* cascade, CvMemStorage* storage,
	std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
	double scaleFactor, int minNeighbors, int flags,
	CvSize minSize, CvSize maxSize, bool outputRejectLevels, VisualCascade* pVisCas);

#endif