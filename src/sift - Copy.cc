#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include  "opencv2/legacy/legacy.hpp" // ????????
#include  "opencv2/nonfree/nonfree.hpp"
#include <iostream>  
#include <vector> 

#include "cv_import_static_lib.h"

using namespace std;
using namespace cv;

void main() {


	Mat img_1 = imread("E:\\3Dtestdata\\3.jpg");
	Mat img_2 = imread("E:\\3Dtestdata\\4.jpg");
	if (!img_1.data || !img_2.data)
	{
		cout << "error reading images " << endl;
		return;
	}

	vector<KeyPoint> keyPoints_1, keyPoints_2;
	Mat descriptors_1, descriptors_2;

	/*-----------------SIFT featrue Point----------------*/
	SIFT sift;
	sift(img_1, Mat(), keyPoints_1, descriptors_1);
	sift(img_2, Mat(), keyPoints_2, descriptors_2);
	

	/*-----------------SURF featrue Point----------------
	SURF surf;
	surf(img_1, Mat(), keyPoints_1, descriptors_1);
	surf(img_2, Mat(), keyPoints_2, descriptors_2);
	//SurfDescriptorExtractor extrator;           // another surf sift operation
	//extrator.compute(img_1, keyPoints_1, descriptors_1);
	//extrator.compute(img_2, keyPoints_2, descriptors_2);
	*/

	//BruteForceMatcher<HammingLUT> matcher;// orb ?float?? 

	cv::FlannBasedMatcher matcher;  // ?? ?uchar???????
	vector< cv::DMatch > matches;
	std::vector<cv::DMatch> viewMatches;
	matcher.match(descriptors_1, descriptors_2, matches);


	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints  
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	cout << "-- Max dist :" << max_dist << endl;
	cout << "-- Min dist :" << min_dist << endl;
	//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )  
	//-- PS.- radiusMatch can also be used here.  
	vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance < 0.6*max_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}



	// ??????F
	// ????
	int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);
	// ?Keypoint???Mat
	Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		pt = keyPoints_1[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = keyPoints_2[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	// ?RANSAC???? ????F
	Mat m_Fundamental;
	vector<uchar> m_RANSACStatus;
	vector<DMatch> m_InlierMatches;

	m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);// ??????																  
    // ????
	for (int i = 0; i<ptCount; i++)
	{
		if (m_RANSACStatus[i] != 0)
		{
			m_InlierMatches.push_back(matches[i]);
		}
	}
	// ????F???????
	Mat OutImage;
	drawMatches(img_1, keyPoints_1, img_2, keyPoints_2, m_InlierMatches, OutImage);

	Mat img_matches;
	drawMatches(img_1, keyPoints_1, img_2, keyPoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite("FASTResult.jpg", img_matches);
	imshow("Match", img_matches);

	imwrite("FmatrixResult.jpg", OutImage);
	imshow("Match2", OutImage);
	waitKey(0);

	return;
}