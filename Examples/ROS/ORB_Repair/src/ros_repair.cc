/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "../../../include/System.h"

#include "MsgSync/MsgSynchronizer.h"

#include "../../../src/IMU/imudata.h"
#include "../../../src/IMU/configparam.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>


#include "opencv2/highgui/highgui.hpp"  
#include  "opencv2/legacy/legacy.hpp" 
#include  "opencv2/nonfree/nonfree.hpp"
#include <opencv2/features2d/features2d.hpp>
#include "GMS-Feature-Matcher-master/include/Header.h"
#include "GMS-Feature-Matcher-master/include/gms_matcher.h"

#include "repairTracking.h"

using namespace cv;

#define raw

using namespace std;

void LoadImu(const string &, std::vector<sensor_msgs::Imu> &);

void LoadImageTime(const string &,std::vector<string> &,const string);

void LoadOneImages(const string,const string ,const string imagetype,sensor_msgs::Image & );

void SplitString(const string& , vector<string>& , const string& );



void SplitString(const string& s, vector<string>& v, const string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));
         
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
}



void LoadImageTime(const string &strPathTimes,std::vector<string> &vstrImagesTime, const string split)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    sensor_msgs::Image getOneImage;
    cout<< " Image getting "<<endl;
    while(!fTimes.eof())
    {
        string s;     
        getline(fTimes,s);     
        if(!s.empty())
        {	
	        vector<string> v;
	        SplitString(s, v,split); 
	        s=v[0];
	        //cout << s <<endl;
	        vstrImagesTime.push_back(s); 
        }
        
    }
    cout<< " finish "<<endl;
}

void LoadOneImages(const string ImagePath,const string Times,const string imagetype,sensor_msgs::Image& ImageMsg)
{  
   
    Mat cv_image = imread(ImagePath + "/" + Times + imagetype,CV_LOAD_IMAGE_UNCHANGED);
    //cout<<cv_image.channels()<<endl;
    if(cv_image.empty() )
    {
    	//cout<< Times<< endl;
		ROS_ERROR("Read the picture failed!");
		return ;
	}
    cv_bridge::CvImage cvi;
    istringstream s1;  
    s1.str(Times);
    double time;
    s1>>setprecision(20)>>time;
    //cout<<setprecision(20)<<time<<endl;
    //cout<<setprecision(20)<<atof(Times.c_str())<<endl;
	cvi.header.stamp = ros::Time(time);//atof(Times.c_str()));
	cvi.header.frame_id = "image";
	cvi.encoding ="rgb8";//"mono8"; //"rgb8";
	cvi.image = cv_image;
	cvi.toImageMsg(ImageMsg);
    //cv::namedWindow("test");
    //cv::imshow("test",cv_image);
    
}

void LoadOneImagesFromDataset(const string ImagePath,const string Times,const string imagetype,sensor_msgs::Image& ImageMsg)
{  
    Mat cv_image = imread(ImagePath + "/" + Times + imagetype,CV_LOAD_IMAGE_GRAYSCALE);
    if(cv_image.empty() )
    {
        //cout<< Times<< endl;
        ROS_ERROR("Read the picture failed!");
        return ;
    }
    cv_bridge::CvImage cvi;
    istringstream s1;  
    s1.str(Times);
    double time;
    s1>>setprecision(20)>>time;
    //cout<<setprecision(20)<<time<<endl;
    //cout<<setprecision(20)<<atof(Times.c_str())<<endl;
    cvi.header.stamp = ros::Time(time/1000000);//atof(Times.c_str()));
    cvi.header.frame_id = "image";
    cvi.encoding = "mono8";
    cvi.image = cv_image;
    cvi.toImageMsg(ImageMsg);
    //cv::namedWindow("test");
    //cv::imshow("test",cv_image);
    
}



void LoadImu(const string &strImuPath, std::vector<sensor_msgs::Imu> &ImuMsg)
{
	ifstream fTimes;
    fTimes.open(strImuPath.c_str());
    sensor_msgs::Imu getOneImu;
    cout<< " IMU getting "<<endl;
    while(!fTimes.eof())
    {
        string s;     
        double time;
        getline(fTimes,s);
        vector<string> v;
        double data[6]={0};
        SplitString(s, v," "); 
        if(!s.empty())
        {
            time = atof(v[0].c_str());   
            for(int j=0;j<6;j++)
            {         
            	data[j] =atof(v[j+1].c_str());
            	//cout<< data[j] <<" ";
            }    
            //cout<<endl;        
			getOneImu.header.stamp = ros::Time(time);
			getOneImu.header.frame_id = "imu";
	        getOneImu.angular_velocity.x = data[0];
	        getOneImu.angular_velocity.y = data[1];
	        getOneImu.angular_velocity.z = data[2];
	        getOneImu.linear_acceleration.x = data[3];
	        getOneImu.linear_acceleration.y = data[4];
	        getOneImu.linear_acceleration.z = data[5];
            ImuMsg.push_back(getOneImu);
        }
        
    }
    cout<< " finish "<<endl;
}

void descriptorsAndMatch(Mat& img_1,Mat& img_2,bool flag)
{
    vector<KeyPoint> keyPoints_1, keyPoints_2;
    Mat descriptors_1, descriptors_2;
    if(flag)
    {
        SIFT sift= SIFT(3000,6,0.008,10,0.9);
        sift(img_1, Mat(), keyPoints_1, descriptors_1);
        sift(img_2, Mat(), keyPoints_2, descriptors_2);
    }
    else
    {
        SURF surf;
        surf(img_1, Mat(), keyPoints_1, descriptors_1);
        surf(img_2, Mat(), keyPoints_2, descriptors_2);  
    }

    cout<<"keyPoints_1"<<keyPoints_1.size()<<endl;
    cout<<"keyPoints_2"<<keyPoints_2.size()<<endl;

     cv::FlannBasedMatcher matcher;  
    //BFMatcher matcher;
    vector< cv::DMatch > matches;
    std::vector<cv::DMatch> viewMatches;
    matcher.match(descriptors_1, descriptors_2, matches);
    cout<<"matcher "<<matches.size()<<endl;
    int ptCount = (int)matches.size();
    Mat p1(ptCount, 2, CV_32F);
    Mat p2(ptCount, 2, CV_32F);
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
    Mat m_Fundamental;
    vector<uchar> m_RANSACStatus;
    vector<DMatch> m_InlierMatches;

    m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);//                                                             
    for (int i = 0; i<ptCount; i++)
    {
        if (m_RANSACStatus[i] != 0)
        {
            m_InlierMatches.push_back(matches[i]);
        }
    }
    cout<<"m_InlierMatches "<<m_InlierMatches.size()<<endl;
    Mat mImage;
    drawMatches(img_1, keyPoints_1, img_2, keyPoints_2, m_InlierMatches, mImage);
    if(flag)
    {
        namedWindow( "SIFT", WINDOW_AUTOSIZE );
        imshow("SIFT", mImage);
    }
    else
    {
        namedWindow( "SURF ", WINDOW_AUTOSIZE );
        imshow("SURF", mImage);
    }

    int num_inliers = 0;
    std::vector<bool> vbInliers;
    vector<DMatch> matches_all, matches_gms;
    matches_all=matches;
    
    gms_matcher gms(keyPoints_1,img_1.size(), keyPoints_2,img_2.size(), matches_all);
    num_inliers = gms.GetInlierMask(vbInliers, false, false);


    // draw matches
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(matches_all[i]);
        }
    }
    cout<<"surf GMS filter matches_gms"<<matches_gms.size()<<endl;
    Mat mImage2;
    drawMatches(img_1, keyPoints_1, img_2, keyPoints_2, matches_gms, mImage2);
    if(flag)
    {
        namedWindow( "SIFT2", WINDOW_AUTOSIZE );
        imshow("SIFT2", mImage2);
    }
    else
    {
        namedWindow( "SURF2 ", WINDOW_AUTOSIZE );
        imshow("SURF2", mImage2);
    }

}

void GmsMatch(Mat &img1, Mat &img2){
    vector<KeyPoint> kp1, kp2;
    Mat d1, d2;
    vector<DMatch> matches_all, matches_gms;

    //Ptr<ORB> orb = ORB::create(10000);
    ORB orb=ORB(10000, 1.2, 8, 31, 0, 2, 0, 31);
    orb(img1, Mat(), kp1, d1);
    orb(img2, Mat(), kp2, d2);
    //orb->setFastThreshold(0);
    //orb->detectAndCompute(img1, Mat(), kp1, d1);
    //orb->detectAndCompute(img2, Mat(), kp2, d2);
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(d1, d2, matches_all);
    Mat mImage;
    //cout<<"ORB"<<kp1.size()<<endl;
    /*
    if(matches_all.size()>1)
    {
        drawMatches(img1, kp1, img2, kp2, matches_all, mImage);
        namedWindow( "show ", WINDOW_AUTOSIZE );
        imshow("show", mImage);
    }
    */
    // GMS filter
    int num_inliers = 0;
    std::vector<bool> vbInliers;
    gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
    num_inliers = gms.GetInlierMask(vbInliers, false, false);

    cout << "Get total " << num_inliers << " matches." << endl;

    // draw matches
    for (size_t i = 0; i < vbInliers.size(); ++i)
    {
        if (vbInliers[i] == true)
        {
            matches_gms.push_back(matches_all[i]);
        }
    }
    cout<<"ORB GMS filter matches_gms"<<matches_gms.size()<<endl;

    //Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
    //imshow("show", show);  

    if(matches_gms.size()>1)
    {
        drawMatches(img1, kp1, img2, kp2, matches_gms, mImage);
        namedWindow( "ORB ", WINDOW_AUTOSIZE );
        imshow("ORB", mImage);
    }

}


int main(int argc, char **argv)
{

    ros::init(argc, argv, "Mono");
    ros::start();
    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 Mono path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }
   
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    /* 
    {
        ROS_WARN("Run repair Tracking");   
        Mat cv_image = imread("/home/hmx/LearnVIORB/Data/pro/image/1499835378.488953510.jpg",CV_LOAD_IMAGE_UNCHANGED);
        if(cv_image.empty() )
        {
            //cout<< Times<< endl;
            //ROS_ERROR("Read the picture failed!");
            return 0;
        }
        namedWindow( "test", WINDOW_AUTOSIZE );
        imshow("test",cv_image);
        while(1);
    }
    */
    
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);
    ORB_SLAM2::ConfigParam config(argv[2]);
    sensor_msgs::Image ImageMsg;
    sensor_msgs::Image ImageMsgBuff;
    std::vector<string> vstrImagesTime;

    //0:ADSC
    //1:KITTY  
    //2:EuRoC
    //3:Airborne
    if(config._DatasetSource==0)
        LoadImageTime(config._timefile,vstrImagesTime," ");
    else if(config._DatasetSource==1)
        LoadImageTime(config._timefile,vstrImagesTime,",");
    else if(config._DatasetSource==2)
        LoadImageTime(config._timefile,vstrImagesTime,",");
    else if(config._DatasetSource==3)
        LoadImageTime(config._timefile,vstrImagesTime,",");

    ros::Rate r(800);
    //ROS_WARN("Run not-realtime");

	int number = vstrImagesTime.size();
	cout<< number << endl;   

    Mat img_1,img_2;

/************************************************************************

    LoadOneImagesFromDataset(config._imagefile,vstrImagesTime[config._nphoto1],".png",ImageMsg); //3374
    img_1=(cv_bridge::toCvCopy(ImageMsg))->image.clone();
    LoadOneImagesFromDataset(config._imagefile,vstrImagesTime[config._nphoto2],".png",ImageMsg);
    img_2=(cv_bridge::toCvCopy(ImageMsg))->image.clone();
    //imresize(img_1, 480);
    //imresize(img_2, 480);
    descriptorsAndMatch(img_1,img_2,false);
    GmsMatch(img_1,img_2);
    while(1)
    {
       if(getchar())
         return 0;
    }
///************************************************************************/
    for(int i=config._nphoto1;i<config._nphoto2;i++)
    {

        if(config._DatasetSource==0)
            LoadOneImages(config._imagefile,vstrImagesTime[i],".jpg",ImageMsg);
        else if(config._DatasetSource==1)
        {
            stringstream stringstreamkitti;  
            string kitti_string;
            stringstreamkitti<<setw(6)<<setfill('0')<<i;
            stringstreamkitti>>kitti_string;
            //cout<<kitti_string<<endl;;
            LoadOneImages(config._imagefile,kitti_string,".png",ImageMsg);
        }
        else if(config._DatasetSource==2)
            LoadOneImagesFromDataset(config._imagefile,vstrImagesTime[i],".png",ImageMsg);
        else if(config._DatasetSource==3)
        {
            stringstream stringstreamAirborne;  
            string Airborne_string;
            stringstreamAirborne<<setw(5)<<setfill('0')<<i;
            stringstreamAirborne>>Airborne_string;
            Airborne_string = "cam0_image"+Airborne_string;
            LoadOneImagesFromDataset(config._imagefile,Airborne_string,".bmp",ImageMsg);
            //Mat imgAirborne = cv_bridge::toCvCopy(ImageMsgBuff)->image.clone();
            //cv::namedWindow("test");
            //cv::imshow("test",imgAirborne);
        }

        cv_bridge::CvImagePtr cv_ptr;
        try
        {
        	cv_ptr = cv_bridge::toCvCopy(ImageMsg);//,"rbg8"
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return -1;
        }

        // Consider delay of image message
        cv::Mat im = cv_ptr->image.clone();
        {
            // To test relocalization
            static double startT=-1;
            if(startT<0)
                startT = ImageMsg.header.stamp.toSec();
            // Below to test relocalizaiton
            //if(imageMsg->header.stamp.toSec() > startT+25 && imageMsg->header.stamp.toSec() < startT+25.3)
            if(ImageMsg.header.stamp.toSec()< startT+config._testDiscardTime)
                im = cv::Mat::zeros(im.rows,im.cols,im.type());
        }
        SLAM.TrackMonocular(im, ImageMsg.header.stamp.toSec());  //data processing

        // Wait local mapping end.
        bool bstop = false;
        while(!SLAM.bLocalMapAcceptKF())
        {
            if(!ros::ok())
            {
                bstop=true;
            }
        };
        if(bstop)
            break;
        ros::spinOnce();
        r.sleep();
        if(!ros::ok())
            break;
	}
    while(!SLAM.checkRepairStop())
    {
        ros::spinOnce();
        r.sleep();
    }
    cout<<endl<<endl<<"press any key to shutdown"<<endl;
    getchar();
       
    // Stop all threads
    SLAM.Shutdown();
    cout<<endl<<"SLAM.getKeyframe"<<endl;
    SLAM.getKeyframe();
    ros::shutdown();

    return 0;
}


