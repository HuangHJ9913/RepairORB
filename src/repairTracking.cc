#include "repairTracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "System.h"
#include <ros/ros.h>
#include "Optimizer.h"
#include "PnPsolver.h"
#include "Sim3Solver.h"

//#include<fstream>
#include<iostream>

#include<mutex>
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "opencv2/highgui/highgui.hpp"  
#include  "opencv2/legacy/legacy.hpp" 
#include  "opencv2/nonfree/nonfree.hpp"
#include "GMS-Feature-Matcher-master/include/Header.h"
#include "GMS-Feature-Matcher-master/include/gms_matcher.h"

using namespace cv;

using namespace std;
//#define REPAIRPNP


namespace ORB_SLAM2
{
unsigned int repairTracking::nRepairStop = 1;
void repairTracking::ProcessNewKeyFrame()
{

    mpCurrentKeyFrame = mlNewKeyFrames.front();
    mlNewKeyFrames.pop_front();

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    //cout<<"vpMapPointMatches.size :"<<vpMapPointMatches.size()<<endl;
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    //cout<<"!pMP->IsInKeyFrame(mpCurrentKeyFrame):"<<endl;
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                    //cout<<"pMP->IsInKeyFrame(mpCurrentKeyFrame):"<<endl;
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void repairTracking::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void repairTracking::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;
    //cout<<"repair CreateNewMapPoints"<<endl;
    int nin=0;
    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));
        //cout<<Tcw1<<endl;
        //cout<<Tcw2<<endl;
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        //cout<<"nmatches"<<nmatches<<endl;
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            //cout<<cosParallaxRays<<endl;
            
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                nin++;
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap,1);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
        //cout<<nmatches<<"nin"<<nin<<endl;
    }
    //cout<<"nnew     "<<nnew<<endl;
}

void repairTracking::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat repairTracking::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

cv::Mat repairTracking::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void repairTracking::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

void repairTracking::InsertKeyFrame(KeyFrame *pKF)
{
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool repairTracking::CheckNewKeyFrames()
{
    return(!mlNewKeyFrames.empty());
}

void repairTracking::InterruptBA()
{
    mbAbortBA = true;
}

void repairTracking::buildMap()
{
    while(CheckNewKeyFrames())
    {
        // BoW conversion and insertion in Map
        ProcessNewKeyFrame();

        // Check recent MapPoints
        MapPointCulling();

        // Triangulate new MapPoints
        CreateNewMapPoints();
        //ROS_WARN("repairTracking as LocalMapping");
        //cout << " Map created" << mpMap->MapPointsInMap() << " points" << endl;
        if(!CheckNewKeyFrames())
        {
            // Find more matches in neighbor keyframes and fuse point duplications
            SearchInNeighbors();
        }

        mbAbortBA = false;

        if(!CheckNewKeyFrames())
        {
            // Local BA
            if(mpMap->KeyFramesInMap()>2)
            {
                //Optimizer::GlobalBundleAdjustemnt(mpMap, 20);
                Optimizer::LocalBundleAdjustment_2(mpCurrentKeyFrame,&mbAbortBA, mpMap); //Optimizer delet point
                //ROS_WARN("LocalBundleAdjustment");
                //cout << "LocalMapping  Map created with " << mpMap->MapPointsInMap() << " points" << endl;
            }
            // Check redundant local Keyframes
            KeyFrameCulling();
        }
        //ROS_WARN("repairTracking as LocalMapping__2");
        //cout << " Map created with " << mpMap->MapPointsInMap() << " points" << endl;
    }
}






void repairTracking::CreateNewSiftMapPoints(Frame& pF1, Frame& pF2, const vector<cv::DMatch>& m_InlierMatches)
{

	cv::Mat Rcw1 = pF1.GetRotation();
	cv::Mat Rwc1 = Rcw1.t();
	cv::Mat tcw1 = pF1.GetTranslation();
	cv::Mat Tcw1(3, 4, CV_32F);
	Rcw1.copyTo(Tcw1.colRange(0, 3));
	tcw1.copyTo(Tcw1.col(3));
	cv::Mat Ow1 = pF1.GetCameraCenter();
    
	const float &fx1 = pF1.fx;
	const float &fy1 = pF1.fy;
	const float &cx1 = pF1.cx;
	const float &cy1 = pF1.cy;
	const float &invfx1 = pF1.invfx;
	const float &invfy1 = pF1.invfy;
    //cout<<invfy1<<endl;
	// Search matches with epipolar restriction and triangulate 
	{
		cv::Mat Ow2 = pF2.GetCameraCenter();
		cv::Mat Rcw2 = pF2.GetRotation();
		cv::Mat Rwc2 = Rcw2.t();
		cv::Mat tcw2 = pF2.GetTranslation();
		cv::Mat Tcw2(3, 4, CV_32F);
		Rcw2.copyTo(Tcw2.colRange(0, 3));
		tcw2.copyTo(Tcw2.col(3));
    //cout<<Tcw1<<endl;
    //cout<<Tcw2<<endl;
		const float &fx2 = pF2.fx;
		const float &fy2 = pF2.fy;
		const float &cx2 = pF2.cx;
		const float &cy2 = pF2.cy;
		const float &invfx2 = pF2.invfx;
		const float &invfy2 = pF2.invfy;

		// Triangulate each match
		const int nmatches = m_InlierMatches.size();

        int nbuild=0;
        int nin=0;
		for (int ikp = 0; ikp<nmatches; ikp++)
		{
            
			const int &idx1 = m_InlierMatches[ikp].queryIdx;
			const int &idx2 = m_InlierMatches[ikp].trainIdx;
            
            //MapPoint* pMP = pF1.sift_mvpMapPoints[idx1];
            //cout<<"pF1.sift_mvpMapPoints"<<pF1.sift_mvpMapPoints.size()<<endl;
			if (pF1.sift_mvpMapPoints[idx1]&&pF2.sift_mvpMapPoints[idx2])
            {

                //cout<<pF1.sift_mvpMapPoints[idx1]->GetWorldPos()<<"kp1.pt.x"<<pF2.sift_mvpMapPoints[idx2]->GetWorldPos()<<endl;
				continue;
            }
            //cout<<"pF1.sift_mvpMapPoints"<<pF1.sift_mvpMapPoints.size()<<endl;
            //cout<<"kp1.pt.x"<<endl;
			const cv::KeyPoint &kp1 = pF1.sift_mvKeysUn[idx1];
			bool bStereo1 = false;

			const cv::KeyPoint &kp2 = pF2.sift_mvKeysUn[idx2];
			bool bStereo2 = false;

            
            //cout<<"kp1.pt.x"<<kp1.pt.x<<endl;
          
			// Check parallax between rays
			cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1)*invfx1, (kp1.pt.y - cy1)*invfy1, 1.0);  ///////
			cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2)*invfx2, (kp2.pt.y - cy2)*invfy2, 1.0);

			cv::Mat ray1 = Rwc1*xn1;
			cv::Mat ray2 = Rwc2*xn2;
			const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

			float cosParallaxStereo = cosParallaxRays + 1;
			float cosParallaxStereo1 = cosParallaxStereo;
			float cosParallaxStereo2 = cosParallaxStereo;

			cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);
        
			cv::Mat x3D;
            //cout<<cosParallaxRays<<endl;
			if (cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<1.0))
			{
				// Linear Triangulation Method
				cv::Mat A(4, 4, CV_32F);
				A.row(0) = xn1.at<float>(0)*Tcw1.row(2) - Tcw1.row(0);
				A.row(1) = xn1.at<float>(1)*Tcw1.row(2) - Tcw1.row(1);
				A.row(2) = xn2.at<float>(0)*Tcw2.row(2) - Tcw2.row(0);
				A.row(3) = xn2.at<float>(1)*Tcw2.row(2) - Tcw2.row(1);

				cv::Mat w, u, vt;
				cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

				x3D = vt.row(3).t();

				if (x3D.at<float>(3) == 0)
					continue;

				// Euclidean coordinates
				x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

			}
			else
            {
                nin++;
				continue;
            }
            


			cv::Mat x3Dt = x3D.t();

			//Check triangulation in front of cameras
			float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
			if (z1 <= 0)
				continue;

			float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
			if (z2 <= 0)
				continue;

			const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
			const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
			const float invz1 = 1.0 / z1;

			if (!bStereo1)
			{
				float u1 = fx1*x1*invz1 + cx1;
				float v1 = fy1*y1*invz1 + cy1;
				float errX1 = u1 - kp1.pt.x;
				float errY1 = v1 - kp1.pt.y;
				if ((errX1*errX1 + errY1*errY1)>10)
					continue;
			}

			//Check reprojection error in second keyframe
			const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
			const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
			const float invz2 = 1.0 / z2;
			if (!bStereo2)
			{
				float u2 = fx2*x2*invz2 + cx2;
				float v2 = fy2*y2*invz2 + cy2;
				float errX2 = u2 - kp2.pt.x;
				float errY2 = v2 - kp2.pt.y;
				if ((errX2*errX2 + errY2*errY2)>10)
					continue;
			}

			//Check scale consistency
			cv::Mat normal1 = x3D - Ow1;
			float dist1 = cv::norm(normal1);

			cv::Mat normal2 = x3D - Ow2;
			float dist2 = cv::norm(normal2);

			if (dist1 == 0 || dist2 == 0)
				continue;
			MapPoint* pMP = new MapPoint(x3D);

			pF1.sift_mvpMapPoints[idx1] = pMP;
			pF2.sift_mvpMapPoints[idx2] = pMP;
            nbuild++;

		}
        //ROS_WARN("nbuild");
        //cout<<nin<<" "<<nbuild<<endl;
	}
}



void repairTracking::matchSURF(Frame& pF1, Frame& pF2, vector<cv::DMatch>& m_InlierMatches)
{
    BFMatcher matcher;
	//cv::FlannBasedMatcher matcher;  // 
	vector< cv::DMatch > matches;
	//std::vector<cv::DMatch> viewMatches;
    //cout<<pF1.sift_mDescriptors.row(1);

	matcher.match(pF1.sift_mDescriptors, pF2.sift_mDescriptors, matches);
    
     /*
        int num_inliers = 0;
        std::vector<bool> vbInliers;
        gms_matcher gms(pF1.sift_mvKeysUn,pF1.imGray.size(), pF2.sift_mvKeysUn,pF2.imGray.size(), matches);
        num_inliers = gms.GetInlierMask(vbInliers, false, false);
        for (size_t i = 0; i < vbInliers.size(); ++i)
        {
            if (vbInliers[i] == true)
            {
                m_InlierMatches.push_back(matches[i]);
            }
        }
        cout<<"sift match "<<m_InlierMatches.size()<<endl;
        
    */
    int ptCount = (int)matches.size();
	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);
	// Keypoint
	Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		pt = pF1.sift_mvKeysUn[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;

		pt = pF2.sift_mvKeysUn[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}

	// RANSAC
	Mat m_Fundamental;
	vector<uchar> m_RANSACStatus;

	m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);//															  

	for (int i = 0; i<ptCount; i++)
	{
		if (m_RANSACStatus[i] != 0)
		{
			m_InlierMatches.push_back(matches[i]);
		}
	}
    //cout<<"sift match "<<m_InlierMatches.size()<<endl;

}

/*
void descriptorsAndMatch(Mat& img_1,Mat& img_2,bool flag)
{
    vector<KeyPoint> keyPoints_1, keyPoints_2;
    Mat descriptors_1, descriptors_2;
    if(flag)
    {
        SIFT sift;
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
    vector< cv::DMatch > matches;
    
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
}
*/


void repairTracking::PnPGetRT(Frame& pF1, Frame& pF2, const vector<cv::DMatch>& m_InlierMatches) //pF2 tranning
{
    int pointmatch=0;
	const int nmatches = m_InlierMatches.size();
	vector<cv::Point2f> mvP2D;
	vector<cv::Point3f> mvP3Dw;

	for (int ikp = 0; ikp < nmatches; ikp++)
	{
		const int &idx1 = m_InlierMatches[ikp].queryIdx;
		const int &idx2 = m_InlierMatches[ikp].trainIdx;
		if (pF2.sift_mvpMapPoints[idx2])
		{
            pointmatch++;
			pF1.sift_mvpMapPoints[idx1] = pF2.sift_mvpMapPoints[idx2];   //put MapPoint in pF1 if matching
			const cv::KeyPoint &kp = pF1.sift_mvKeysUn[idx1];//找特征点，
			mvP2D.push_back(kp.pt); //用特征点去找图像的坐标

			cv::Mat Pos = pF2.sift_mvpMapPoints[idx2]->GetWorldPos();  //获取三维坐标
			mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));
		}		
	}
    //cout<<"pointmatch " <<pointmatch<<endl;
    if(pointmatch>10)
    {
        cv::Mat rvec, tvec, inliers;
        int iterationsCount = 200;      // number of Ransac iterations.
        float reprojectionError = 4.991;  // maximum allowed distance to consider it an inlier.
        double minInliersCount = 100;        // ransac successful confidence.
        cv::solvePnPRansac(mvP3Dw, mvP2D, pF1.mK, pF1.mDistCoef, rvec, tvec, false, iterationsCount, reprojectionError, minInliersCount, inliers);

        cv::Mat R;
        cv::Rodrigues(rvec, R); //罗德里格斯变换

        cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
        R.copyTo(Tcw.rowRange(0,3).colRange(0,3));
        tvec.copyTo(Tcw.rowRange(0,3).col(3));
        pF1.SetPose(Tcw); 
    }
    else
        pF1.SetPose(pF2.mTcw);
    //cout<<Tcw<<endl;
    
}


repairTracking::repairTracking(std::vector<Frame> &rFrames, Map*pMap, Map*rMap, KeyFrameDatabase* rKFDB, LoopClosing* LoopCloser,LocalMapping *mapMapper,KeyFrame* rKFcur, KeyFrame* rKFini,
                    Frame& rLastFrame, Frame& riniFrame, const int rMaxFrames, const int sensor) :
	mrepairFrames(rFrames), moldMap(pMap), mpMap(rMap), mpKeyFrameDB(rKFDB),mpLoopCloser(LoopCloser),mpMapper(mapMapper),mState(OK), mSensor(sensor), mbOnlyTracking(false), 
    mbVO(false), mnLastRelocFrameId(0),mMinFrames(0), mMaxFrames(rMaxFrames), mLastFrame(rLastFrame),mbMonocular(true),miniFrame(riniFrame)//mSensor==System::MONOCULAR
{
	mnLastKeyFrameId = mLastFrame.mnId;
	mpLastKeyFrame = rKFcur;    //
	mpReferenceKF = rKFcur;    //
	mvpLocalKeyFrames.push_back(rKFcur);
	mvpLocalKeyFrames.push_back(rKFini);
    //nfirstRepair = false;
	mvpLocalMapPoints = mpMap->GetAllMapPoints();
    //cout << "repair Map created with " << mpMap->MapPointsInMap() << " points" << endl;
	mpMap->SetReferenceMapPoints(mvpLocalMapPoints); //
	mpMap->mvpKeyFrameOrigins.push_back(rKFini);

    InsertKeyFrame(rKFini);
    InsertKeyFrame(rKFcur);
    

}



void repairTracking::Run()
{
    nRepairStop = 0;
    while(mpLoopCloser->isRunningGBA())
    {
        usleep(1000);
    }
    mrepairFrames.erase(mrepairFrames.end());
	int flag=mrepairFrames.size();
    int mCurrentFrameMnId=mnLastKeyFrameId;
    ROS_WARN("Run repair Tracking");
    cout<<flag<<endl;
    cout << " Map with " << mpMap->MapPointsInMap() << " points" << endl;
    buildMap();
    cout << " Map created with " << mpMap->MapPointsInMap() << " points" << endl;
#ifdef REPAIRPNP
    vector<cv::DMatch> m_InlierMatches;
    mLastFrame.ComputeSURF();
    miniFrame.ComputeSURF();
    cout<<"mLastFrame sift :"<<mLastFrame.sift_mvKeysUn.size()<<endl;
    cout<<"miniFrame sift :"<<miniFrame.sift_mvKeysUn.size()<<endl;
    //cout<<"mLastFrame sift_mvpMapPoints"<<mLastFrame.sift_mvpMapPoints.size()<<endl;
    matchSURF(mLastFrame, miniFrame, m_InlierMatches);
    //cout<<"mLastFrame.mvKeysUn[].pt.x :"<<mLastFrame.mvKeysUn[1].pt.x<<"   mLastFrame.mvKeys[].pt.x :"<<mLastFrame.mvKeys[1].pt.x<<endl;
    cout<<"surf good match "<<m_InlierMatches.size()<<endl;
    fill(mLastFrame.sift_mvpMapPoints.begin(), mLastFrame.sift_mvpMapPoints.end(), static_cast<MapPoint*>(NULL));
    fill(miniFrame.sift_mvpMapPoints.begin(), miniFrame.sift_mvpMapPoints.end(), static_cast<MapPoint*>(NULL));
    CreateNewSiftMapPoints(mLastFrame, miniFrame, m_InlierMatches);
#endif
/*
    Mat OuImage;
    drawMatches(mLastFrame.imGray, mLastFrame.sift_mvKeysUn, miniFrame.imGray, miniFrame.sift_mvKeysUn, m_InlierMatches, OuImage);
    namedWindow( "Match", WINDOW_AUTOSIZE );
    imshow("Match", OuImage);


    vector<cv::DMatch> InlierMatches;
    mCurrentFrame=mrepairFrames.back();
    mCurrentFrame.ComputeSURF();
    matchSURF(mCurrentFrame, mLastFrame, InlierMatches); //get m_InlierMatches
    Mat OutImage;
    drawMatches(mCurrentFrame.imGray, mCurrentFrame.sift_mvKeysUn, mLastFrame.imGray, mLastFrame.sift_mvKeysUn, InlierMatches, OutImage);
    namedWindow( "Match2", WINDOW_AUTOSIZE );
    imshow("Match2", OutImage);
    PnPGetRT(mCurrentFrame, mLastFrame, InlierMatches);
    CreateNewSiftMapPoints(mCurrentFrame, mLastFrame, InlierMatches); //creat more MapPoint
  
    vector<cv::Point2f> mvP2D;
    vector<cv::Point3f> mvP3Dw;

    for (int ikp = 0; ikp < mLastFrame.sift_mvpMapPoints.size(); ikp++)
    {
        MapPoint *mp=mLastFrame.sift_mvpMapPoints[ikp];
        if(mp)
        {        
            const cv::KeyPoint &kp = mLastFrame.sift_mvKeysUn[ikp];//找特征点，
            mvP2D.push_back(kp.pt); //用特征点去找图像的坐标
            cv::Mat Pos = mp->GetWorldPos();  //获取三维坐标
            mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));
        }       
    }
    cv::Mat rvec, tvec, inliers;
    int iterationsCount = 200;      // number of Ransac iterations.
    float reprojectionError = 5.991;  // maximum allowed distance to consider it an inlier.
    double minInliersCount = 100;        // ransac successful confidence.
    cv::solvePnPRansac(mvP3Dw, mvP2D, mLastFrame.mK, mLastFrame.mDistCoef, rvec, tvec, false, iterationsCount, reprojectionError, minInliersCount, inliers);
    cv::Mat R;
    cv::Rodrigues(rvec, R); //罗德里格斯变换
    cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
    R.copyTo(Tcw.rowRange(0,3).colRange(0,3));
    tvec.copyTo(Tcw.rowRange(0,3).col(3));
    cout<<Tcw<<endl; 
    cout<<mLastFrame.mTcw<<endl; 

    //ofstream siftGetWorldPos;
    //siftGetWorldPos.open("/home/hmx/LearnVIORB/Data/pro/repairSIFT_MapPoints.txt",ofstream::binary);
    //for(int i=0;i<mLastFrame.sift_mvpMapPoints.size();i++)
    //{
    //    MapPoint *mp=mLastFrame.sift_mvpMapPoints[i];
    //    if(mp)
    //    {
    //       siftGetWorldPos<<mp->GetWorldPos()<<endl;
    //    }
    //}
    //siftGetWorldPos.close();

   //descriptorsAndMatch(mLastFrame.imGray,  miniFrame.imGray,false);
*/

    /************main*******************/
    cout<<" nbreakKFSN  :  "<<Tracking::nbreakKFSN<<" " <<moldMap->KeyFramesInMap()<<endl;
	for(int i=0;i<flag;i++)
	{
        //normal
        //mCurrentFrame = mrepairFrames.front();
        //mrepairFrames.erase(mrepairFrames.begin());     
		mCurrentFrame = mrepairFrames.back(),
        mCurrentFrame.mnId =i+mCurrentFrameMnId+1;
        cout<<"repair ID :"<<mCurrentFrame.mnId<<" "<<setprecision(20)<<mCurrentFrame.FrameSN<<endl;
		      
		repairTrack();//main
        mrepairFrames.erase(mrepairFrames.end());
        if(mState != OK)
        {
            ROS_WARN("mState != OK");
            break;
        }
	}
    cout<<"Finishing repairTracking"<<endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);
    usleep(3000);
    vector<KeyFrame*> MapGetAllKeyFrames=mpMap->GetAllKeyFrames();   //mpMap  repair
    sort(MapGetAllKeyFrames.begin(),MapGetAllKeyFrames.end(),KeyFrame::up);
    cout<<" "<<MapGetAllKeyFrames[0]->FrameSN<<endl;



    mpMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while(!mpMapper->isStopped())
    {
        usleep(1000);
    }
    //
    /************get old map*******************/
        cout<<"get old map"<<endl;
        vector<KeyFrame*> OldKeyFrames=moldMap->GetAllKeyFrames();  //moldMap
        vector<MapPoint*> OldMapPoints=mpMap->GetAllMapPoints();
        vector<KeyFrame*> OldMapKeyFrames;  //moldMap
        for(size_t i=0;i<OldKeyFrames.size();i++)
        {
            if(OldKeyFrames[i]->FrameSN<=Tracking::nbreakKFSN)
            { 
                OldMapKeyFrames.push_back(OldKeyFrames[i]);
            }
            else if(OldKeyFrames[i]->mnId==Tracking::LocalMappingFixId)
            {
                rOldMapKFini = OldKeyFrames[i];
                //cout<<rOldMapKFini->mnId<<endl;
                //cout<<rOldMapKFini->GetPose()<<endl;
            }
            else if(OldKeyFrames[i]->mnId==Tracking::LocalMappingFixId+1)
            {
                rOldMapKFcur = OldKeyFrames[i];
                //cout<<rOldMapKFcur->mnId<<endl;
                //cout<<rOldMapKFcur->GetPose()<<endl;
            }
        }
        sort(OldMapKeyFrames.begin(),OldMapKeyFrames.end(),KeyFrame::up);//  
    /************get old map*******************/

    /**********************************add connection***************************************/
    {  
        unique_lock<mutex> lock(moldMap->mMutexMapUpdate);
        cout<<"change connection"<<endl;
        {
            //delete first and second KF;
            vector<KeyFrame*> CurrepairConnectedKFs =MapGetAllKeyFrames[0]->GetVectorCovisibleKeyFrames();
            vector<KeyFrame*> InirepairConnectedKFs =MapGetAllKeyFrames[1]->GetVectorCovisibleKeyFrames();
            MapGetAllKeyFrames[0]->SetBadFlag();        
            //cout<<"MapGetAllKeyFrames[1]->GetChilds()"<<MapGetAllKeyFrames[1]->GetChilds().size()<<endl; //ini only have one child.
            for(vector<KeyFrame*>::iterator mit = InirepairConnectedKFs.begin(), mend=InirepairConnectedKFs.end(); mit!=mend; mit++)
                (*mit)->EraseConnection(MapGetAllKeyFrames[1]);
            
            set<MapPoint*>mvpMapPoint = MapGetAllKeyFrames[1]->GetMapPoints();
            for(set<MapPoint*>::iterator mit = mvpMapPoint.begin(), mend=mvpMapPoint.end(); mit!=mend; mit++)
                if(*mit)
                    (*mit)->EraseObservation(MapGetAllKeyFrames[1]);

            set<KeyFrame*> ChildKF=MapGetAllKeyFrames[1]->GetChilds();
            for(set<KeyFrame*>::iterator mit = ChildKF.begin(), mend=ChildKF.end(); mit!=mend; mit++)
                    (*mit)->ChangeParent(rOldMapKFini);

            if(MapGetAllKeyFrames[1]->GetParent())
                MapGetAllKeyFrames[1]->GetParent()->EraseChild(MapGetAllKeyFrames[1]);
            MapGetAllKeyFrames[1]->setBad();
         
        }
        vector<KeyFrame*> OldMapConnect;
        OldMapConnect.push_back(rOldMapKFcur);
        OldMapConnect.push_back(rOldMapKFini);
        cout<<"add connection"<<endl;
        for(int i=0;i<2;i++)
        {
            if(MapGetAllKeyFrames[i]->FrameSN==OldMapConnect[i]->FrameSN)
            {
                //cout<<"OldMapConnect[i]->GetVectorCovisibleKeyFrames()"<<OldMapConnect[i]->GetVectorCovisibleKeyFrames().size()<<endl;
                int nkeypoint = MapGetAllKeyFrames[i]->N;
                //cout<<"check if the same keyframe "<<MapGetAllKeyFrames[0]->N<<" "<<OldMapConnect[i]->N;
                for(int sit=0; sit<nkeypoint; sit++)
                {
                    MapPoint* pMP1 = MapGetAllKeyFrames[i]->GetMapPoint(sit);
                    MapPoint* pMP2 = OldMapConnect[i]->GetMapPoint(sit);

                    if(pMP1)
                    {
                        if(pMP2)
                        {
                            if(!pMP2->isBad())
                            {
                                pMP1->Replace(pMP2);   
                            }                      
                        }
                        else
                        {
                            if(!pMP1->isBad())
                            {
                                OldMapConnect[i]->AddMapPoint(pMP1,sit);
                                //mpMap->AddMapPoint(pMP1); //??????
                            }
                        }
                    }
                }   
            /*
                vector<KeyFrame*> ConnectedKFs =MapGetAllKeyFrames[i]->GetVectorCovisibleKeyFrames();
                for(vector<KeyFrame*>::iterator mit = ConnectedKFs.begin(), mend=ConnectedKFs.end(); mit!=mend; mit++)
                    (*mit)->UpdateConnections();
       
                vector<KeyFrame*> vpConnectedKFs =OldMapConnect[i]->GetVectorCovisibleKeyFrames();
                for(int sit=0;sit<vpConnectedKFs.size();sit++)
                {
                    int mTs =0;
                    if(vpConnectedKFs[sit]->GetParent())
                        mTs =vpConnectedKFs[sit]->GetParent()->FrameSN;
                    cout<<vpConnectedKFs[sit]->FrameSN<<"Parent("<<mTs<<") ";
                }
                cout<<endl;
                OldMapConnect[i]->UpdateConnections();
                vector<KeyFrame*> vpConnectedKFafter =OldMapConnect[i]->GetVectorCovisibleKeyFrames();
                for(int sit=0;sit<vpConnectedKFafter.size();sit++)
                {
                    int mTs =0;
                    if(vpConnectedKFafter[sit]->GetParent())
                        mTs =vpConnectedKFafter[sit]->GetParent()->FrameSN;
                    cout<<vpConnectedKFafter[sit]->FrameSN<<"Parent("<<mTs<<") ";
                }
                cout<<endl;
                cout<<"OldMapConnect[i]->GetVectorCovisibleKeyFrames()"<<OldMapConnect[i]->GetVectorCovisibleKeyFrames().size()<<endl;
            */    
            }
        }
    }
    /**********************************add connection***************************************/

    /**********************************put into oldmap***************************************/
    {
        unique_lock<mutex> lock(moldMap->mMutexMapUpdate);
        cout<<"put into oldmap"<<endl;
        for(size_t i=2;i<MapGetAllKeyFrames.size();i++)  //keyFrame
        {
            if(MapGetAllKeyFrames[i]->isBad())
                continue;
            MapGetAllKeyFrames[i]->mnId = Tracking::LocalMappingFixId-1-i;
            MapGetAllKeyFrames[i]->UpdateConnections();
            MapGetAllKeyFrames[i]->changeMap(moldMap);
            moldMap->AddKeyFrame(MapGetAllKeyFrames[i]);
            mpKeyFrameDB->add(MapGetAllKeyFrames[i]);
        }
        vector<MapPoint*> ChangeMapPoints=mpMap->GetAllMapPoints();  //Mappoint
        for(size_t i=0;i<ChangeMapPoints.size();i++)
        {
            if(ChangeMapPoints[i]->isBad())
                ChangeMapPoints[i]->SetBadFlag();
            else
            {
                ChangeMapPoints[i]->changeMap(moldMap);
                moldMap->AddMapPoint(ChangeMapPoints[i]);
            }
        }
        for(int i=0;i<OldKeyFrames.size();i++)
        {
            if(OldKeyFrames[i]->isBad())
                OldKeyFrames[i]->SetBadFlag();
            else
                OldKeyFrames[i]->UpdateConnections();
        }
        for(int i=0;i<OldKeyFrames.size();i++)
        {
            if(OldKeyFrames[i]->isBad())
                OldKeyFrames[i]->SetBadFlag();
            else
                OldKeyFrames[i]->UpdateConnections();
        }
        /*for(int i=0;i<OldMapPoints.size();i++)
        {
            if(OldMapPoints[i])
                if(OldMapPoints[i]->isBad())
                    OldMapPoints[i]->SetBadFlag();
        }*/
        
    }
    //mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
    /**********************************put into oldmap***************************************/

    /**********************************computer sim3***************************************/
    cout<<"detect loop"<<endl;
    std::vector<KeyFrame* > matchKFvector;
    std::vector<KeyFrame* > currentKFvector;
    vector<vector<MapPoint*> > vvpMapPointMatches;
    int KFflag;
    for(size_t i=2;i<MapGetAllKeyFrames.size();i++)  //detect loop
    {
        if(MapGetAllKeyFrames[i]->FrameSN>OldMapKeyFrames[0]->FrameSN+6)
            continue;
        for(size_t j=0;j<OldMapKeyFrames.size();j++)  //-+5
        {
            if(MapGetAllKeyFrames[i]->FrameSN<OldMapKeyFrames[j]->FrameSN-6)
                continue;
            if(MapGetAllKeyFrames[i]->FrameSN>OldMapKeyFrames[j]->FrameSN+6)
                break;
            ORBmatcher matcher(0.7,true);
            vector<MapPoint*> KF1PointMatches;
            int nmatches = matcher.SearchByBoW(OldMapKeyFrames[j],MapGetAllKeyFrames[i],KF1PointMatches);
            if(nmatches<66)
                continue;
            currentKFvector.push_back(OldMapKeyFrames[j]);
            matchKFvector.push_back(MapGetAllKeyFrames[i]);
            vvpMapPointMatches.push_back(KF1PointMatches);
            cout<<" "<<OldMapKeyFrames[j]->FrameSN<<" "<<MapGetAllKeyFrames[i]->FrameSN<<" "<<nmatches<<endl;      
        }
    }
    cout<<"computer sim3 :"<<matchKFvector.size()<<endl;
    for(size_t i=0;i<currentKFvector.size();i++)
    {
        if(ComputeSim3(currentKFvector[i],matchKFvector[i],vvpMapPointMatches[i]))
        {

            cout<<"computer sim3 success ";
            cout<<" "<<currentKFvector[i]->FrameSN<<" "<<matchKFvector[i]->FrameSN<<" "<<vvpMapPointMatches[i].size()<<endl;
            cout<<"start correct loop "<<endl;
            while(mpLoopCloser->isRunningGBA())
            {
                usleep(1000);
            }
            CorrectLoop(currentKFvector[i],matchKFvector[i]);
            //matchKFvector[i]->SetNotErase();
            cout<<"correct loop Finishing"<<endl;
            KFflag = matchKFvector[i]->FrameSN;
            break;
        }
    }
    /**********************************computer sim3***************************************/  

    //Optimizer::GlobalBundleAdjustemnt(moldMap,20);
    mpMapper->Release(); 



    /*************************************text******************************************
    {
        ofstream repairR_and_T;
        repairR_and_T.open("/home/hmx/LearnVIORB/Data/repair_RT.txt",ofstream::binary);
        for(size_t i=0;i<MapGetAllKeyFrames.size();i++)
        {
            if(MapGetAllKeyFrames[i]->isBad())
                continue;
            repairR_and_T<<setprecision(18)<<MapGetAllKeyFrames[i]->mTimeStamp<<" "<<MapGetAllKeyFrames[i]->mnId<<" "<<MapGetAllKeyFrames[i]->FrameSN<<endl;
            repairR_and_T<<MapGetAllKeyFrames[i]->GetPose().t()<<endl;
            //cout<<setprecision(20)<<MapGetAllKeyFrames[i]->mTimeStamp<<endl;
            //cout<<MapGetAllKeyFrames[i]->GetPose()<<endl;
        }
        repairR_and_T.close();

        vector<MapPoint*> MapGetAllMapPoints=mpMap->GetAllMapPoints();
        ofstream repairGetWorldPos;
        repairGetWorldPos.open("/home/hmx/LearnVIORB/Data/repair_MapPoints.txt",ofstream::binary);
        for(size_t i=0;i<MapGetAllMapPoints.size();i++)
        {
            if(MapGetAllMapPoints[i]->isBad())
                continue;
            repairGetWorldPos<<MapGetAllMapPoints[i]->GetWorldPos()<<endl;
            //cout<<setprecision(20)<<MapGetAllKeyFrames[i]->mTimeStamp<<endl;
            //cout<<MapGetAllKeyFrames[i]->GetPose()<<endl;
        }
        repairGetWorldPos.close();
    }
    /*************************************text******************************************/
    
    KeyFrame::nRepairId = 0;  
    nRepairStop = 1;
    cout << "Reseting Database...";
    cout << " done" << endl;
    cout << " Map created with " << mpMap->MapPointsInMap() << " points" << endl;
    cout << " Map created with " << mpMap->KeyFramesInMap() << " KeyFrames" << endl;
    //mpMap->clear();
    //mrepairFrames.clear();
	ROS_WARN("repair Tracking stop");

}


bool repairTracking::ComputeSim3(KeyFrame* mpCurrentKF,KeyFrame* MatchedKF,vector<MapPoint*> vvpMapPointMatches)
{

    ORBmatcher matcher(0.75,true);
    bool bMatch = false;
    bool mbFixScale =false;
    cv::Mat mScw;

    vector<bool> vbInliers;
    int nInliers;
    bool bNoMore;
    Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,MatchedKF,vvpMapPointMatches,mbFixScale);
    pSolver->SetRansacParameters(0.99,20,300);
    cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);
    //cout<<"start ComputeSim3"<<endl;
    // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
    if(!Scm.empty())
    {
        //cout<<"!Scm.empty()"<<endl;
        mvpCurrentMatchedPoints.resize(vvpMapPointMatches.size());
        for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
        {
            if(vbInliers[j])
               mvpCurrentMatchedPoints[j]=vvpMapPointMatches[j];
        }
        cout<<"vbInliers.size()"<<vbInliers.size()<<endl;
        cv::Mat R = pSolver->GetEstimatedRotation();
        cv::Mat t = pSolver->GetEstimatedTranslation();
        const float s = pSolver->GetEstimatedScale();
        matcher.SearchBySim3(mpCurrentKF,MatchedKF,mvpCurrentMatchedPoints,s,R,t,7.5);

        g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
        const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, MatchedKF, mvpCurrentMatchedPoints, gScm, 10, mbFixScale);

        // If optimization is succesful stop ransacs and continue
        if(nInliers>=20)
        {
            bMatch = true;
            g2o::Sim3 gSmw(Converter::toMatrix3d(MatchedKF->GetRotation()),Converter::toVector3d(MatchedKF->GetTranslation()),1.0);
            mg2oScw = gScm*gSmw;
            mScw = Converter::toCvMat(mg2oScw);
        }
    }

    if(!bMatch)
    {
        return false;
    }
    //cout<<"bMatch"<<endl;
    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<KeyFrame*> vpLoopConnectedKFs = MatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(MatchedKF);
    
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)
    {
        return true;
    }
    else
    {
        return false;
    }

}


void repairTracking::CorrectLoop(KeyFrame* mpCurrentKF,KeyFrame* mpMatchedKF) 
{
    cout << "repair Loop detected!" << mpCurrentKF->mTimeStamp<<endl;
    mpMapper->RequestStop();
    // Wait until Local Mapping has effectively stopped
    while(!mpMapper->isStopped())
    {
        usleep(1000);
    }

    /*
    {
        
        // Send a stop signal to Local Mapping
        // Avoid new keyframes are inserted while correcting the loop
        mpMapper->RequestStop();

        // If a Global Bundle Adjustment is running, abort it
        if(isRunningGBA())
        {
            unique_lock<mutex> lock(mMutexGBA);
            mbStopGBA = true;

            mnFullBAIdx++;

            if(mpThreadGBA)
            {
                mpThreadGBA->detach();
                delete mpThreadGBA;
            }
        }

        // Wait until Local Mapping has effectively stopped
        while(!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }
        // Ensure current keyframe is updated
        vector<KeyFrame*> vpConnectedKFs =mpCurrentKF->GetVectorCovisibleKeyFrames();
        for(int sit=0;sit<vpConnectedKFs.size();sit++)
        {
            int mTs =0;
            if(vpConnectedKFs[sit]->GetParent())
                mTs =vpConnectedKFs[sit]->GetParent()->FrameSN;
            cout<<vpConnectedKFs[sit]->FrameSN<<"Parent("<<mTs<<") ";
        }
        mpCurrentKF->UpdateConnections();
        vpConnectedKFs =mpCurrentKF->GetVectorCovisibleKeyFrames();
        for(int sit=0;sit<vpConnectedKFs.size();sit++)
        {
            int mTs =0;
            if(vpConnectedKFs[sit]->GetParent())
                mTs =vpConnectedKFs[sit]->GetParent()->FrameSN;
            cout<<vpConnectedKFs[sit]->FrameSN<<"Parent("<<mTs<<") ";
        }
        
    }
    */
    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    std::vector<KeyFrame*> mvpCurrentConnectedKFs;
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    LoopClosing::KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();

    {
        // Get Map Mutex
        unique_lock<mutex> lock(moldMap->mMutexMapUpdate);
        // get CorrectedSim3 and NonCorrectedSim3
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;

            cv::Mat Tiw = pKFi->GetPose();

            if(pKFi!=mpCurrentKF)
            {
                cv::Mat Tic = Tiw*Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(LoopClosing::KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                if(pCurMP)
                    pCurMP->Replace(pLoopMP);
                else
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
                    pLoopMP->AddObservation(mpCurrentKF,i);
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }
    }
    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //SearchAndFuse(CorrectedSim3);
    //void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
    {
        ORBmatcher matcherFuse(0.8);

        for(LoopClosing::KeyFrameAndPose::const_iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend;mit++)
        {
            KeyFrame* pKF = mit->first;

            g2o::Sim3 g2oScw = mit->second;
            cv::Mat cvScw = Converter::toCvMat(g2oScw);

            vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
            matcherFuse.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

            // Get Map Mutex
            unique_lock<mutex> lock(moldMap->mMutexMapUpdate);
            const int nLP = mvpLoopMapPoints.size();
            for(int i=0; i<nLP;i++)
            {
                MapPoint* pRep = vpReplacePoints[i];
                if(pRep)
                {
                    pRep->Replace(mvpLoopMapPoints[i]);
                }
            }
        }
    }

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    Optimizer::OptimizeEssentialGraph(moldMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, false);

    moldMap->InformNewBigChange();
    
    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);
    mpMapper->Release(); 
    /*
    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    
    mLastLoopKFid = mpCurrentKF->mnId;  
    */
}

/*
void repairTracking::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            cout<<"Correct keyframes starting at map first keyframe"<<endl;
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                //if(pKF->isBad())
                    //continue;
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    //if()
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            cout<<"Correct MapPoints"<<endl;
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();
            ofstream logCorrectMapPoints;
            logCorrectMapPoints.open("/home/hmx/LearnVIORB/Data/logCorrectMapPoints.txt",ofstream::binary);
            logCorrectMapPoints<<"Correct MapPoints"<<endl;
            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];
                if(!pMP)
                    continue;
                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    logCorrectMapPoints<<pMP->mPosGBA<<endl;
                    pMP->SetWorldPos(pMP->mPosGBA);
                    logCorrectMapPoints<<pMP->mnId<<endl;
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    //if(pRefKF->isBad())
                        //continue;
                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    logCorrectMapPoints<<Rcw<<endl;
                    logCorrectMapPoints<<tcw<<endl;
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
            logCorrectMapPoints<< "Map updated!" << endl;
            logCorrectMapPoints.close();
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}
*/
void repairTracking::repairTrack()
{

    // Get Map Mutex -> Map cannot be changed
    //unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
    // System is initialized. Track Frame.
    bool bOK;
    // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
    // Localization Mode: Local Mapping is deactivated
	if (mState == OK)
	{
		// Local Mapping might have changed some MapPoints tracked in last frame
		CheckReplacedInLastFrame();
		
		if (mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId + 2)
		{
                cout<<"repair Track ReferenceKey Frame :"<<mCurrentFrame.mnId<<endl;
            #ifdef REPAIRPNP
                vector<cv::DMatch> m_InlierMatches;
                mCurrentFrame.ComputeSURF();
                matchSURF(mCurrentFrame, mLastFrame, m_InlierMatches); //get m_InlierMatches
                PnPGetRT(mCurrentFrame, mLastFrame, m_InlierMatches);
            #endif    
    			bOK = TrackReferenceKeyFrame();
                //cout<<mCurrentFrame.mTcw<<endl;
            #ifdef REPAIRPNP
                CreateNewSiftMapPoints(mCurrentFrame, mLastFrame, m_InlierMatches); //creat more MapPoint
            #endif

		}
		else
		{
            #ifdef REPAIRPNP
                vector<cv::DMatch> m_InlierMatches;
                mCurrentFrame.ComputeSURF();
                matchSURF(mCurrentFrame, mLastFrame, m_InlierMatches); //get m_InlierMatches
                PnPGetRT(mCurrentFrame, mLastFrame, m_InlierMatches);
            #endif
    			bOK = TrackWithMotionModel();
            #ifdef REPAIRPNP
                CreateNewSiftMapPoints(mCurrentFrame, mLastFrame, m_InlierMatches);
            #endif
    			if (!bOK)
                {
                    cout<<"repair Track With ReferenceKey after Motion Model :"<<mCurrentFrame.mnId<<endl;
    				bOK = TrackReferenceKeyFrame();
                }
		}
		
	}
	else
	{
        cout<<"Relocalization :"<<mCurrentFrame.mnId<<endl;
		bOK = Relocalization();
	}

    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    if(bOK)
        bOK = TrackLocalMap();
    if(bOK)
        mState = OK;
    else
        mState=LOST;
    // If tracking were good, check if we insert a keyframe
    if(bOK)
    {
        // Update motion model
        if(!mLastFrame.mTcw.empty())
        {
            cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
            mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
            mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
            mVelocity = mCurrentFrame.mTcw*LastTwc;
        }
        else
            mVelocity = cv::Mat();

        // Clean VO matches
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
                if(pMP->Observations()<1)
                {
                    mCurrentFrame.mvbOutlier[i] = false;
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                }
        }

        // Delete temporal MapPoints
        for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            delete pMP;
        }
        mlpTemporalPoints.clear();

        // Check if we need to insert a new keyframe
        if(NeedNewKeyFrame())
        {
            //cout<<"repair Need New KeyFrame "<<endl;
            CreateNewKeyFrame();
        }

        // We allow points with high innovation (considererd outliers by the Huber Function)
        // pass to the new keyframe, so that bundle adjustment will finally decide
        // if they are outliers or not. We don't want next frame to estimate its position
        // with those points so we discard them in the frame.
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
        }
    }

    // Reset if the camera get lost soon after initialization
    if(mState==LOST)
    {
        if(mpMap->KeyFramesInMap()<=5)
        {
            cout << "repair Track lost soon after initialisation, reseting..." << endl;
            return;
        }
    }

    if(!mCurrentFrame.mpReferenceKF)
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

    mLastFrame = Frame(mCurrentFrame);

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}


void repairTracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool repairTracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void repairTracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr*pRef->GetPose());

}

bool repairTracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();
    #ifndef REPAIRPNP
        mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
    #endif

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}


bool repairTracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }
    //cout<<mvflag<<endl;
    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    //cout<<"mnMatchesInliers :"<<mnMatchesInliers<<endl;
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
    {
        //cout<<"mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50"<<endl;
        //cout<<"mCurrentFrame.mnId :"<<mCurrentFrame.mnId<<endl;
        //cout<<"mnLastRelocFrameId+mMaxFrames :"<<mnLastRelocFrameId+mMaxFrames<<endl;
       // cout<<"mnMatchesInliers :"<<mnMatchesInliers<<endl;
        return false;
    }

    if(mnMatchesInliers<30)
    {
        cout<<"mnMatchesInliers<30"<<endl;
        return false;
    }
    else
        return true;
}


bool repairTracking::NeedNewKeyFrame()
{

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);


    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {

        return true;

    }
    else
        return false;
}


void repairTracking::CreateNewKeyFrame()
{

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB,1);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    InsertKeyFrame(pKF);
    buildMap();
    /*
    if(repairMapping)
    {
        repairMapping->detach();
        delete repairMapping;
    }
    repairMapping=new thread(&repairTracking::buildMap,this); 
    */
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void repairTracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void repairTracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void repairTracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}



void repairTracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool repairTracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

} //namespace ORB_SLAM
