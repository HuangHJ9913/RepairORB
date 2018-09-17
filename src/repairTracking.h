#ifndef REPAIRTRACKING_H
#define REPAIRTRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class repairTracking
{  

protected:
    std::thread* repairMapping;
    //computer sim3
    g2o::Sim3 mg2oScw;
    vector<MapPoint*> mvpLoopMapPoints;
    vector<MapPoint*> mvpCurrentMatchedPoints;
    bool ComputeSim3(KeyFrame* mpCurrentKF,KeyFrame* MatchedKF,vector<MapPoint*> vvpMapPointMatches);
    void CorrectLoop(KeyFrame* mpCurrentKF,KeyFrame* mpMatchedKF);
    bool mbMonocular;
    bool mbAbortBA;

    KeyFrame* mpCurrentKeyFrame;
    KeyFrame* rOldMapKFcur;
    KeyFrame* rOldMapKFini;

    std::list<MapPoint*> mlpRecentAddedMapPoints;
    std::list<KeyFrame*> mlNewKeyFrames;
    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    void ProcessNewKeyFrame();
    void MapPointCulling();
    void CreateNewMapPoints();
    void SearchInNeighbors();
    cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);
    void KeyFrameCulling();
    void InsertKeyFrame(KeyFrame *pKF);
    bool CheckNewKeyFrames();
    void buildMap();
    

public:
    static unsigned int nRepairStop;
    bool nfirstRepair;
    void static PnPGetRT(Frame & pF1, Frame & pF2, const vector<cv::DMatch>& m_InlierMatches);
    void static matchSURF(Frame & pF1, Frame & pF2, vector<cv::DMatch>& m_InlierMatches);
    void static CreateNewSiftMapPoints(Frame & pF1, Frame & pF2, const vector<cv::DMatch>& m_InlierMatches);
    void static CreateNewSiftMapPoints(KeyFrame* pKF1, KeyFrame* pKF2);


	repairTracking(std::vector<Frame> & rFrames, Map*pMap , Map*rMap, KeyFrameDatabase* rKFDB, LoopClosing* LoopCloser, LocalMapping *mapMapper,
     KeyFrame* rKFcur , KeyFrame* rKFini, Frame & rLastFrame , Frame & riniFrame, const int rMaxFrames,const int sensor);
	// Main tracking function. 
	void Run();
    void InterruptBA();
    
public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;

    // Current Frame
    Frame mCurrentFrame;

	// Input sensor
	int mSensor;

	bool mbOnlyTracking;
    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;


protected:
	void repairTrack();
    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;


    std::vector<Frame> mrepairFrames;
    
    //Map
    Map* mpMap;
    Map* moldMap;

    //BoW
    KeyFrameDatabase* mpKeyFrameDB;
    LoopClosing* mpLoopCloser;
    LocalMapping *mpMapper;


    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    


    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;


    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    Frame miniFrame;

    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    list<MapPoint*> mlpTemporalPoints;
};

} //namespace ORB_SLAM

#endif 
