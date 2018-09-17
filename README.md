# RepairORB

In recent years, Artificial Intelligence has received extensive attention and has promoted the development of many technical fields, of which the field of computer vision is booming. Monocular SLAM (simultaneous localization and mapping) developed by computer vision can make the robot rely on the camera to locate and navigate without GPS . The cost of monocular SLAM is very low, but there are also many limitations and difficulties. Monocular SLAM is easily disturbed by occlusion, dynamic light sources and object, or the moving-fast camera. In addition, the lack of features would lead to loss of track. After error occurs, the general practice is not to add new map information, but to use the old map information already existing, and try to relocate. However, in many cases, the machine would only pass through where it used to pass after a long time or never. Even if the system can relocate, it would lose a lot of map information and poses. 

Following the approach of ORB-SLAM, now we propose a method to solve this problem. We reopen the system, repair the map, and then make the system to return to normal quickly. The quick relocalization could improve the stability of the monocular SLAM, especially in outdoor.



## main
/src/repairTracking.cc
/src/repairTracking.h


