echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../g2o

echo ""
echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

cd ../../../

echo ""
echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo ""
echo "Configuring and building ORB_SLAM2 ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..

echo ""
echo "Build ROS node ..."

cd Examples/ROS/ORB_Repair
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
cd ../../../../

echo ""
echo "Launch file in Examples/ROS/ORB_Repair/launch."
echo "Modify the configuration file config/***.yaml"
echo "Run as: roslaunch ORB_Repair testRepair.launch"
echo ""

#echo "Converting vocabulary to binary"
#./tools/bin_vocabulary
