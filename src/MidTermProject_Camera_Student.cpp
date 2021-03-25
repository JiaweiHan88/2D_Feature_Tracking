/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

class InputParser
{
public:
    InputParser(int &argc, const char **argv)
    {
        for (int i = 1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }
    /// @author iain
    const std::string &getCmdOption(const std::string &option) const
    {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end())
        {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }
    /// @author iain
    bool cmdOptionExists(const std::string &option) const
    {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

private:
    std::vector<std::string> tokens;
};

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    bool bVisKey = false, bVisMatch = false;
    // only keep keypoints on the preceding vehicle
    bool bFocusOnVehicle = true;
    // optional : limit number of keypoints (helpful for debugging and learning)
    bool bLimitKpts = false;
    int maxKeypoints = 50;
    std::vector<std::string> detectorarray = {"SHITOMASI"};
    std::vector<std::string> descriptorarray = {"BRISK"};
    string matcherType = "MAT_BF";   // MAT_BF, MAT_FLANN
    string selectorType = "SEL_NN";  // SEL_NN, SEL_KNN
    InputParser input(argc, argv);
    if (input.cmdOptionExists("-vk"))
    {
        bVisKey = true;
    }
    if (input.cmdOptionExists("-vm"))
    {
        bVisMatch = true;
    }
    if (input.cmdOptionExists("-v"))
    {
        bVisKey = true;
        bVisMatch = true;
    }
    if (input.cmdOptionExists("-a"))
    {
        bFocusOnVehicle = false;
    }
    const std::string &limit = input.getCmdOption("-l");
    if (!limit.empty())
    {
        bLimitKpts = true;
        maxKeypoints = stoi(limit);
    }
    const std::string &detectorTypeParsed = input.getCmdOption("-d");
    if (!detectorTypeParsed.empty())
    {
        detectorarray[0] = detectorTypeParsed;
    }
    const std::string &extractorTypeParsed = input.getCmdOption("-e");
    if (!extractorTypeParsed.empty())
    {
        descriptorarray[0] = extractorTypeParsed;
    }
    if (input.cmdOptionExists("-knn"))
    {
        selectorType = "SEL_KNN";
    }
    if (input.cmdOptionExists("-flann"))
    {
        matcherType = "MAT_FLANN";
    }
    if(input.cmdOptionExists("-auto"))
    {
        bVisKey = false;
        bVisMatch = false;
        selectorType = "SEL_KNN";
        detectorarray = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
        descriptorarray = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
    }
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)
    // misc
    int dataBufferSize = 2; // no. of images which are held in memory (ring buffer) at the same time

    /* MAIN LOOP OVER ALL IMAGES */
    for (auto detectorType : detectorarray)
    {
        for (auto descriptorType : descriptorarray)
        {
            if ((descriptorType == "AKAZE" && detectorType != "AKAZE") ||  (descriptorType == "ORB"   && detectorType == "SIFT"))
            {

                continue;
            }
            ring_buffer<DataFrame> dataRingBuffer(dataBufferSize); // list of data frames which are held in memory at the same time

            cout << detectorType << "---" << descriptorType << endl;
            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                dataRingBuffer.insert(frame);

                //// EOF STUDENT ASSIGNMENT
                //cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, bVisKey);
                }
                else if (detectorType.compare("HARRIS") == 0)
                {
                    detKeypointsHarris(keypoints, imgGray, bVisKey);
                }
                else
                {
                    detKeypointsModern(keypoints, imgGray, detectorType, bVisKey);
                }
                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                cv::Rect vehicleRect(535, 180, 180, 150);
                if (bFocusOnVehicle)
                {
                    //cout << "Keypoint size before filtering" << keypoints.size() << endl;
                    keypoints.erase(std::remove_if(keypoints.begin(),
                                                   keypoints.end(),
                                                   [vehicleRect](cv::KeyPoint k) { return !vehicleRect.contains(k.pt); }),
                                    keypoints.end());
                    //cout << "Keypoint size after filtering" << keypoints.size() << endl;
                }

                //// EOF STUDENT ASSIGNMENT
                auto mean = std::accumulate(keypoints.begin(), keypoints.end(), 0.0,
                                            [](const size_t sum, const cv::KeyPoint &kp2) { return sum + kp2.size; }) /
                            keypoints.size();

                auto minmaxpair = std::minmax_element(keypoints.begin(), keypoints.end(),
                                                      [](const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) { return kp1.size < kp2.size; });

                // cout << "Keypoints data" << keypoints.size() << ' '
                //      << minmaxpair.first->size << ' '
                //      << minmaxpair.second->size << ' '
                //      << mean << ' '
                //      << '\n';

                if (bLimitKpts)
                {
                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    //cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                dataRingBuffer.head()->keypoints = keypoints;
                //cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;
                descKeypoints(dataRingBuffer.head()->keypoints, dataRingBuffer.head()->cameraImg, descriptors, descriptorType);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                dataRingBuffer.head()->descriptors = descriptors;

                //cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (dataRingBuffer.size() > 1) // wait until at least two images have been processed
                {

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors(dataRingBuffer.head(1)->keypoints, dataRingBuffer.head()->keypoints,
                                     dataRingBuffer.head(1)->descriptors, dataRingBuffer.head()->descriptors,
                                     matches, descriptorType, matcherType, selectorType);

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    dataRingBuffer.head()->kptMatches = matches;
                    //cout << "matched data " << imgIndex - 1 << ' ' << imgIndex << ' ' << matches.size() << '\n';
                    //cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    if (bVisMatch)
                    {
                        cv::Mat matchImg = (dataRingBuffer.head()->cameraImg).clone();
                        cv::drawMatches(dataRingBuffer.head(1)->cameraImg, dataRingBuffer.head(1)->keypoints,
                                        dataRingBuffer.head()->cameraImg, dataRingBuffer.head()->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        //cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                }
            } // eof loop over all images
        }
    }
    return 0;
}