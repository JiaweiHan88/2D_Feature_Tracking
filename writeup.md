# **Camera Based 2D Feature Tracking** 


#### The goals / steps of this project are the following:
* Implement Ringbuffer for effective memory management for a series of images
* Implement different methods for Keypoint detection (HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT)
* Remove all keypoints outside of a pre-defined rectangle (Preceeding vehicle)
* Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT
* Implement FLANN matching as well as k-nearest neighbor selection
* Use knn matching to implement the descriptor distance ratio
#### Performance of different combination of detectors and descriptors
* Number of Keypoints on the preceding vehicle and distribution of neighborhood size
* Number of matched keypoints using BF matching and descriptor distance ratio of 0.8
* Time for keypoint detection and descriptor extraction
* Recommend the best choice for our purpose of detecting keypoints on vehicles



[//]: # (Image References)

[image1]: ./writeup_images/keypoints.jpg "nvidia"
[image2]: ./writeup_images/min_neighbor.jpg "center"
[image3]: ./writeup_images/max_neighbor.jpg "recovery_1"
[image4]: ./writeup_images/mean_neighbor.jpg "recovery_2"
[image5]: ./writeup_images/matched.jpg "recovery_3"
[image6]: ./writeup_images/matched_percent.jpg "recovery_4"
[image7]: ./writeup_images/processing.jpg "recovery_5"



### 1. Steps of this project 

---
#### 1.1 Data Buffer Optimization

The implementation can be found in dataStructure.h, a simple ring buffer implementation is presented using two index variable for the header and tail.
On each insert, the header and tail are updated respectively depending on whether buffer is full. A reference to the current head of the buffer is provided including a offset for selection of the second newest data in the buffer.
```sh
template <class T>
class ring_buffer
{
public:
    explicit ring_buffer(size_t size) : m_pbuf(std::unique_ptr<T[]>(new T[size])),
                                        m_maxSize(size)
    {
    }
    void insert(T item)
    {
        m_pbuf[m_head] = item;

        if (m_full)
        {
            m_tail = (m_tail + 1) % m_maxSize;
        }
        m_head = (m_head + 1) % m_maxSize;

        m_full = m_head == m_tail;
    }
    T retrieve()
    {
        if (empty())
        {
            return T();
        }

        //Read data and advance the tail (we now have a free space)
        auto val = m_pbuf[m_tail];
        m_full = false;
        m_tail = (m_tail + 1) % m_maxSize;

        return val;
    }
    //unchecked, assumes user knows which offset can be used
    T *head(size_t offset = 0)
    {
        size_t index = (((m_head - 1 - offset) % m_maxSize + m_maxSize) % m_maxSize);
        return &m_pbuf[index];
    }
    //unchecked, assumes user knows which index can be used
    T *at(size_t index)
    {
        return &m_pbuf[(m_tail + index) % m_maxSize];
    }
    //unchecked, assumes user knows which index can be used
    T &operator[](size_t index)
    {
        return m_pbuf[(m_tail + index) % m_maxSize];
    }
    void reset()
    {
        m_head = m_tail;
        m_full = false;
    }
    bool empty() const
    {
        return m_head == m_tail;
    }
    bool full() const
    {
        return m_full;
    }
    size_t capacity() const
    {
        return m_maxSize;
    }
    size_t size() const
    {
        size_t size = m_maxSize;

        if (!m_full)
        {
            if (m_head >= m_tail)
            {
                size = m_head - m_tail;
            }
            else
            {
                size = m_maxSize + m_head - m_tail;
            }
        }

        return size;
    }

private:
    std::unique_ptr<T[]> m_pbuf;
    size_t m_head = 0;
    size_t m_tail = 0;
    const size_t m_maxSize;
    bool m_full = 0;
};
```
#### 1.2 Implement different methods for Keypoint detection (HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT)
The implementation can be found in  `detKeypointsHarris()` and  `detKeypointsModern()` in matching2D_Student.cpp.

For Harris corner detection we use parameters provided by the class lessons.

For all opencv provided algorithm, we use the default parameters provided by the default constructor.

To run the code with different detectors, we can use `./2D_feature_tracking -d "XXX"` where XXX can be one of  {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"}. No -d parameter will use SHITOMASI by default.

To look at the found keypoints we can enable visualization of the keypoints with `./2D_feature_tracking -d "XXX" -vk`

#### 1.3 Remove all keypoints outside of a pre-defined rectangle (Preceeding vehicle)
The following code is used to remove all keypoints outside of the preceeding vehicle bounding box:

```
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
```
The removal can be prevented using command line "-a", then all keypoints are processed.

#### 1.4 Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT
The implementation can be found in  `descKeypoints()` in matching2D_Student.cpp.

For all opencv provided algorithm, we use the default parameters provided by the default constructor.

To run the code with different descriptors, we can use `./2D_feature_tracking -e "XXX"` where XXX can be one of  {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"}. No -e parameter will use BRISK by default.

To look at the matched keypoints we can enable visualization of the matching with `./2D_feature_tracking -d "XXX" -vm`

#### 1.5 Implement FLANN matching as well as k-nearest neighbor selection & Use knn matching to implement the descriptor distance ratio
The implementation of FLANN and knn can be found in  `matchDescriptors()` in matching2D_Student.cpp. 
```
    ...
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation

            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation

            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        //cout << "FLANN matching" << endl;
    }
```

Code for knn:
```
...
else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        //cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
        //cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
```


### 2. Performance
Detailed data of the performance evaluation can be found in performance.xls
The following combinations are not working:

AKAZE descriptor can only work with AKAZE keypoints, any other detector will not work with AKAZE descriptor.
SIFT keypoints can not be processed by ORB descriptor.

That leaves us 42 - 6 -1 = 35 combinations.
#### 2.1 Number of Keypoints on the preceding vehicle and distribution of neighborhood size from all 10 images
The following data can be retrieved from the code by enable the cout in the code for keypoints data and running `./2D_feature_tracking -auto`

"-auto" will loop through all combination of detector and descriptor using BF matching and knn selection while disabling all visualization.

```
    // cout << "Keypoints data" << keypoints.size() << ' '
    //      << minmaxpair.first->size << ' '
    //      << minmaxpair.second->size << ' '
    //      << mean << ' '
    //      << '\n';
```

As can be seen in the following diagram, BRISK and FAST detected the most keypoints.


 ![alt text][image1]
 
 The highest average neighborhood size can be found for ORB and BRISK.
 
 ![alt text][image2]
  
 ![alt text][image3]
 
 ![alt text][image4]
 
#### 2.2 Number of matched keypoints using BF matching and descriptor distance ratio of 0.8
The following data can be retrieved from the code by enable the cout in the code for matched keypoints data and running "./2D_feature_tracking -auto"
```
    //cout << "matched data " << imgIndex - 1 << ' ' << imgIndex << ' ' << matches.size() << '\n';
```
The highest absolute number of matched keypoints can be found for combinations using the FAST detector, followed by the BRIST detector.
The combination with the highest number of matched keypoints are FAST_BRIEF, FAST_ORB and FAST_SIFT.

 ![alt text][image5]
 
 ![alt text][image6]
  

#### 2.3 Time for keypoint detection and descriptor extraction
The following data can be retrieved from the code by enabling all "cout" lines in matching2D_Student.cpp  and running "./2D_feature_tracking -auto"

From the previous section we already know that detector with FAST, BRISK and AKAZE provide the most keypoints and also matched keypoints.
So we focus on these detectors. From the processing time point of view, we see that FAST, ORB and HARRIS provide the best results. If we assume 30 fps for the camera in the car and real time detection/processing, we need a processing time less then 33 ms.

From the following diagram we find that all AKAZE, SIFT are too slow whereas BRISK is merely fast enough. 

 ![alt text][image7]

#### 2.4 Recommend the best choice for our purpose of detecting keypoints on vehicles
Combined with results from keypoint detection and matching i suggest the combination of
FAST detector and BRIEF descriptor. There are several other candidates with promising speed such as FAST_BRISK and FAST_ORB, since we did not analyze the accuracy, that might be the next step to do in order to decide which combination to use.

