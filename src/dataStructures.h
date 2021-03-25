#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <mutex>

#include <opencv2/core.hpp>

struct DataFrame
{ // represents the available sensor information at the same time instance

    cv::Mat cameraImg; // camera image

    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors;                 // keypoint descriptors
    std::vector<cv::DMatch> kptMatches;  // keypoint matches between previous and current frame
};

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
#endif /* dataStructures_h */
