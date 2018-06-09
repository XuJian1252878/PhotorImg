//
// Created by 许舰 on 2018/1/10.
//

#ifndef IMAGEREGISTRATION_STARIMAGEREGISTBUILDER_H
#define IMAGEREGISTRATION_STARIMAGEREGISTBUILDER_H

#endif //IMAGEREGISTRATION_STARIMAGEREGISTBUILDER_H

#include <iostream>
#include <opencv/cv.hpp>


#include "StarImage.h"
#include "opencv2/xfeatures2d.hpp"
#include <map>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

class StarImageRegistBuilder
{
public:
    static const int MERGE_MODE_MEAN = 1;
    static const int MERGE_MODE_MEDIAN = 2;

private:
    StarImage targetStarImage; // 用于作为配准基准的图像信息
    std::vector<StarImage> sourceStarImages; // 用于配准的图像信息
    int rowParts;
    int columnParts;  // 期望图像将被划分为 rowParts * columnParts 部分
    int imageCount;

public:

    StarImageRegistBuilder(Mat_<Vec3b> targetImage, std::vector<Mat_<Vec3b>> sourceImages, int rowParts, int columnParts);

    void addSourceImagePath(string imgPath);

    void setTargetImagePath(string imgPath);

    Mat_<Vec3b> registration(int mergeMode);

private:

    Mat getImgTransform(StarImagePart sourceImagePart, StarImagePart targetImagePart);

    Mat mergeImage(int mergeMode);
};
