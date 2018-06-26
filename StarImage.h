//
// Created by 许舰 on 2018/1/10.
//

#ifndef IMAGEREGISTRATION_STARIMAGE_H
#define IMAGEREGISTRATION_STARIMAGE_H

#endif //IMAGEREGISTRATION_STARIMAGE_H

#include <iostream>
#include <opencv/cv.hpp>

#include "StarImagePart.h"

using namespace cv;
using namespace std;

class StarImage
{
public:

private:
    Mat_<Vec3b> image;
    int rowParts;
    int columnParts;  // 整幅图像会被分成　rowParts * column 块

    std::vector<vector<StarImagePart>> starImageParts;

public:
    StarImage();

    StarImage(Mat image, int rowParts, int columnParts, bool isClone = false);

    void splitImage(bool isClone = false);

    StarImagePart& getStarImagePart(int rowPartIndex, int columnPartIndex);

    void setStarImagePart(int rowPartIndex, int columnPartIndex, Mat_<Vec3b> imageMat);

    void addUpStarImagePart(int rowPartIndex, int columnPartIndex, Mat_<Vec3b> imageMat);

    Mat mergeStarImageParts();

    Mat getImage();

private:

    void initStarImageParts();

};