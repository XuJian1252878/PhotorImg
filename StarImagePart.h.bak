//
// Created by 许舰 on 2018/1/10.
//

#ifndef IMAGEREGISTRATION_STARIMAGEPART_H
#define IMAGEREGISTRATION_STARIMAGEPART_H

#endif //IMAGEREGISTRATION_STARIMAGEPART_H

#include <opencv/cv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

class StarImagePart
{
public:

private:
    Mat_<Vec3b> imagePart;
    int rowPartIndex; // 是父图像中的行位置的第几部分
    int columnPartIndex;  // 是父图像中列位置的第几部分
    int atParentStartRowIndex;
    int atParentEndRowIndex;
    int atParentStartColumnIndex;
    int atParentEndColumnIndex;

public:
    StarImagePart(const Mat parentMat, int atParentStartRowIndex, int atParentEndRowIndex,
                  int atParentStartColumnIndex, int atParentEndColumnIndex, int rowPartIndex, int columnPartIndex);

    Mat getImage();

    void setImage(Mat_<Vec3b> imageMat);

    void addImagePixelValue(Mat resultImg, Mat targetImage, int imageCount);

    void addUpStarImagePart(Mat_<Vec3b> imageMat);

    int getAtParentStartRowIndex() const;

    int getAtParentEndRowIndex() const;

    int getAtParentStartColumnIndex() const;

    int getAtParentEndColumnIndex() const;

    int getRowPartIndex() const;

    int getColumnPartIndex() const;
};