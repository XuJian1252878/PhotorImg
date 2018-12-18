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

    // 作为最终结果的图片区域
    int atParentStartRowIndex;
    int atParentEndRowIndex;
    int atParentStartColumnIndex;
    int atParentEndColumnIndex;

    // 作为对齐的图片区域
    int alignStartRowIndex;
    int alignEndRowIndex;
    int alignStartColumnIndex;
    int alignEndColumnIndex;

    Mat homo; // 存储射影变换的参数

public:
    StarImagePart(const Mat parentMat,
                  int atParentStartRowIndex, int atParentEndRowIndex,
                  int atParentStartColumnIndex, int atParentEndColumnIndex,
                  int rowPartIndex, int columnPartIndex,
                  int alignStartRowIndex, int alignEndRowIndex,
                  int alignStartColumnIndex, int alignEndColumnIndex,
                  bool isClone = false);

    Mat& getImage();

    void setImage(Mat_<Vec3b> imageMat);

    void addImagePixelValue(Mat& resultImg,
                            Mat& queryImgTransform, Mat& skyMaskImg, int imageCount);

    void addUpStarImagePart(Mat_<Vec3b> imageMat);

    int getAtParentStartRowIndex() const;

    int getAtParentEndRowIndex() const;

    int getAtParentStartColumnIndex() const;

    int getAtParentEndColumnIndex() const;

    int getRowPartIndex() const;

    int getColumnPartIndex() const;

    int getAlignStartRowIndex() const;

    int getAlignEndRowIndex() const;

    int getAlignStartColumnIndex() const;

    int getAlignEndColumnIndex() const;

    void addImagePixelValue(Mat& resultImg, int imageCount);
};