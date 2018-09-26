//
// Created by 许舰 on 2018/3/11.
//

#ifndef PHOTOR_UTIL_H
#define PHOTOR_UTIL_H

#endif //PHOTOR_UTIL_H

#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <errno.h>
#include <opencv/cv.hpp>
#include <time.h>

#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

Mat_<Vec3b> addMeanImgs(std::vector<Mat_<Vec3b>>& sourceImages);

Mat_<Vec3b> superimposedImg(vector<Mat_<Vec3b>>& images, Mat_<Vec3b>& trainImg);

Mat_<Vec3b> superimposedImg(Mat_<Vec3b>& queryImg, Mat_<Vec3b>& trainImg);

int getFiles(string path, vector<string>& files);

/**
 * 根据配准参数homo，获得queryImg根据配准参数 变换 后的图像
 * @param queryImg
 * @param homo
 * @return
*/
Mat_<Vec3b> getTransformImgByHomo(Mat_<Vec3b>& queryImg, Mat homo);

bool adjustMaskPixel(Mat& mask);