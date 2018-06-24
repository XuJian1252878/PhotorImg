#include <iostream>
#include <opencv/cv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "StarImageRegistBuilder.h"
#include "Util.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


//bool compare(Vec3b a, Vec3b b);

Mat process(std::vector<Mat_<Vec3b>> sourceImages, Mat_<Vec3b> targetImage) {

    Mat groundMaskImgMat = imread("/Users/xujian/Desktop/JPEG_20180618_074240_C++.jpg", IMREAD_UNCHANGED);
    Mat skyMaskMat = ~groundMaskImgMat;

    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0101.jpg", skyMaskMat);
    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0102.jpg", groundMaskImgMat);

    // 目标矩阵的操作
    Mat_<Vec3b> skyImgMat;
    targetImage.copyTo(skyImgMat, skyMaskMat);
    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0103.jpg", skyImgMat);

    Mat_<Vec3b> groundImgMat;
    targetImage.copyTo(groundImgMat, groundMaskImgMat);
    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0104.jpg", groundImgMat);

    // 源矩阵的操作
    std::vector<Mat_<Vec3b>> skyImgs;
    for (int i = 0; i < sourceImages.size(); i ++) {
        Mat mat = sourceImages[i];
        Mat skyPartMat;
        mat.copyTo(skyPartMat, skyMaskMat);
        imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0105.jpg", skyPartMat);
        skyImgs.push_back(skyPartMat);
    }

    std::vector<Mat_<Vec3b>> groundImgs;
    for (int i = 0; i < sourceImages.size(); i ++) {
        Mat mat = sourceImages[i];
        Mat groundPartMat;
        mat.copyTo(groundPartMat, groundMaskImgMat);
        imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0106.jpg", groundPartMat);
        groundImgs.push_back(groundPartMat);
    }

    int rowParts = 5;
    int columnParts = 5;

    StarImageRegistBuilder starImageRegistBuilder = StarImageRegistBuilder(skyImgMat, skyImgs, skyMaskMat, rowParts, columnParts);
    Mat_<Vec3b> resultImage = starImageRegistBuilder.registration(StarImageRegistBuilder::MERGE_MODE_MEAN);

    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0107.jpg", resultImage);

    Mat skyRes;
    resultImage.copyTo(skyRes, skyMaskMat);

    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0108.jpg", skyRes);

    Mat groundRes;
    groundImgs.push_back(groundImgMat);
    addMeanImgs(groundImgs).copyTo(groundRes, groundMaskImgMat);
//    addWeighted(groundImgMat, 0.5, groundImgs[0], 0.5, 0, groundRes);

    Mat finalRes = skyRes | groundRes;

//    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/result-image-" + std::to_string(index) + ".jpg", finalRes);

    return finalRes;
}

/** @function main */
int main( int argc, char** argv )
{

    vector<string> files;
    string folder = "/Users/xujian/Downloads/11";
    getFiles(folder, files);

    if (files.size() <= 0) {
        return -1;
    }

    int targetIndex = (int)(files.size() / 2);
    string targetImgPath = files[files.size() / 2];
    Mat_<Vec3b> targetImage = imread(targetImgPath, IMREAD_UNCHANGED);

//    // 多张整合成一张照片的逻辑
//    std::vector<Mat_<Vec3b>> sourceImages;
//    for (int i = 0 ; i < files.size(); i ++) {
//        if (i == targetIndex) {
//            continue;
//        }
//
//        cout << files[i] << endl;
//        sourceImages.push_back(imread(files[i], IMREAD_UNCHANGED));
//    }
//
//    Mat tmpResult = process(sourceImages, targetImage);  // 以后的逻辑中, sourceImages不是vector，而变成了一个string的图片路径
//    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/result-image.jpg", tmpResult);



//    // 11 张结果照片的逻辑
//    for (int index = 0; index < targetIndex; index ++) {
//        string sourceImgPath = files[index];
//        std::vector<Mat_<Vec3b>> sourceImages;
//        sourceImages.push_back(imread(sourceImgPath, IMREAD_UNCHANGED));
//
//        for (int j = index + 1; j < targetIndex; j ++) {
//            Mat_<Vec3b> tmpResult = imread(files[j], IMREAD_UNCHANGED);
//            sourceImages[0] = process(sourceImages, tmpResult);
//            imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/test/"+ std::to_string(j)+".jpg", sourceImages[0]);
//        }
//
//        Mat tmpResult = process(sourceImages, targetImage);  // 以后的逻辑中, sourceImages不是vector，而变成了一个string的图片路径
//        imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/result-image-" + std::to_string(index) + ".jpg", tmpResult);
//    }
//
//    for (int index = (int)files.size() - 1; index > targetIndex; index --) {
//        string sourceImgPath = files[index];
//        std::vector<Mat_<Vec3b>> sourceImages;
//        sourceImages.push_back(imread(sourceImgPath, IMREAD_UNCHANGED));
//
//        for (int j = index; j < targetIndex; j ++) {
//            Mat_<Vec3b> tmpResult = imread(files[j], IMREAD_UNCHANGED);
//            sourceImages[0] = process(sourceImages, tmpResult);
//        }
//
//        Mat tmpResult = process(sourceImages, targetImage);  // 以后的逻辑中, sourceImages不是vector，而变成了一个string的图片路径
//        imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/result-image-" + std::to_string(index) + ".jpg", tmpResult);
//    }

}


///** @function main */
//int main( int argc, char** argv )
//{
//
//    vector<string> files;
//    string folder = "/Users/xujian/Downloads/11";
//    getFiles(folder, files);
//
////    string sourceImgPath1 = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/b1-1Y8A4155.jpg";
//    string targetImgPath = "/Users/xujian/Downloads/1Y1.JPG";
//    string sourceImgPath2 =  "/Users/xujian/Downloads/1Y6.JPG";
//
//    Mat_<Vec3b> targetImage = imread(targetImgPath, IMREAD_UNCHANGED);
//    std::vector<Mat_<Vec3b>> sourceImages;
////    sourceImages.push_back(imread(sourceImgPath1, IMREAD_UNCHANGED));
//    sourceImages.push_back(imread(sourceImgPath2, IMREAD_UNCHANGED));
//
//    Mat groundMaskImgMat = imread("/Users/xujian/Desktop/JPEG_20180618_074240_C++.jpg", IMREAD_UNCHANGED);
////    Mat groundMaskImgMat ;
////    groundMaskImgMat.create(groundMaskImgMat_.size(), CV_8UC1);
////    groundMaskImgMat = groundMaskImgMat_ & 1;
//    Mat skyMaskMat = ~groundMaskImgMat;
//
//    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0101.jpg", skyMaskMat);
//    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0102.jpg", groundMaskImgMat);
//
//    // 目标矩阵的操作
//    Mat_<Vec3b> skyImgMat;
//    targetImage.copyTo(skyImgMat, skyMaskMat);
//    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0103.jpg", skyImgMat);
//
//    Mat_<Vec3b> groundImgMat;
//    targetImage.copyTo(groundImgMat, groundMaskImgMat);
//    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0104.jpg", groundImgMat);
//
//    // 源矩阵的操作
//    std::vector<Mat_<Vec3b>> skyImgs;
//    for (int i = 0; i < sourceImages.size(); i ++) {
//        Mat mat = sourceImages[i];
//        Mat skyPartMat;
//        mat.copyTo(skyPartMat, skyMaskMat);
//        imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0105.jpg", skyPartMat);
//        skyImgs.push_back(skyPartMat);
//    }
//
//    std::vector<Mat_<Vec3b>> groundImgs;
//    for (int i = 0; i < sourceImages.size(); i ++) {
//        Mat mat = sourceImages[i];
//        Mat groundPartMat;
//        mat.copyTo(groundPartMat, groundMaskImgMat);
//        imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0106.jpg", groundPartMat);
//        groundImgs.push_back(groundPartMat);
//    }
//
//    int rowParts = 5;
//    int columnParts = 5;
//
//    StarImageRegistBuilder starImageRegistBuilder = StarImageRegistBuilder(skyImgMat, skyImgs, skyMaskMat, rowParts, columnParts);
//    Mat_<Vec3b> resultImage = starImageRegistBuilder.registration(StarImageRegistBuilder::MERGE_MODE_MEAN);
//
//    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0107.jpg", resultImage);
//
//    Mat skyRes;
//    resultImage.copyTo(skyRes, skyMaskMat);
//
//    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0108.jpg", skyRes);
//
//    Mat groundRes;
//    addWeighted(groundImgMat, 0.5, groundImgs[0], 0.5, 0, groundRes);
////    Mat groundRes_ = superimposedImg(groundImgs, groundImgMat);
////    Mat groundRes;
////    groundRes_.copyTo(groundRes, groundMaskImgMat);
//
//    Mat finalRes = skyRes | groundRes;
//
//    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/result-image.jpg", finalRes);
//
//}

//bool compare(Vec3b a, Vec3b b) {
//    for (int i = 0; i < 3; i++) {
//
//        cout << "a" << std::to_string(i) << std::to_string(a[i]) << "\t" << "b" << std::to_string(i) << std::to_string(b[i]) << endl;
//
//        if (a[i] != b[i]) {
//            return false;
//        }
//    }
//    return true;
//}


/**
    string sourceImgPath = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/a1-1Y8A0106.jpg";
    string targetImgPath = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/a2-1Y8A0108.jpg";

    Mat_<Vec3b> img_1 = imread( sourceImgPath, IMREAD_UNCHANGED );
    Mat_<Vec3b> img_2 = imread( targetImgPath, IMREAD_UNCHANGED );

    int type = img_1.type();

    if( !img_1.data || !img_2.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;

    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints_1, keypoints_2;

    Mat img_1 = skyImgMat;
    Mat img_2 = skyImgs[0];

    detector->detect( img_1, keypoints_1 );
    detector->detect( img_2, keypoints_2 );

//    SurfDescriptorExtractor extractor;
    Ptr<SURF> extractor = SURF::create();
    Mat descriptors_1, descriptors_2;
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);

    //-- Step 2: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< vector<DMatch> > knnMatches;
    matcher.knnMatch(descriptors_1, descriptors_2, knnMatches, 2);

    std::vector<DMatch> matches_;
    for (int index = 0; index < knnMatches.size(); index ++) {
        DMatch firstMatch = knnMatches[index][0];
        DMatch secondMatch = knnMatches[index][1];
        if (firstMatch.distance < 0.75 * secondMatch.distance) {
            matches_.push_back(firstMatch);
        }
    }

    sort(matches_.begin(), matches_.end());

    std::vector<DMatch> matches;
    vector<Point2f> imagePoints1, imagePoints2;
    for (int index = 0; index < matches_.size(); index ++) {
        cout << matches_[index].distance << "\n";

        int y1 = keypoints_1[matches_[index].queryIdx].pt.y + 288;
        int y2 = keypoints_2[matches_[index].trainIdx].pt.y + 288;
        int x1 = keypoints_1[matches_[index].queryIdx].pt.x;
        int x2 = keypoints_2[matches_[index].trainIdx].pt.x;

        if (y2 > skyImgMat.rows || y1 > skyImgMat.rows) {
            continue;
        }

        if (skyMaskMat.at<uchar>(y2, x2) == 0 || skyMaskMat.at<uchar>(y1, x1) == 0) {
            continue;
        }

        matches.push_back(matches_[index]);

        imagePoints1.push_back(keypoints_1[matches_[index].queryIdx].pt);
        imagePoints2.push_back(keypoints_2[matches_[index].trainIdx].pt);
    }

    Mat img_matches;
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
    string matchPath = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/0101test.jpg";
    imwrite(matchPath, img_matches);

    //获取图像1到图像2的投影映射矩阵 尺寸为3*3
    Mat homo=findHomography(imagePoints1,imagePoints2,CV_RANSAC);
    ////也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差
    //Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);
    cout<<"变换矩阵为：\n"<<homo<<endl<<endl; //输出映射矩阵

    //图像配准
    Mat imageTransform1,imageTransform2;
    warpPerspective(img_1,imageTransform1,homo,Size(img_2.cols,img_2.rows));
    imshow("经过透视矩阵变换后",imageTransform1);

    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/result-image-transform.jpg", imageTransform1);

    Mat imageResult;
    addWeighted(imageTransform1, 0.5, img_2, 0.5, 0, imageResult);
    imwrite("/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/result-image.jpg", imageResult);

    waitKey(0);
    return 0;
 *
 */