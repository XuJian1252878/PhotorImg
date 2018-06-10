//
// Created by 许舰 on 2018/3/11.
//

#include "Util.h"

Mat_<Vec3b> getTransformImgByHomo(Mat_<Vec3b>& queryImg, Mat homo) {

    //图像配准
    Mat imageTransform;
    warpPerspective(queryImg, imageTransform, homo, Size(queryImg.cols, queryImg.rows));

    return imageTransform;

}

Mat_<Vec3b> superimposedImg(vector<Mat_<Vec3b>>& images, Mat_<Vec3b>& trainImg) {

    int count = images.size();
    Mat resImg;

    // 如果images没有图像，返回一个空的Mat
    if (count <= 0) {
        return resImg;
    } else if (count <= 1) {
        return images[0];  // 如果只有一幅图像，那么没有办法进行配准操作，直接返回唯一的一幅图像
    }

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> trainKeyPoints;
    detector->detect( trainImg, trainKeyPoints );

    Ptr<SURF> extractor = SURF::create();
    Mat trainDescriptors;
    extractor->compute(trainImg, trainKeyPoints, trainDescriptors);

    resImg = Mat::zeros(images[0].rows, images[0].cols, images[0].type());

    for (int index = 1; index < count; index ++) {

        // 检测特征点以及特征点描述符
        Mat_<Vec3b> queryImg = images[index];
        std::vector<KeyPoint> queryKeyPoints;
        detector->detect( queryImg, queryKeyPoints );

        Mat queryDescriptors;
        extractor->compute(queryImg, queryKeyPoints, queryDescriptors);

        // 生成匹配点信息
        FlannBasedMatcher matcher;
        std::vector< vector<DMatch> > knnMatches;
        matcher.knnMatch(trainDescriptors, queryDescriptors, knnMatches, 2);

        // 筛选符合条件的特征点信息（最近邻次比率法）
        std::vector<DMatch> matches;
        vector<Point2f> queryMatchPoints, trainMatchPoints;  //  用于存储已经匹配上的特征点对
        for (int index = 0; index < knnMatches.size(); index ++) {
            DMatch firstMatch = knnMatches[index][0];
            DMatch secondMatch = knnMatches[index][1];
            if (firstMatch.distance < 0.75 * secondMatch.distance) {
                matches.push_back(firstMatch);

                trainMatchPoints.push_back(trainKeyPoints[firstMatch.queryIdx].pt);
                queryMatchPoints.push_back(queryKeyPoints[firstMatch.trainIdx].pt);
            }
        }

        // 计算映射关系
        //获取图像1到图像2的投影映射矩阵 尺寸为3*3
        Mat homo = findHomography(queryMatchPoints, trainMatchPoints, CV_RANSAC);

        //图像配准
        Mat imageTransform;
        warpPerspective(queryImg, imageTransform, homo, Size(trainImg.cols, trainImg.rows));

        resImg += (imageTransform / count);
    }

    return resImg;
}


Mat_<Vec3b> superimposedImg(Mat_<Vec3b>& queryImg, Mat_<Vec3b>& trainImg) {
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create(minHessian);
    std::vector<KeyPoint> trainKeyPoints;
    detector->detect(trainImg, trainKeyPoints);

    Ptr<SURF> extractor = SURF::create();
    Mat trainDescriptors;
    extractor->compute(trainImg, trainKeyPoints, trainDescriptors);



    // 检测特征点以及特征点描述符
    std::vector<KeyPoint> queryKeyPoints;
    detector->detect(queryImg, queryKeyPoints);

    Mat queryDescriptors;
    extractor->compute(queryImg, queryKeyPoints, queryDescriptors);

    // 生成匹配点信息
    FlannBasedMatcher matcher;
    std::vector<vector<DMatch> > knnMatches;
    matcher.knnMatch(trainDescriptors, queryDescriptors, knnMatches, 2);

    // 筛选符合条件的特征点信息（最近邻次比率法）
    std::vector<DMatch> matches;
    vector<Point2f> queryMatchPoints, trainMatchPoints;  //  用于存储已经匹配上的特征点对
    for (int index = 0; index < knnMatches.size(); index++) {
        DMatch firstMatch = knnMatches[index][0];
        DMatch secondMatch = knnMatches[index][1];
        if (firstMatch.distance < 0.75 * secondMatch.distance) {
            matches.push_back(firstMatch);

            trainMatchPoints.push_back(trainKeyPoints[firstMatch.queryIdx].pt);
            queryMatchPoints.push_back(queryKeyPoints[firstMatch.trainIdx].pt);
        }
    }

    // 计算映射关系
    //获取图像1到图像2的投影映射矩阵 尺寸为3*3
    Mat homo = findHomography(queryMatchPoints, trainMatchPoints, CV_RANSAC);

    //图像配准
    Mat imageTransform;
    warpPerspective(queryImg, imageTransform, homo, Size(trainImg.cols, trainImg.rows));

    return imageTransform;
}
