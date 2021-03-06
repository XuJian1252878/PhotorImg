//
// Created by 许舰 on 2018/3/11.
//

#include <sys/stat.h>
#include "Util.h"

Mat_<Vec3b> getTransformImgByHomo(Mat_<Vec3b>& queryImg, Mat homo) {
    //图像配准
    Mat imageTransform;
    warpPerspective(queryImg, imageTransform, homo, Size(queryImg.cols, queryImg.rows));

    return imageTransform;
}

/**
 * 按照 平均比例将图片叠加到一起
 * @param sourceImages
 * @return
 */
Mat_<Vec3b> addMeanImgs(std::vector<Mat_<Vec3b>>& sourceImages) {
    Mat_<Vec3b> resImage;
    if (sourceImages.size() <= 0) {
        return resImage;
    }
    resImage = Mat(sourceImages[0].rows, sourceImages[0].cols, sourceImages[0].type());
    for (int index = 0; index < sourceImages.size(); index ++) {
        resImage += (sourceImages[index] / sourceImages.size());
    }
    return resImage;
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


int getFiles(string path, vector<string>& files) {
    unsigned char isFile =0x8;
    DIR* p_dir;
    const char* str = path.c_str();

    p_dir = opendir(str);
    if (p_dir == NULL) {
        cout << "can't open " + path << endl;
        return -1;
    }

    struct dirent* p_dirent;
    while (p_dirent = readdir(p_dir)) {
        if (p_dirent->d_type == isFile) {
            // 该file是文件信息
            string tmpFileName = p_dirent->d_name;
            if (tmpFileName =="." || tmpFileName == "..") {
                continue;
            } else {
                // 获取文件状态信息
                struct stat buf;
                int result;
                result = stat(tmpFileName.c_str(), &buf);

                if (result == 0) {
                    cout << "文件状态信息出错, " + tmpFileName << endl;
                    return -1;
                } else {
                    cout << tmpFileName << endl;
                    cout << buf.st_ctimespec.tv_sec << endl;
                    cout << buf.st_mtimespec.tv_sec << endl;
                }

                files.push_back(path + "/" + tmpFileName);
            }
        }
    }

    closedir(p_dir);
    sort(files.begin(), files.end());

    return (int)files.size();
}
