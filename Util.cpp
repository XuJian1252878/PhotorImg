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

Mat_<Vec3b> superimposedImg(vector<Mat_<Vec3b>>& images, Mat_<Vec3b>& trainImg, Mat& skyMask) {

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
        matcher.knnMatch(queryDescriptors, trainDescriptors, knnMatches, 2);

        // 筛选符合条件的特征点信息（最近邻次比率法）
        std::vector<DMatch> matches;
        vector<Point2f> queryMatchPoints, trainMatchPoints;  //  用于存储已经匹配上的特征点对
        int skyBoundaryRange = (int)(skyMask.rows * 0.005);
        for (int index = 0; index < knnMatches.size(); index ++) {
            DMatch firstMatch = knnMatches[index][0];
            DMatch secondMatch = knnMatches[index][1];

            int queryIdx = firstMatch.queryIdx;
            int trainIdx = firstMatch.trainIdx;

            int qy = (int)(queryKeyPoints[queryIdx].pt.y - skyBoundaryRange),
                    ty = (int)(trainKeyPoints[trainIdx].pt.y - skyBoundaryRange);
            int qx = (int)(queryKeyPoints[queryIdx].pt.x),
                    tx = (int)(trainKeyPoints[trainIdx].pt.x);

            if (qy >= skyMask.rows || ty >= skyMask.rows) {
                continue;
                // Mat.at(行数, 列数)
            } else if ( skyMask.at<uchar>(qy, qx) > 0 || skyMask.at<uchar>(ty, tx) > 0 ) {
                continue;
            }

            if (firstMatch.distance < 0.75 * secondMatch.distance) {
                matches.push_back(firstMatch);

                trainMatchPoints.push_back(trainKeyPoints[firstMatch.trainIdx].pt);
                queryMatchPoints.push_back(queryKeyPoints[firstMatch.queryIdx].pt);
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
                    return -1;
                } else {
//                    cout << tmpFileName << endl;
//                    cout << buf.st_ctimespec.tv_sec << endl;
//                    cout << buf.st_mtimespec.tv_sec << endl;
                }

                files.push_back(path + "/" + tmpFileName);
            }
        }
    }

    closedir(p_dir);
    sort(files.begin(), files.end());

    return (int)files.size();
}


int MASK_PIXEL_THRESHOLD = 127;
bool adjustMaskPixel(Mat& mask) {
    if (mask.rows <= 0 || mask.cols <= 0) {
        return false;
    }

    for (int x = 0; x < mask.cols; x ++) {
        for (int y = 0; y < mask.rows; y ++) {
            if (mask.at<uchar>(y, x) < MASK_PIXEL_THRESHOLD) {
                mask.at<uchar>(y, x) = 0;
            } else {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    return true;
}


float subtractionImage(Mat_<Vec3b>& img1, Mat_<Vec3b>& img2) {

    if (img1.rows != img2.rows || img1.cols != img2.cols) {
        return -1;
    }

    float result = 0.0;
    for (int col = 0; col < img1.cols; col ++) {
        for (int row = 0; row < img1.rows; row ++) {
            Vec3b a = img1.at<Vec3b>(row, col);
            Vec3b b = img2.at<Vec3b>(row, col);

            int gray1 = (a[0]*299 + a[1]*587 + a[2]*114 + 500) / 1000;
            int gray2 = (b[0]*299 + b[1]*587 + b[2]*114 + 500) / 1000;

            result += abs(gray1 - gray2);

//            for (int i = 0; i < 3; i ++) {
//                result += abs(a[i] - b[i]);
//            }
        }
    }

    return result / (img1.rows * img1.cols);
}


/**
 * 计算三点所形成夹角的角度信息
 * @param basePtVec
 * @param lightestPtVec
 * @param destPtVec
 * @return
 */
float generateAngleOnThreePoints(vector<float>& basePtVec, vector<float>& lightestPtVec, vector<float>& destPtVec) {
    // cb 代表原点跟最亮星点的连线向量，向量指向b；ca代表原点跟目标星点的连线，向量指向a
    vector<float> cb;
    vector<float> ca;

    float cbMod = 0, caMod = 0, dotMulValue = 0;

    // 获得cb ca的具体表现形式
    for (int index = 0; index < basePtVec.size(); index ++) {
        cbMod += pow(lightestPtVec[index] - basePtVec[index], 2);
        cb.push_back(lightestPtVec[index] - basePtVec[index]);

        caMod += pow(destPtVec[index] - basePtVec[index], 2);
        ca.push_back(destPtVec[index] - basePtVec[index]);

        dotMulValue += ( (lightestPtVec[index] - basePtVec[index]) * (destPtVec[index] - basePtVec[index]) );
    }
    caMod = sqrt(caMod);
    cbMod = sqrt(cbMod);

    // 返回弧度信息
    float angle;  // 用弧度表示具体的角度信息，范围为[0, 2 * pi]
    float nyPi = (float)acos(-1.0);  // 角度为pi时候，对应的弧度

    // 判断夹角的正负信息（二维的情况下可以用叉积判断，但是这里也仅仅是涉及二维）
    // 二维的情况下通过行列式来判断旋转的方向，高维叉积的计算如：https://spaces.ac.cn/archives/2219 所示
    float x1 = cb[0], y1 = cb[1];
    float x2 = ca[0], y2 = ca[1];
    float crossMulValue = x1 * y2 - x2 * y1;

    //  crossMulValue 为正，逆时针；为负，顺时针；0的话向量共线，可能夹角是0度，也可能是180度
    if (crossMulValue == 0) {
        // 两向量共线
        if (dotMulValue > 0) {
            angle = 0.0;
        } else {
            angle = nyPi;
        }
    } else {
        // 当 dotMulValue / (caMod * cbMod) 的值接近于 正负1.0的时候，acos计算比较危险，容易出现越界
        float cosValue = dotMulValue / (caMod * cbMod);
        if (cosValue < -1.0) {
            cosValue = -1.0f;
        } else if (cosValue > 1.0) {
            cosValue = 1.0;
        }

        angle = acos(cosValue);

        if (crossMulValue < 0) {
            // 说明 ca 相对于 cb 发生了顺时针旋转，角度为负
            angle = 2 * nyPi - angle;
        }
    }

    return angle;
}


/**
 * 进行数据标准化
 * @param array
 */
void normalizationVector(vector<float>& array) {

    if (array.size() <= 0) {
        return;
    }

//    // 1. 寻找数组的最大最小值
//    float minValue = numeric_limits<float>::max();
//    float maxValue = numeric_limits<float>::min();
//
//    for (int index = 0; index < array.size(); index ++) {
//        if (array[index] < minValue) {
//            minValue = array[index];
//        }
//
//        if (array[index] > maxValue) {
//            maxValue = array[index];
//        }
//    }
//    float maxMinRange = maxValue - minValue;

//    // 数据标准化 采用 min max标准化：https://blog.csdn.net/pipisorry/article/details/52247379
//    for (int index = 0; index < array.size(); index ++) {
//        array[index] = (array[index] - minValue) / maxMinRange;
//    }

//    // 2. z-score 标准化
//    float mean = 0;
//    float standardDeviation = 0;
//
//    for (int index = 0; index < array.size(); index ++) {
//        mean += array[index];
//    }
//    mean /= array.size();
//
//    for (int index = 0; index < array.size(); index ++) {
//        standardDeviation += pow(array[index] - mean, 2);
//    }
//    standardDeviation = sqrt(standardDeviation / (array.size() - 1));
//
//    for (int index = 0; index < array.size(); index ++) {
//        array[index] = (array[index] - mean) / standardDeviation;
//    }


    // 1. 寻找数组的最大最小值
    float sqrtNum = 0;
    for (int index = 0; index < array.size(); index ++) {
        sqrtNum += array[index];
    }
    sqrtNum = sqrt(sqrtNum);

    vector<float> tmpArray = array;
    // 标准化信息
    for (int index = 0; index < array.size(); index ++) {
        array[index] /= sqrtNum;
    }

}


void generateSkySeparatorMaskImgMat(Mat &skySeparatorMaskImg, Mat& realSkyMaskImg, Mat &skyMaskMat, int separatorRange) {

    if (skyMaskMat.rows <= 0 || skyMaskMat.cols <= 0) {
        return;
    }

    for (int rIndex = 1; rIndex < skyMaskMat.rows; rIndex ++) {

        for (int cIndex = 0; cIndex < skyMaskMat.cols; cIndex ++) {

            if (rIndex - separatorRange > 0) {
                // 构建星空分隔带mask
                skySeparatorMaskImg.at<uchar>(rIndex - separatorRange - 1, cIndex) = 0;
            }

            int before = skyMaskMat.at<uchar>(rIndex - 1, cIndex);
            int after = skyMaskMat.at<uchar>(rIndex, cIndex);

            if (before > MASK_PIXEL_THRESHOLD && after < MASK_PIXEL_THRESHOLD) {
                // 寻找到分界线之后
                // 1. 最终sky mask的操作
                for (int j = 0; j < separatorRange; j ++) {
                    realSkyMaskImg.at<uchar>(rIndex - j - 1, cIndex) = 0;
                }
            }
            break;

        }

    }

}