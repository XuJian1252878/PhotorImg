//
// Created by 许舰 on 2018/1/11.
//

#include "StarImageRegistBuilder.h"
#include "Util.h"

struct NearPoint {
    int queryIndex;
    int trainIndex;
    float dist;
};

/**
 *
 * @param targetImage:  train image
 * @param sourceImages: query image
 * @param rowParts
 * @param columnParts
 */
StarImageRegistBuilder::StarImageRegistBuilder(Mat_<Vec3b>& targetImage, std::vector<Mat_<Vec3b>>& sourceImages,
                                               Mat& skyMaskMat, int rowParts, int columnParts) {
    this->rowParts = rowParts;
    this->columnParts = columnParts;
    this->imageCount = (int)sourceImages.size() + 1;

    this->skyMaskMat = skyMaskMat;
    this->skyBoundaryRange = (int)(skyMaskMat.rows * 0.05);

    this->kNeighbor = 10; // 默认特征点描述符是5向量，最终的特征向量有10维。

    this->targetImage = targetImage;
    this->sourceImages = sourceImages;

    // 开始对每一张图片进行分块操作
    for (int index = 0; index < sourceImages.size(); index ++) {
        StarImage starImage = StarImage(sourceImages[index], this->rowParts, this->columnParts);
        this->sourceStarImages.push_back(starImage);
    }

    this->targetStarImage = StarImage(targetImage, this->rowParts, this->columnParts);
}


/**
 *
 * @param imgPath
 */
void StarImageRegistBuilder::addSourceImagePath(string imgPath) {
    /**
     * 必须首先调用 addTargetImagePath。
     * 如果图片大小和targetImage不相同，那么不能加入图片
     */
    Mat image = imread(imgPath, IMREAD_UNCHANGED);
    if (image.rows != targetStarImage.getImage().rows || image.cols != targetStarImage.getImage().cols) {
        return;
    }
    this->sourceStarImages.push_back(StarImage(image, this->rowParts, this->columnParts));
}

/**
 *
 * @param imgPath
 */
void StarImageRegistBuilder::setTargetImagePath(string imgPath) {
    this->targetStarImage = StarImage(imread(imgPath, IMREAD_UNCHANGED), this->rowParts, this->columnParts);
}

/**
 *
 * @return
 */
Mat_<Vec3b> StarImageRegistBuilder::registration(int mergeMode) {

    // 最终配准的图像信息

    StarImage resultStarImage = StarImage(Mat(this->targetStarImage.getImage().rows,
                                              this->targetStarImage.getImage().cols,
                                              this->targetStarImage.getImage().type(), cv::Scalar(0, 0, 0)),
                                          this->rowParts, this->columnParts, true); // true 表示 clone，深拷贝，不然会出现图片重叠的现象

    // 开始对图像的每一个部分进行对齐操作，分别与targetStarImage 做对比
    for (int index = 0; index < this->sourceStarImages.size(); index ++) {

        StarImage tmpStarImage = this->sourceStarImages[index];  // 直接赋值，不是指针操作，
        // 对于每一小块图像都做配准操作
        for (int rPartIndex = 0; rPartIndex < this->rowParts; rPartIndex ++) {
            for (int cPartIndex = 0; cPartIndex < this->columnParts; cPartIndex ++) {
                Mat homo;
                bool existHomo = false;

                Mat tmpRegistMat = this->getImgTransform(tmpStarImage.getStarImagePart(rPartIndex, cPartIndex),
                                                         this->targetStarImage.getStarImagePart(rPartIndex, cPartIndex), homo, existHomo);

//                Mat tmpRegistMat = this->getImgTransformNew(tmpStarImage.getStarImagePart(rPartIndex, cPartIndex),
//                                                         this->targetStarImage.getStarImagePart(rPartIndex, cPartIndex), homo, existHomo);

                Mat_<Vec3b>& queryImgTransform = this->sourceImages[index];
                if (existHomo) {
                    queryImgTransform = getTransformImgByHomo(queryImgTransform, homo);
                } else {
                    queryImgTransform = this->targetImage;
                }
                resultStarImage.getStarImagePart(rPartIndex, cPartIndex).addImagePixelValue(tmpRegistMat, queryImgTransform, this->skyMaskMat, this->imageCount);

                tmpStarImage.getStarImagePart(rPartIndex, cPartIndex).getImage().release();
            }
        }

        this->sourceImages[index].release();
    }

    // 对于配准图像和待配准图像做平均值操作（先买上目标图像的那一部分，这一段代码不能放在source整合的前面，不然图片会出现缝隙，原因待查）
    for (int rPartIndex = 0; rPartIndex < this->rowParts; rPartIndex ++) {
        for (int cPartIndex = 0; cPartIndex < this->columnParts; cPartIndex++) {
            Mat_<Vec3b> targetImg = this->targetStarImage.getStarImagePart(rPartIndex, cPartIndex).getImage();
            resultStarImage.getStarImagePart(rPartIndex, cPartIndex).addImagePixelValue(targetImg, this->targetImage, this->skyMaskMat, this->imageCount);
        }
    }

    // 对配准好的图像进行整合
    return resultStarImage.mergeStarImageParts();
}

/**
 *
 * @param sourceImagePart
 * @param targetImagePart
 * @return
 */
Mat StarImageRegistBuilder::getImgTransform(StarImagePart sourceImagePart, StarImagePart targetImagePart, Mat& oriImgHomo, bool& existHomo) {
    Mat sourceImg = sourceImagePart.getImage(); // query image
    Mat targetImg = targetImagePart.getImage(); // train image

    // 取出当前mask起始点的位置
    int rMaskIndex = sourceImagePart.getAlignStartRowIndex();
    int cMaskIndex = sourceImagePart.getAlignStartColumnIndex();


//        if( !sourceImg.data || !targetImg.data )
//        { std::cout<< " --(!) Error loading images " << std::endl; return NULL; }

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;

    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector->detect( sourceImg, keypoints_1 );
    detector->detect( targetImg, keypoints_2 );

    int rPartIndex = sourceImagePart.getRowPartIndex();
//    if (rPartIndex < 3)
//    {
//        cout << "----------------------------------------------------------------------------------------------------------" << endl;
//        cout << "rPartIndex: " << sourceImagePart.getRowPartIndex() << ", cPartIndex: " << sourceImagePart.getColumnPartIndex() << endl;
//        cout << "sourceImg: " << "row-" << sourceImg.rows << ", col-" << sourceImg.cols << endl;
//        cout << "keypoints_1: " << keypoints_1.size() << endl;
//        cout << "targetImg: " << "row-" << targetImg.rows << ", col-" << targetImg.cols << endl;
//        cout << "keypoints_2: " << keypoints_2.size() << endl;
//        cout << "----------------------------------------------------------------------------------------------------------" << endl;
//    }


    // SurfDescriptorExtractor extractor;
    Ptr<SURF> extractor = SURF::create();
    Mat descriptors_1, descriptors_2;
    extractor->compute(sourceImg, keypoints_1, descriptors_1);
    extractor->compute(targetImg, keypoints_2, descriptors_2);

    //-- Step 2: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< vector<DMatch> > knnMatches;
    try {
        /**
         * 经过调试，发现照片有些部分是全暗的，根本无法找到特征点，这个时候 FlannBasedMatcher 会报
         * opencv knnMatch error: (-210) type=0
         */
        matcher.knnMatch(descriptors_1, descriptors_2, knnMatches, 2);
    } catch (cv::Exception) {
        return sourceImg; // 直接采集source img的信息。
    }

    // 1. 最近邻次比律法，取出错误匹配点
    std::vector<DMatch> tempMatches; // 符合条件的匹配对
    for (int index = 0; index < knnMatches.size(); index ++) {
        DMatch firstMatch = knnMatches[index][0];
        DMatch secondMatch = knnMatches[index][1];
        if (firstMatch.distance < 0.9 * secondMatch.distance) {
            tempMatches.push_back(firstMatch);
        }
    }

    // 2. 对应 query image中的多个特征点对应 target image中的同一个特征点的情况（导致计算出的映射关系不佳），只取最小的匹配
    vector<Point2f> imagePoints1, imagePoints2;
//    std::map<int, DMatch> matchRepeatRecords;
//    for (int index = 0; index < tempMatches.size(); index ++) {
////        int queryIdx = matches[index].queryIdx;
//        int trainIdx = tempMatches[index].trainIdx;
//
//        // 记录标准图像中的每个点被配准了多少次，如果被配准多次，那么说明这个特征点匹配不合格
//        if (matchRepeatRecords.count(trainIdx) <= 0) {
//            matchRepeatRecords[trainIdx] = tempMatches[index];
//        } else {
//            // 多个query image的特征点对应 target image的特征点时，只取距离最小的一个匹配（这个算是双向匹配的改进）
//            if (matchRepeatRecords[trainIdx].distance > tempMatches[index].distance) {
//                matchRepeatRecords[trainIdx] = tempMatches[index];
//            }
//        }
//    }


    // 3. 计算匹配特征点对的标准差信息（标准差衡量标准之间只有相互的，那么要将标准差，以及match等一切中间过程存储起来，之后再进行配准，太耗内存）
    std::map<int, DMatch>::iterator iter;
    // 3.1 获取准确的最大最小值
//    double maxMatchDist = 0;
//    double minMatchDist = 100;
//    for (iter = matchRepeatRecords.begin(); iter != matchRepeatRecords.end(); iter ++) {
//        if (iter->second.distance < minMatchDist) {
//            minMatchDist = iter->second.distance;
//        }
//        if (iter->second.distance > maxMatchDist) {
//            maxMatchDist = iter->second.distance;
//        }
//    }

    // 4. 根据特征点匹配对，分离出两幅图像中已经被匹配的特征点
    std::vector<DMatch> matches;
//    double matchThreshold = minMatchDist + (maxMatchDist - minMatchDist) * 0.1;  // 阈值越大，留下的特征点越多（这个阈值是一个做文章的地方）
    double slopeThreshold = 0.3;
//    for (iter = matchRepeatRecords.begin(); iter != matchRepeatRecords.end(); iter ++) {
    for (int index = 0; index < tempMatches.size(); index ++) {

//        DMatch match = iter->second;
        DMatch match = tempMatches[index];
        int queryIdx = match.queryIdx;
        int trainIdx = match.trainIdx;

        // 将检测出的靠近边缘的特征点去除
        int qy = (int)(keypoints_1[queryIdx].pt.y + this->skyBoundaryRange + rMaskIndex),
                ty = (int)(keypoints_2[trainIdx].pt.y + this->skyBoundaryRange + rMaskIndex);
        int qx = (int)(keypoints_1[queryIdx].pt.x + cMaskIndex), tx = (int)(keypoints_2[trainIdx].pt.x + cMaskIndex);

        if (qy >= this->skyMaskMat.rows || ty >= this->skyMaskMat.rows) {
            continue;
            // Mat.at(行数, 列数)
        } else if ( this->skyMaskMat.at<uchar>(qy, qx) == 0 || this->skyMaskMat.at<uchar>(ty, tx) == 0 ) {
            continue;
        }

        // 5. 设置 匹配点之间的 distance 阈值来选出 质量好的特征点（方差策略的替代方案）
        // 6. 计算两个匹配点之间的斜率
        double slope = abs( (keypoints_1[queryIdx].pt.y - keypoints_2[trainIdx].pt.y) * 1.0 /
                                    (keypoints_1[queryIdx].pt.x -
                                            (keypoints_2[trainIdx].pt.x + targetImg.cols) ) );

        // && iter->second.distance < matchThreshold
        if ( slope < slopeThreshold) {
//            matches.push_back(iter->second);
            matches.push_back(match);
            imagePoints1.push_back(keypoints_1[queryIdx].pt);
            imagePoints2.push_back(keypoints_2[trainIdx].pt);
        }
    }

    // 测试代码：
    Mat img_matches;
    drawMatches( sourceImg, keypoints_1, targetImg, keypoints_2, matches, img_matches );
//    int rPartIndex = sourceImagePart.getRowPartIndex();
    int cPartIndex = sourceImagePart.getColumnPartIndex();
    string sfile1 = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/01/" + std::to_string(rPartIndex) + "_" + std::to_string(cPartIndex) + ".jpg";
    string sfile2 = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/02/" + std::to_string(rPartIndex) + "_" + std::to_string(cPartIndex) + ".jpg";
    imwrite(sfile1, sourceImg);
    imwrite(sfile2, targetImg);
    string matchPath = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/match/" + std::to_string(rPartIndex) + "_" + std::to_string(cPartIndex) + ".jpg";
    imwrite(matchPath, img_matches);

    if (rPartIndex < 3)
    {
        cout << "----------------------------------------------------------------------------------------------------------" << endl;
        cout << "rPartIndex\t" << sourceImagePart.getRowPartIndex() << "\tcPartIndex\t" << sourceImagePart.getColumnPartIndex() << endl;
        cout << "sourceImg\t" << sourceImg.rows << "\t" << sourceImg.cols << "\t" << "imagePoints1\t" << imagePoints1.size() << endl;

        cout << "targetImg\t" << targetImg.rows << "\t" << targetImg.cols << "\t" << "imagePoints2\t" << imagePoints2.size() << endl;
        cout << "----------------------------------------------------------------------------------------------------------" << endl;
    }

    int IMG_MATCH_POINT_THRESHOLD = 10;  // 这里是个做文章的地方
    // 对应图片部分中没有特征点的情况（导致计算出的映射关系不佳，至少要4对匹配点才能计算出匹配关系）
    if (imagePoints1.size() >= IMG_MATCH_POINT_THRESHOLD && imagePoints2.size() < IMG_MATCH_POINT_THRESHOLD) {
        // 没有特征点信息，那么说明这个区域是没有特征的，所以返回 查询图片部分，作为内容填充
        return sourceImg;
    } else if (imagePoints1.size() < IMG_MATCH_POINT_THRESHOLD && imagePoints2.size() >= IMG_MATCH_POINT_THRESHOLD) {
        return targetImg;
    } else if (imagePoints1.size() < IMG_MATCH_POINT_THRESHOLD && imagePoints2.size() < IMG_MATCH_POINT_THRESHOLD) {
        return targetImg;  // 特征点都不足时，以targetImage为基础进行的配准
    }

    // 获取图像1到图像2的投影映射矩阵 尺寸为3*3
    Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
    // 也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差
    // Mat homo = getPerspectiveTransform(imagePoints1,imagePoints2);
    /**
     * 这里如果有一副图片中的特征点过少，导致查询图片部分 中的多个特征点直接 和 目标图片部分 中的同一个特征点相匹配，
     * 那么会导致算不出变换矩阵，变换矩阵为 [] 。导致错误。
     */
    if (homo.rows < 3 || homo.cols < 3) {
        existHomo = false;
        if (imagePoints1.size() > imagePoints2.size()) {
            return sourceImg; // 因为是星空图片，移动不会很大，在目标图片部分的特征点几乎没有的情况下，那么直接返回待配准图像进行填充细节。
        } else {
            return targetImg;
        }

    }

    oriImgHomo = homo;
    existHomo = true;
    //图像配准
    Mat sourceImgTransform;
    warpPerspective(sourceImg, sourceImgTransform ,homo , Size(targetImg.cols, targetImg.rows));

    string sfile3 = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/translatre/" + std::to_string(rPartIndex) + "_" + std::to_string(cPartIndex) + ".jpg";
    imwrite(sfile3, sourceImgTransform);

    return sourceImgTransform;
}

/**
 * 改进的星空图像配准方法
 * 采用新的特征点描述符替代SURF特征点描述符
 * @param sourceImagePart
 * @param targetImagePart
 * @return
 */
Mat StarImageRegistBuilder::getImgTransformNew(StarImagePart sourceImagePart, StarImagePart targetImagePart,
                                               Mat& oriImgHomo, bool& existHomo) {

    Mat sourceImg = sourceImagePart.getImage(); // query image
    Mat targetImg = targetImagePart.getImage(); // train image

    // 取出当前mask起始点的位置
    int rMaskIndex = sourceImagePart.getAlignStartRowIndex();
    int cMaskIndex = sourceImagePart.getAlignStartColumnIndex();

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;

    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints_1_, keypoints_2_;
    detector->detect( sourceImg, keypoints_1_ );
    detector->detect( targetImg, keypoints_2_ );

    // 去除星空地景边缘的特征点（减少之后的计算量）
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<KeyPoint>::iterator pts_iter = keypoints_1_.begin();
    while (pts_iter != keypoints_1_.end()) {

        int y = (int)(pts_iter->pt.y + this->skyBoundaryRange + rMaskIndex);
        int x = (int)(pts_iter->pt.x + cMaskIndex);

        if (y >= this->skyMaskMat.rows || this->skyMaskMat.at<uchar>(y, x) == 0 ) {
            // 星空地景边缘的特征点
        } else {
            keypoints_1.push_back(*pts_iter);
        }

        pts_iter ++;
    }

    pts_iter = keypoints_2_.begin();
    while (pts_iter != keypoints_2_.end()) {

        int y = (int)(pts_iter->pt.y + this->skyBoundaryRange + rMaskIndex);
        int x = (int)(pts_iter->pt.x + cMaskIndex);

        if (y >= this->skyMaskMat.rows || this->skyMaskMat.at<uchar>(y, x) == 0 ) {
            // 星空地景边缘的特征点
        } else {
            keypoints_2.push_back(*pts_iter);
        }

        pts_iter ++;
    }

    // 初始化 flann 最近邻查询
    vector<vector<float>> pts_1;
    vector<vector<float>> pts_2;
    for (int index = 0; index < keypoints_1.size(); index ++) {
        vector<float> tmpVec;
        tmpVec.push_back(keypoints_1[index].pt.x);
        tmpVec.push_back(keypoints_1[index].pt.y);
        pts_1.push_back(tmpVec);
    }

    for (int index = 0; index < keypoints_2.size(); index ++) {
        vector<float> tmpVec;
        tmpVec.push_back(keypoints_2[index].pt.x);
        tmpVec.push_back(keypoints_2[index].pt.y);
        pts_2.push_back(tmpVec);
    }

    vector<vector<NearPoint>> NN1 = getNearestNPoints(pts_1, pts_1, this->kNeighbor + 1);
    vector<vector<NearPoint>> NN2 = getNearestNPoints(pts_2, pts_2, this->kNeighbor + 1);

    // 构建特征点描述符
    vector<vector<float>> pt_desc_1 = getPointDesc(NN1, pts_1, sourceImg);
    vector<vector<float>> pt_desc_2 = getPointDesc(NN2, pts_2, targetImg);

    // 根据特征点描述符对特征点进行匹配
    vector<vector<NearPoint>> NNMatch = getNearestNPoints(pt_desc_1, pt_desc_2, 2); // 找到距离最近的两个邻居

    // 1. 根据最近邻次比率法找出符合要求的特征点
    vector<NearPoint> tmpMatches;
    for (int index = 0; index < NNMatch.size(); index ++) {
        NearPoint& firstMatch = NNMatch[index][0].dist <= NNMatch[index][1].dist ? NNMatch[index][0] : NNMatch[index][1];
        NearPoint& secondMatch = NNMatch[index][0].dist > NNMatch[index][1].dist ? NNMatch[index][0] : NNMatch[index][1];

        if (firstMatch.dist < 0.9 * secondMatch.dist) {
            tmpMatches.push_back(firstMatch);
        }
    }

    // 2. 根据匹配的斜率信息去除没有用处的特征点信息（两类：1. 位于星空地景的边缘；2. 匹配点之间的连线超过阈值）
    vector<NearPoint> finalMatches;
    vector<Point2f> imagePoints1, imagePoints2;
    vector<DMatch> testMatches;

    double slopeThreshold = 0.3;
    for (int index = 0; index < tmpMatches.size(); index ++) {
        NearPoint match = tmpMatches[index];
        int queryIdx = match.queryIndex;
        int trainIdx = match.trainIndex;

        // 4. 计算两个匹配点之间的斜率，去除斜率过大的匹配点
        double slope = abs( (keypoints_1[queryIdx].pt.y - keypoints_2[trainIdx].pt.y) * 1.0 /
                            (keypoints_1[queryIdx].pt.x -
                             (keypoints_2[trainIdx].pt.x + targetImg.cols) ) );

        if ( slope < slopeThreshold) {
            finalMatches.push_back(match);
            imagePoints1.push_back(keypoints_1[queryIdx].pt);
            imagePoints2.push_back(keypoints_2[trainIdx].pt);

            // 添加match信息
            DMatch mt(queryIdx, trainIdx, match.dist);
            testMatches.push_back(mt);
        }
    }

    // 测试代码：
    Mat img_matches;
    drawMatches( sourceImg, keypoints_1, targetImg, keypoints_2, testMatches, img_matches );
    int rPartIndex = sourceImagePart.getRowPartIndex();
    int cPartIndex = sourceImagePart.getColumnPartIndex();
    string sfile1 = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/01/" + std::to_string(rPartIndex) + "_" + std::to_string(cPartIndex) + ".jpg";
    string sfile2 = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/02/" + std::to_string(rPartIndex) + "_" + std::to_string(cPartIndex) + ".jpg";
    imwrite(sfile1, sourceImg);
    imwrite(sfile2, targetImg);
    string matchPath = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/match/" + std::to_string(rPartIndex) + "_" + std::to_string(cPartIndex) + ".jpg";
    imwrite(matchPath, img_matches);

    if (rPartIndex < 3)
    {
        cout << "----------------------------------------------------------------------------------------------------------" << endl;
        cout << "rPartIndex\t" << sourceImagePart.getRowPartIndex() << "\tcPartIndex\t" << sourceImagePart.getColumnPartIndex() << endl;
        cout << "sourceImg\t" << sourceImg.rows << "\t" << sourceImg.cols << "\t" << "imagePoints1\t" << imagePoints1.size() << endl;

        cout << "targetImg\t" << targetImg.rows << "\t" << targetImg.cols << "\t" << "imagePoints2\t" << imagePoints2.size() << endl;
        cout << "----------------------------------------------------------------------------------------------------------" << endl;
    }


    int IMG_MATCH_POINT_THRESHOLD = 10;  // 这里是个做文章的地方
    // 对应图片部分中没有特征点的情况（导致计算出的映射关系不佳，至少要4对匹配点才能计算出匹配关系）
    if (imagePoints1.size() >= IMG_MATCH_POINT_THRESHOLD && imagePoints2.size() < IMG_MATCH_POINT_THRESHOLD) {
        // 没有特征点信息，那么说明这个区域是没有特征的，所以返回 查询图片部分，作为内容填充
        return sourceImg;
    } else if (imagePoints1.size() < IMG_MATCH_POINT_THRESHOLD && imagePoints2.size() >= IMG_MATCH_POINT_THRESHOLD) {
        return targetImg;
    } else if (imagePoints1.size() < IMG_MATCH_POINT_THRESHOLD && imagePoints2.size() < IMG_MATCH_POINT_THRESHOLD) {
        return targetImg;  // 特征点都不足时，以targetImage为基础进行的配准
    }

    // 获取图像1到图像2的投影映射矩阵 尺寸为3*3
    Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
    // 也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差
    // Mat homo = getPerspectiveTransform(imagePoints1,imagePoints2);
    /**
     * 这里如果有一副图片中的特征点过少，导致查询图片部分 中的多个特征点直接 和 目标图片部分 中的同一个特征点相匹配，
     * 那么会导致算不出变换矩阵，变换矩阵为 [] 。导致错误。
     */
    if (homo.rows < 3 || homo.cols < 3) {
        existHomo = false;
        if (imagePoints1.size() > imagePoints2.size()) {
            return sourceImg; // 因为是星空图片，移动不会很大，在目标图片部分的特征点几乎没有的情况下，那么直接返回待配准图像进行填充细节。
        } else {
            return targetImg;
        }

    }

    oriImgHomo = homo;
    existHomo = true;
    //图像配准
    Mat sourceImgTransform;
    warpPerspective(sourceImg, sourceImgTransform ,homo , Size(targetImg.cols, targetImg.rows));

    string sfile3 = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/translatre/" + std::to_string(rPartIndex) + "_" + std::to_string(cPartIndex) + ".jpg";
    imwrite(sfile3, sourceImgTransform);

    return sourceImgTransform;
}


vector<vector<NearPoint>> StarImageRegistBuilder::getNearestNPoints(vector<vector<float>>& targetPoints,
                                                                    vector<vector<float>>& pointArray,
                                                                    int knnNum) {

    vector<vector<NearPoint>> result; // 最终的返回结果
    // 把n个数据点转化成flann参数的矩阵形式
    if (targetPoints.size() <= 0 || pointArray.size() <= 0) {
        return result;
    }
    int nFeature = targetPoints[0].size();

    // 把数据点存为矩阵，其中 M(i,j) 表示第i个点的第j维数据
    Mat browMat(pointArray.size(), nFeature, CV_32FC1);  // Mat::Mat(int _rows, int _cols, int _type)
    for (int rIndex = 0; rIndex < pointArray.size(); rIndex ++) {
        vector<float>& browVec = pointArray[rIndex];
        for (int cIndex = 0; cIndex < browVec.size(); cIndex ++) {
            browMat.at<float>(rIndex, cIndex) = browVec[cIndex];
        }
    }

    // 建立kd树
    flann::Index knn(browMat, flann::KDTreeIndexParams());

    // 将数据点targetPoint转化为矩阵 （1行，nFeature列）
    // 如果要同时搜索m个数据点的最近邻点，那么矩阵为m行，nFeature列
    Mat targetMat(targetPoints.size(), nFeature, CV_32FC1);
    for (int rIndex = 0; rIndex < targetPoints.size(); rIndex ++) {
        vector<float>& vec = targetPoints[rIndex];
        for (int cIndex = 0; cIndex < vec.size(); cIndex ++) {
            targetMat.at<float>(rIndex, cIndex) = vec[cIndex];
        }
    }

    // 结果
    Mat nearestIndexMat;
    Mat nearestDistMat;
    knn.knnSearch(targetMat, nearestIndexMat, nearestDistMat, knnNum);

    //获取结果，nearestIndexMat、nearestDistMat(targetMat.rows,num_knn)，因为这里指搜索 数据点的 knnNum 个最近邻点，所以返回的矩阵大小为(m, knnNum)
    for (int rIndex = 0; rIndex < nearestIndexMat.rows; rIndex ++) {
        vector<NearPoint> tmpVec;
        for (int cIndex = 0; cIndex < knnNum; cIndex ++) {

            if (nearestIndexMat.at<int>(rIndex, cIndex) >= pointArray.size() || nearestIndexMat.at<int>(rIndex, cIndex) < 0) {
                // 当特征点的坐标值中有NaN值时，将会出现这种情况。
            }

            NearPoint np = {
                    .queryIndex = rIndex,
                    .trainIndex = nearestIndexMat.at<int>(rIndex, cIndex),
                    .dist = nearestDistMat.at<float>(rIndex, cIndex)
            };
            tmpVec.push_back(np);
        }
        result.push_back(tmpVec);
    }

    return result;
}


/**
 * 获得每个特征丢安自定义的特征点描述符
 * @param NN
 * @param pts
 * @param imgMat
 * @return
 */
vector<vector<float>> StarImageRegistBuilder::getPointDesc(vector<vector<NearPoint>>& NN, vector<vector<float>>& pts, Mat& imgMat) {

    vector<vector<float>> result; // 最终返回结果（每个特征点的描述符）

    // .1 寻找 某一个特征点 k个最近的邻居中，最亮的邻居，以此作为基准
    for (int i = 0; i < NN.size(); i ++) {

        // 开始检查k个最近的邻居（为了找出k个邻居中最亮的邻居点）
        vector<NearPoint>& npVec = NN[i];
        int lightestIndex = -1; // k个邻居中最亮特征点的下标
        int maxLightest = -1;  // 像素的亮度不可能超过 300
        for (int j = 1; j < npVec.size(); j ++) {
            // 获取当前邻居点的坐标vec
            vector<float>& ptVec = pts[npVec[j].trainIndex];
            // 取邻居点的像素判断亮度（at 行数，列数）
            Vec3b& vec3b = imgMat.at<Vec3b>((int)ptVec[1], (int)ptVec[0]);
            int light = (vec3b[2] * 30 + vec3b[1] * 59 + vec3b[0] * 11 + 50) / 100;

            if (light > maxLightest) {
                maxLightest = light;
                lightestIndex = npVec[j].trainIndex;
            }
        }

        // 开始生成 每一个特征点的 2k维 描述符
        int baseIndex = i;
        vector<float>& basePtVec = pts[i];  // 当前特征点（作为原点）
        vector<float>& lightestPtVec = pts[lightestIndex];  // k个邻居中最亮的特征点（作为角度的基准起始）

        vector<float> angleVec;  // 存储角度描述符信息
        vector<float> distVec;  // 存储距离描述符信息

        for (int j = 1; j < npVec.size(); j ++) {
            vector<float>& destPtVec = pts[npVec[j].trainIndex]; // 待测量的特征点信息

            angleVec.push_back(generateAngleOnThreePoints(basePtVec, lightestPtVec, destPtVec));
            distVec.push_back(npVec[j].dist);  // 填充距离信息
        }

        // 分别进行向量的标准化
        normalizationVector(angleVec);
        normalizationVector(distVec);

        // 2k维 的描述符中，前k个是描述角度的，范围为[0, 2 * pi]；后k个是描述距离的。
        vector<float> tmpDescVec;
        tmpDescVec.insert(tmpDescVec.end(), angleVec.begin(), angleVec.end());
        tmpDescVec.insert(tmpDescVec.end(), distVec.begin(), distVec.end());

        result.push_back(tmpDescVec);
    }

    return result;
}



