//
// Created by 许舰 on 2018/1/11.
//

#include "StarImagePart.h"

/**
 *
 * @param parentMat
 * @param atParentStartRowIndex
 * @param atParentEndRowIndex
 * @param atParentStartColumnIndex
 * @param atParentEndColumnIndex
 * @param rowPartIndex
 * @param columnPartIndex
 */
StarImagePart::StarImagePart(const Mat parentMat,
                             int atParentStartRowIndex, int atParentEndRowIndex,
                             int atParentStartColumnIndex, int atParentEndColumnIndex,
                             int rowPartIndex, int columnPartIndex,
                             int alignStartRowIndex, int alignEndRowIndex,
                             int alignStartColumnIndex, int alignEndColumnIndex,
                             bool isClone) {
    // [atParentStartRowIndex, atParentEndRowIndex)  [atParentStartColumnIndex, atParentEndColumnIndex)
    // [alignStartRowIndex, alignEndRowIndex)  [alignStartColumnIndex, alignEndColumnIndex)

    /**
     * this->imagePart = parentMat(Range(alignStartRowIndex, alignEndRowIndex), Range(alignStartColumnIndex, alignEndColumnIndex))
     * 这样的话，只是对父图的一部分进行引用，对this->imagePart做改变，那么父图也将会改变
     */

    if (isClone) {
        this->imagePart = parentMat(Range(alignStartRowIndex, alignEndRowIndex),
                                    Range(alignStartColumnIndex, alignEndColumnIndex)).clone();
    } else {
        this->imagePart = parentMat(Range(alignStartRowIndex, alignEndRowIndex),
                                    Range(alignStartColumnIndex, alignEndColumnIndex));
    }


    this->rowPartIndex = rowPartIndex;
    this->columnPartIndex = columnPartIndex;

    this->atParentStartRowIndex = atParentStartRowIndex;
    this->atParentEndRowIndex = atParentEndRowIndex;
    this->atParentStartColumnIndex = atParentStartColumnIndex;
    this->atParentEndColumnIndex = atParentEndColumnIndex;

    this->alignStartRowIndex = alignStartRowIndex;
    this->alignEndRowIndex = alignEndRowIndex;
    this->alignStartColumnIndex = alignStartColumnIndex;
    this->alignEndColumnIndex = alignEndColumnIndex;
}


/**
 *
 * @return
 */
Mat& StarImagePart::getImage() {
    return this->imagePart;
}

/**
 * 设置Image 中每一部分 Image Mat的信息，一般是存储 经过射影变换的 图形矩阵
 * @param imageMat
 */
void StarImagePart::setImage(Mat_<Vec3b> imageMat) {
    this->imagePart = imageMat;
}


/**
 * 前提 this->imagePart 初始为0矩阵
 * @param resultImg 当前被配准的 图片部分
 * @param queryImgTransform 经过 homo 射影变换参数变换后的一整副图像，以当前的 targetSkyPart为基准
 * @param imageCount 一共有多少张图片需要被配准
 */
void StarImagePart::addImagePixelValue(Mat& resultImg,
                                       Mat& queryImgTransform, Mat& skyMaskImg, int imageCount) {

    int boundaryGap = 100;
    int imgAreaGap = 100;

    // 取出当前mask起始点的位置
    int rMaskIndex = this->getAlignStartRowIndex();
    int cMaskIndex = this->getAlignStartColumnIndex();

    for (int rIndex = 0; rIndex < this->imagePart.rows; rIndex ++) {
        for (int cIndex = 0; cIndex < this->imagePart.cols; cIndex ++) {

            if (rMaskIndex + rIndex >= skyMaskImg.rows || cMaskIndex + cIndex >= skyMaskImg.cols) {
                continue; ////
            }

            // 检查skymask的边界两点，确定填充方式
            int skyMaskPixel = skyMaskImg.at<uchar>(rMaskIndex + rIndex, cMaskIndex + cIndex);
            if (skyMaskPixel < 127) {
                continue;
            }

            int skyMaskPixelAhead = -1;
            bool isTransition = false;
            if (rMaskIndex + rIndex + boundaryGap >= skyMaskImg.rows) {
                skyMaskPixelAhead = skyMaskImg.at<uchar>(skyMaskImg.rows - 1, cMaskIndex + cIndex);
            } else if (cMaskIndex + cIndex + boundaryGap >= skyMaskImg.cols) {
                skyMaskPixelAhead = skyMaskImg.at<uchar>(rMaskIndex + rIndex, skyMaskImg.cols - 1);
            } else {
                skyMaskPixelAhead = skyMaskImg.at<uchar>(rMaskIndex + rIndex + boundaryGap, cMaskIndex + cIndex + boundaryGap);
            }

            if (skyMaskPixelAhead < 127) {
                isTransition = true;
            }

            bool isBlackPixel = true;
            Vec3b resultImgItem;

            if (rIndex + imgAreaGap >= resultImg.rows) {
                // 初始下边界
                resultImgItem = resultImg.at<Vec3b>(resultImg.rows - 1, cIndex);
            } else if (cIndex + imgAreaGap >= resultImg.cols) {
                // 初始右边界
                resultImgItem = resultImg.at<Vec3b>(rIndex, resultImg.cols - 1);
            } else if (rIndex <= imgAreaGap) {
                // 初始上边界
                resultImgItem = resultImg.at<Vec3b>(0, cIndex);
            } else if (cIndex <= imgAreaGap) {
                // 初始左边界
                resultImgItem = resultImg.at<Vec3b>(rIndex, 0);
            } else {
                // 初始星空、地景边界
                resultImgItem = resultImg.at<Vec3b>(rIndex + imgAreaGap, cIndex + imgAreaGap);
            }

            for (int i = 0; i < 3; i ++) {
                if (resultImgItem[i] > 0) {
                    isBlackPixel = false;
                }
            }

            int new_x = cIndex;
            int new_y = rIndex;
            if (isTransition) {
                // 在过渡带中，直接使用target的像素
                if (new_x >= 0 && new_x < resultImg.cols && new_y >= 0 && new_y < resultImg.rows) {
                    this->imagePart.at<Vec3b>(new_y, new_x) += (queryImgTransform.at<Vec3b>(rMaskIndex + new_y, cMaskIndex + new_x) * 1.0   / imageCount);
                }
            } else {
                if (isBlackPixel) {
                    // 不在过渡带，但是是黑像素，则使用target的像素
                    if (new_x >= 0 && new_x < resultImg.cols && new_y >= 0 && new_y < resultImg.rows) {
                        this->imagePart.at<Vec3b>(new_y, new_x) += (queryImgTransform.at<Vec3b>(rMaskIndex + new_y, cMaskIndex + new_x) * 1.0   / imageCount);
                    }
                } else {
                    // 不在过渡带，无黑像素，使用已经配准之后的图像
                    this->imagePart.at<Vec3b>(rIndex, cIndex) += (resultImg.at<Vec3b>(rIndex, cIndex) * 1.0 / imageCount);
                }
            }
        }
    }
}


int StarImagePart::getAtParentStartRowIndex() const {
    return atParentStartRowIndex;
}

int StarImagePart::getAtParentEndRowIndex() const {
    return atParentEndRowIndex;
}

int StarImagePart::getAtParentStartColumnIndex() const {
    return atParentStartColumnIndex;
}

int StarImagePart::getAtParentEndColumnIndex() const {
    return atParentEndColumnIndex;
}

/**
 * 用于 对已经计算出 射影偏移的图片信息的 累加
 * @param imageMat
 */
void StarImagePart::addUpStarImagePart(Mat_<Vec3b> imageMat) {
    this->imagePart += imageMat;
}

int StarImagePart::getRowPartIndex() const {
    return rowPartIndex;
}

int StarImagePart::getColumnPartIndex() const {
    return columnPartIndex;
}

int StarImagePart::getAlignStartRowIndex() const {
    return alignStartRowIndex;
}

int StarImagePart::getAlignEndRowIndex() const {
    return alignEndRowIndex;
}

int StarImagePart::getAlignStartColumnIndex() const {
    return alignStartColumnIndex;
}

int StarImagePart::getAlignEndColumnIndex() const {
    return alignEndColumnIndex;
}
