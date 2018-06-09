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
StarImagePart::StarImagePart(const Mat parentMat, int atParentStartRowIndex, int atParentEndRowIndex,
              int atParentStartColumnIndex, int atParentEndColumnIndex, int rowPartIndex, int columnPartIndex) {
    // [atParentStartRowIndex, atParentEndRowIndex)  [atParentStartColumnIndex, atParentEndColumnIndex)

    this->imagePart = parentMat(Range(atParentStartRowIndex, atParentEndRowIndex),
                                Range(atParentStartColumnIndex, atParentEndColumnIndex));
    this->rowPartIndex = rowPartIndex;
    this->columnPartIndex = columnPartIndex;

    this->atParentStartRowIndex = atParentStartRowIndex;
    this->atParentEndRowIndex = atParentEndRowIndex;
    this->atParentStartColumnIndex = atParentStartColumnIndex;
    this->atParentEndColumnIndex = atParentEndColumnIndex;
}


/**
 *
 * @return
 */
Mat StarImagePart::getImage() {
    return this->imagePart;
}

/**
 * 设置Image 中每一部分 Image Mat的信息，一般是存储 经过射影变换的 图形矩阵
 * @param imageMat
 */
void StarImagePart::setImage(Mat_<Vec3b> imageMat) {
//    cout << std::to_string(this->imagePart.rows) << "\t" << std::to_string(this->imagePart.cols) << endl;
    this->imagePart = imageMat;
//    cout << std::to_string(imageMat.rows) << "\t" << std::to_string(imageMat.cols) << endl;
}


/**
 * 前提 this->imagePart 初始为0矩阵
 * @param resultImg 当前被配准的 图片部分
 * @param targetImg 作为配准基准的 图像目标区域
 * @param imageCount 一共有多少张图片需要被配准
 */
void StarImagePart::addImagePixelValue(Mat& resultImg, Mat& targetImg,
                                       Mat& queryImgTransform, Mat& skyMaskImg, int imageCount) {


    this->imagePart += (resultImg / imageCount * 1.0);
//    this->imagePart += resultImg;

    // 注意的是使用方向数组的概念
    static const int dx[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    static const int dy[] = {0, 0, -1, 1, -1, 1, -1, 1};

    // 取出当前mask起始点的位置
    int rMaskIndex = this->getRowPartIndex() * this->getImage().rows;
    int cMaskIndex = this->getColumnPartIndex() * this->getImage().cols;

    for (int rIndex = 0; rIndex < this->imagePart.rows; rIndex ++) {
        for (int cIndex = 0; cIndex < this->imagePart.cols; cIndex ++) {

//            this->imagePart.at<Vec3b>(rIndex, cIndex) = resultImg.at<Vec3b>(rIndex, cIndex);

            int skyMaskPixel = skyMaskImg.at<uchar>(rMaskIndex + rIndex, cMaskIndex + cIndex);
            if (skyMaskPixel == 0) {
                continue;
            }

            bool isBlackPixel = true;
            Vec3b resultImgItem = resultImg.at<Vec3b>(rIndex, cIndex);
            for (int i = 0; i < 3; i ++) {
                if (resultImgItem[i] > 0) {
                    isBlackPixel = false;
                }
            }

            if (isBlackPixel) {
//                cout << "rIndex: " << rIndex << "\t" << "cIndex: " << cIndex << endl;
//                this->imagePart.at<Vec3b>(rIndex, cIndex) = (queryImgTransform.at<Vec3b>(rMaskIndex + rIndex, cMaskIndex + cIndex) * 1.0   / imageCount);
//                this->imagePart.at<Vec3b>(rIndex, cIndex) = queryImgTransform.at<Vec3b>(rMaskIndex + rIndex, cMaskIndex + cIndex);

                // 在检测出的黑点四周填充原图像的像素，分割线得以覆盖
                for (int i = 0; i < 8; i ++) {
                    int new_x = dx[i] + cIndex;
                    int new_y = dy[i] + rIndex;
                    if (new_x >= 0 && new_x < resultImg.cols && new_y >= 0 && new_y < resultImg.rows) {
                        this->imagePart.at<Vec3b>(new_y, new_x) = (queryImgTransform.at<Vec3b>(rMaskIndex + new_y, cMaskIndex + new_x) * 1.0   / imageCount);
                    }
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
