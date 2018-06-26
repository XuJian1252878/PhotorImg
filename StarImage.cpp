//
// Created by 许舰 on 2018/1/11.
//

#include "StarImage.h"

StarImage::StarImage() {
}

/**
 *
 * @param image
 * @param rowParts
 * @param columnParts
 * @param isClone  StarImage中的imagePart是不是 原始oriImg 的引用, true 是；false 不是
 */
StarImage::StarImage(Mat image, int rowParts, int columnParts, bool isClone) {
    this->image = image;
    this->rowParts = rowParts;
    this->columnParts = columnParts;

    // 初始化时就对每一个图片进行分块
    splitImage(isClone);
}


/**
 *
 */
void StarImage::splitImage(bool isClone) {

    this->initStarImageParts();

    int rowStep = this->image.rows / this->rowParts;  // y step
    int columnStep = this->image.cols / this->columnParts;  // x step

    for (int rIndex = 0; rIndex < this->rowParts; rIndex ++) {

        int atParentStartRowIndex = rIndex * rowStep;
        int atParentEndRowIndex = 0;
        if (rIndex == this->rowParts - 1) {
            atParentEndRowIndex = this->image.rows;
        } else {
            atParentEndRowIndex = (rIndex + 1) * rowStep;
        }

        for(int cIndex = 0; cIndex < this->columnParts; cIndex ++) {

            // 上 左 下 右
            int directions[] = {1, 1, 1, 1};
            // 纵向范围确定
            if (rIndex == 0) {
                directions[0] = 0;
            } else if (rIndex == this->rowParts - 1) {
                directions[2] = 0;
            }
            // 横向范围确定
            if (cIndex == 0) {
                directions[1] = 0;
            } else if (cIndex == this->columnParts - 1) {
                directions[3] = 0;
            }

            // 确定最终结果拼接范围
            int atParentStartColumnIndex = cIndex * columnStep;
            int atParentEndColumnIndex = 0;
            if (cIndex == this->columnParts - 1) {
                atParentEndColumnIndex = this->image.cols;
            } else {
                atParentEndColumnIndex = (cIndex + 1) * columnStep;
            }

            // 确定对齐范围
            int alignStartRowIndex = atParentStartRowIndex - directions[0] * rowStep / 2;
            int alignEndRowIndex = atParentEndRowIndex + directions[2] * rowStep / 2;
            int alignStartColumnIndex = atParentStartColumnIndex - directions[1] * columnStep / 2;
            int alignEndColumnIndex = atParentEndColumnIndex + directions[3] * columnStep / 2;

            StarImagePart starImagePart = StarImagePart(this->image,
                                                        atParentStartRowIndex, atParentEndRowIndex,
                                                        atParentStartColumnIndex, atParentEndColumnIndex,
                                                        rIndex, cIndex,
                                                        alignStartRowIndex, alignEndRowIndex,
                                                        alignStartColumnIndex, alignEndColumnIndex,
                                                        isClone);

            this->starImageParts[rIndex].push_back(starImagePart);

        }
    }
}

/**
 *
 * @param rowPartIndex
 * @param columnPartIndex
 * @return
 */
StarImagePart& StarImage::getStarImagePart(int rowPartIndex, int columnPartIndex) {
    return this->starImageParts[rowPartIndex][columnPartIndex];
}

/**
 * 设置Image 中每一部分 Image Mat的信息，一般是存储 经过射影变换的 图形矩阵
 * @param rowPartIndex
 * @param columnPartIndex
 * @param imageMat
 */
void StarImage::setStarImagePart(int rowPartIndex, int columnPartIndex, Mat_<Vec3b> imageMat) {
    this->starImageParts[rowPartIndex][columnPartIndex].setImage(imageMat);
}

/**
 * 对StarImage中的各部分的Image Mat进行整合，主要是对配准之后的每一个小块进行整合
 * @return
 */
Mat StarImage::mergeStarImageParts() {
    Mat_<Vec3b> resultImage = Mat::zeros(this->image.rows, this->image.cols, this->image.type());
    for(int rPartIndex = 0; rPartIndex < this->rowParts; rPartIndex ++) {
        for(int cPartIndex = 0; cPartIndex < this->columnParts; cPartIndex ++) {

            StarImagePart tmpPart = this->starImageParts[rPartIndex][cPartIndex];

//            string sfile = "/Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/img/merge_part/"  + std::to_string(rPartIndex) + "_" + std::to_string(cPartIndex) + ".jpg";
//            imwrite(sfile, tmpPart.getImage());

            int atParentStartRowIndex = tmpPart.getAtParentStartRowIndex();
            int atParentEndRowIndex = tmpPart.getAtParentEndRowIndex();
            int atParentStartColumnIndex = tmpPart.getAtParentStartColumnIndex();
            int atParentEndColumnIndex = tmpPart.getAtParentEndColumnIndex();

            int alignStartRowIndex = tmpPart.getAlignStartRowIndex();
            int alignEndRowIndex = tmpPart.getAlignEndRowIndex();
            int alignStartColumnIndex = tmpPart.getAlignStartColumnIndex();
            int alignEndColumnIndex = tmpPart.getAlignEndColumnIndex();

            Mat_<Vec3b> tmpAlignImage = tmpPart.getImage();

            int rowStart = atParentStartRowIndex - alignStartRowIndex, rowEnd = atParentEndRowIndex - alignStartRowIndex;
            int columnStart = atParentStartColumnIndex - alignStartColumnIndex, columnEnd = atParentEndColumnIndex - alignStartColumnIndex;

            Mat_<Vec3b> tmpImage = tmpAlignImage(Range(rowStart, rowEnd),
                    Range(columnStart, columnEnd));

            for (int i = atParentStartRowIndex, it = 0; i < atParentEndRowIndex; i ++, it ++) {
                for (int j = atParentStartColumnIndex, jt = 0; j < atParentEndColumnIndex; j ++, jt ++) {
//                        resultImage.at<cv::Vec3b>(i, j) = tmpPart.getImage().at<cv::Vec3b>(it, jt);
                    resultImage(i, j) = tmpImage(it, jt);
//                    cout << "merge: " << std::to_string(i) << "\t" << std::to_string(j) << endl;
                }
            }

        }
    }

    return resultImage;
}

/**
 * 获得原始的图像Mat
 * @return
 */
Mat StarImage::getImage(){
    return this->image;
}

void StarImage::initStarImageParts(){
    // 如果之前进行过分块操作，那么将之前的分块数据清除。
    if (this->starImageParts.size() > 0) {
        for (int index = 0; index < this->starImageParts.size(); index ++) {
            this->starImageParts[index].clear();
        }
    }
    this->starImageParts.clear();

    for (int index = 0; index < rowParts; index ++) {
        std::vector<StarImagePart> tmp = std::vector<StarImagePart>();
        this->starImageParts.push_back(tmp);
    }
}

void StarImage::addUpStarImagePart(int rowPartIndex, int columnPartIndex, Mat_<Vec3b> imageMat) {
    this->starImageParts[rowPartIndex][columnPartIndex].addUpStarImagePart(imageMat);
}
