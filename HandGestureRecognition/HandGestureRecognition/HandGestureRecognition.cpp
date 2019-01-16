#include "HandGestureRecognition.h"

#define NR_OF_CLASSES_BAYES 24
#define IMG_HEIGHT 240
#define IMG_WIDTH 320
#define NR_BINS 5

using namespace cv;
using namespace std;

const int nrDimBayes = IMG_HEIGHT * IMG_WIDTH;
const int gray_interval_length = 255 / NR_BINS;
char letters[] = { 'A', 'B', 'C', 'D', 'E',
                    'F', 'G', 'H', 'I', 'K',
                    'L', 'M', 'N', 'O', 'P',
                    'Q', 'R', 'S', 'T', 'U',
                    'V', 'W', 'X', 'Y' };

double priors[NR_OF_CLASSES_BAYES];
Mat likelihood(NR_OF_CLASSES_BAYES * NR_BINS, nrDimBayes, CV_64FC1, Scalar(0.0));

char classifyBayes(Mat img) {
    cout << "classifying" << endl;
    double class_logs[NR_OF_CLASSES_BAYES];

    int c, i, j, k;
    double probability = 0;

    for (c = 0; c < NR_OF_CLASSES_BAYES; c++) {
        class_logs[c] = log(priors[c]);
        cout << letters[0] << " prior: " << class_logs[0] << endl;
        for (i = 0; i < IMG_HEIGHT; i++) {
            for (j = 0; j < IMG_WIDTH; j++) {
                for (k = 0; k < NR_BINS; k++) {
                    if (img.at<uchar>(i, j) > k*gray_interval_length &&
                        img.at<uchar>(i, j) < (k + 1)*gray_interval_length) {
                        probability = likelihood.at<double>(c*NR_BINS + k, i*IMG_WIDTH + j);
                        break;
                    }
                }
                class_logs[c] += log(probability);
            }
        }
    }

    double max = class_logs[0];
    int maxc = 0;
    cout << letters[0] << " probability: " << class_logs[0] << endl;

    for (c = 1; c < NR_OF_CLASSES_BAYES; c++) {
        cout << letters[c] << " probability: " << class_logs[c] << endl;
        if (class_logs[c] > max) {
            max = class_logs[c];
            maxc = c;
        }
    }

    return letters[maxc];
}

bool is_skin_color_rgb(Vec3b pixel) {
    int B = pixel[0];
    int G = pixel[1];
    int R = pixel[2];

    double r = (double)R / (double)(R + G + B);
    double g = (double)G / (double)(R + G + B);

    bool is_skin_color = true;

    if (!(0.3 <= r && r <= 0.7)) {
        is_skin_color = false;
    }

    if (!(0.25 <= g && g <= 0.4245)) {
        is_skin_color = false;
    }

    if (!(R > G)) {
        is_skin_color = false;
    }

    if (!(R > B)) {
        is_skin_color = false;
    }

    return is_skin_color;
}

bool is_skin_color_crcb(Vec3b pixel) {
    float Cr = pixel[1];
    float Cb = pixel[2];

    bool is_skin_color = true;

    if (!(77 <= Cb && Cb <= 127)) {
        is_skin_color = false;
    }

    if (!(133 <= Cr && Cr <= 173)) {
        is_skin_color = false;
    }

    return is_skin_color;
}

void resizeImage(Mat& img, int height = 0, int width = 0) {
    int h = img.rows;
    int w = img.cols;
    float r;

    if (width == 0) {
        r = height / (float)h;
        w = (int)w*r;
        h = height;
    }
    else {
        r = width / (float)w;
        h = (int)h*r;
        w = width;
    }

    resize(img, img, Size(w, h));
}

void trainNaiveBayes() {
    int c, i, j, k, n, d;
    char fname[1000];

    int classindex[NR_BINS];

    int totalindex = 0;
    Mat img;
    Mat img_orig;
    Mat img_ycrcb;
    Mat img_gray;
    int margin;
    Rect roi_rect;

    Vec3b black(0, 0, 0);
    Vec3b white(255, 255, 255);

    cout << "TRAINING" << endl;
    vector<String> filenames;
    vector<Mat> images;
    int count;
    c = 0;
    //Training algorithm
    for (c = 0; c < NR_OF_CLASSES_BAYES; c++) {
        sprintf_s(fname, "hand-gestures\\train\\%c\\*.jpg", letters[c]);
        glob(fname, filenames, false);
        count = filenames.size(); //number of png files in images folder
        cout << letters[c] << endl;

        for (k = 0; k < NR_BINS; k++) {
            classindex[k] = 0;
        }

        for (n = 0; n < count; n++) {
            img_orig = imread(filenames[n], CV_LOAD_IMAGE_COLOR);

            pyrDown(img_orig, img_orig);

            //resize bigger images
            if (img_orig.rows != 240 || img_orig.cols != 320) {
                if (img_orig.rows < img_orig.cols) {
                    resizeImage(img_orig, 240);  //resize fixed height
                    //crop width
                    margin = (img_orig.cols - 320) / 2;
                    roi_rect = Rect(margin, 0, 320, img_orig.rows);
                    img = img_orig(roi_rect);
                }
                else {
                    resizeImage(img_orig, 0, 320);
                    //crop height
                    margin = (img_orig.rows - 240) / 2;
                    roi_rect = Rect(0, margin, img_orig.cols, 240);
                    img = img_orig(roi_rect);
                }
            }
            else {
                img_orig.copyTo(img);
            }

            cvtColor(img, img_ycrcb, CV_BGR2YCrCb);
            cvtColor(img, img_gray, CV_BGR2GRAY);

            //Binarization (thresholding)
            for (i = 0; i < img.rows; i++) {
                for (j = 0; j < img.cols; j++) {
                    if (!is_skin_color_rgb(img.at<Vec3b>(i, j))) {
                        img_gray.at<uchar>(i, j) = 0;
                    }
                    if (!is_skin_color_crcb(img_ycrcb.at<Vec3b>(i, j))) {
                        img_gray.at<uchar>(i, j) = 0;
                    }
                }
            }

            //Compute likelihood sum
            for (i = 0; i < img_gray.rows; i++) {
                for (j = 0; j < img_gray.cols; j++) {
                    for (k = 0; k < NR_BINS; k++) {
                        if (img_gray.at<uchar>(i, j) > k*gray_interval_length &&
                            img_gray.at<uchar>(i, j) < (k + 1)*gray_interval_length) {
                            likelihood.at<double>(c*NR_BINS + k, i*img_gray.cols + j)++;
                            classindex[k]++;
                        }
                    }
                }
            }

            totalindex++;

            //Add total nr in prior probability
            for (k = 0; k < NR_BINS; k++) {
                priors[c] = (double)classindex[k];
            }

            //Divide likelihood sum with total number of class elements
            for (d = 0; d < nrDimBayes; d++) {
                for (k = 0; k < NR_BINS; k++) {
                    if (likelihood.at<double>(c*NR_BINS + k, d) == 0) {
                        likelihood.at<double>(c*NR_BINS + k, d) = 1e-5;
                    }
                    else {
                        likelihood.at<double>(c*NR_BINS + k, d) /= (double)classindex[k];
                    }
                    //printf("%c %lf\n", letters[c], likelihood.at<double>(c, d));
                }
            }
        }
    }

    //Divide priors with total number of elements
    for (c = 0; c < NR_OF_CLASSES_BAYES; c++) {
        priors[c] /= (double)totalindex;
        //printf("%lf\n", priors[c]);
    }

    //totalindex = 0;
    //int classified = 0;
    //Mat confusion_mat(NR_OF_CLASSES_BAYES, NR_OF_CLASSES_BAYES, CV_32SC1, Scalar(0));

    //cout << "CLASSIFYING" << endl;
    //vector<String> filenames;
    //vector<Mat> images;
    //int count;

    ////Classification algorithm
    //for (c = 0; c < NR_OF_CLASSES_BAYES; c++) {
    //    sprintf_s(fname, "hand-gestures\\test\\%c\\*.png", letters[c]);
    //    glob(fname, filenames, false);
    //    count = filenames.size(); //number of png files in images folder
    //    for (classindex = 0; classindex < count; classindex++) {
    //        img_orig = imread(filenames[classindex], CV_LOAD_IMAGE_COLOR);

    //        pyrDown(img_orig, img_orig);

    //        //resize bigger images
    //        if (img_orig.rows > 240 || img_orig.cols > 320) {
    //            if (img_orig.rows < img_orig.cols) {
    //                resizeImage(img_orig, 240);  //resize fixed height
    //                                             //crop width
    //                margin = (img_orig.cols - 320) / 2;
    //                roi_rect = Rect(margin, 0, 320, img_orig.rows);
    //                img = img_orig(roi_rect);
    //            }
    //            else {
    //                resizeImage(img_orig, 0, 320);
    //                //crop height
    //                margin = (img_orig.rows - 240) / 2;
    //                roi_rect = Rect(0, margin, img_orig.cols, 240);
    //                img = img_orig(roi_rect);
    //            }
    //        }
    //        else {
    //            resizeImage(img_orig, 240, 320);
    //            img_orig.copyTo(img);
    //        }

    //        cv::cvtColor(img, img_ycrcb, CV_BGR2YCrCb);
    //        cv::cvtColor(img, img_gray, CV_BGR2GRAY);

    //        //Binarization (thresholding)
    //        for (i = 0; i < img.rows; i++) {
    //            for (j = 0; j < img.cols; j++) {
    //                if (!is_skin_color_rgb(img.at<Vec3b>(i, j))) {
    //                    img_gray.at<uchar>(i, j) = 0;
    //                }
    //                else {
    //                    if (!is_skin_color_crcb(img_ycrcb.at<Vec3b>(i, j))) {
    //                        img_gray.at<uchar>(i, j) = 0;
    //                    }
    //                    else {
    //                        img_gray.at<uchar>(i, j) = 255;
    //                    }
    //                }
    //            }
    //        }

    //        classified = classifyBayes(img, priors, likelihood);

    //        //printf("Predicted class: %d\nActual class: %d\n\n", classified, c);
    //        confusion_mat.at<int>(c, classified)++;

    //        classindex++;
    //        totalindex++;
    //    }
    //}

    //for (i = 0; i < NR_OF_CLASSES_BAYES; i++) {
    //    for (int j = 0; j < NR_OF_CLASSES_BAYES; j++) {
    //        printf("%5d ", confusion_mat.at<int>(i, j));
    //    }
    //    printf("\n");
    //}

    //int sum_main_diag = 0;
    //int sum_all_elem = 0;
    //for (i = 0; i < NR_OF_CLASSES_BAYES; i++) {
    //    for (int j = 0; j < NR_OF_CLASSES_BAYES; j++) {
    //        sum_all_elem += confusion_mat.at<int>(i, j);
    //        if (i == j) {
    //            sum_main_diag += confusion_mat.at<int>(i, j);
    //        }
    //    }
    //}

    //double ACC = (double)sum_main_diag / (double)sum_all_elem;
    //printf("ACC: %lf\n", ACC);

    //getchar();
    //getchar();
}