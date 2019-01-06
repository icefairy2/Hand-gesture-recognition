#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

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
    int Cr = pixel[1];
    int Cb = pixel[2];

    bool is_skin_color = true;

    if (!(77 <= Cb && Cb <= 127)) {
        is_skin_color = false;
    }

    if (!(133 <= Cr && Cr <= 173)) {
        is_skin_color = false;
    }

    return is_skin_color;
}

//Mat fit_gaussian_model(Mat image) {
//    int i, j;
//    Mat image_ycrcb;
//    Mat gray = Mat::zeros(image.rows, image.cols, CV_8UC1);
//    Vec3b black(0, 0, 0);
//
//    int Cri, Cbi;
//    int N = 0;
//    Mat xi, mean;
//    xi = Mat(1, 2, CV_64FC1);
//    double Cr = 0, Cb = 0;
//    double variance = 0;
//    double P;
//
//    cvtColor(image, image_ycrcb, CV_BGR2YCrCb);
//
//    for (i = 0; i < image.rows; i++) {
//        for (j = 0; j < image.cols; j++) {
//            if (image.at<Vec3b>(i, j) != black) {
//                Cri = image_ycrcb.at<Vec3b>(i, j)[1];
//                Cbi = image_ycrcb.at<Vec3b>(i, j)[2];
//                Cr += Cri;
//                Cb += Cbi;
//                N++;
//            }
//        }
//    }
//
//    Cr /= (double)N;
//    Cb /= (double)N;
//    mean = Mat(1, 2, CV_64FC1);
//    mean.at<double>(0) = Cb;
//    mean.at<double>(1) = Cr;
//    Mat temp;
//
//    for (i = 0; i < image.rows; i++) {
//        for (j = 0; j < image.cols; j++) {
//            if (image.at<Vec3b>(i, j) != black) {
//                Cri = image_ycrcb.at<Vec3b>(i, j)[1];
//                Cbi = image_ycrcb.at<Vec3b>(i, j)[2];
//                xi.at<double>(0) = Cbi;
//                xi.at<double>(1) = Cri;
//                temp = (xi - mean)*(xi - mean).t();
//                variance += temp.at<double>(0);
//            }
//        }
//    }
//
//    variance /= (double)(N - 1);
//
//    for (i = 0; i < image.rows; i++) {
//        for (j = 0; j < image.cols; j++) {
//            if (image.at<Vec3b>(i, j) != black) {
//                Cri = image_ycrcb.at<Vec3b>(i, j)[1];
//                Cbi = image_ycrcb.at<Vec3b>(i, j)[2];
//                xi.at<double>(0) = Cbi;
//                xi.at<double>(1) = Cri;
//                temp = (xi - mean)/variance*(xi - mean).t();
//                P = 1.0 / (2 * CV_PI*sqrt(abs(variance))) * exp(-0.5* temp.at<double>(0));
//            }
//        }
//    }
//
//    return gray;
//}

int main()
{
    char c;

    int i, j;
    Mat frame;
    Mat res_frame;
    Mat res_ycrcb;

    Vec3b black(0, 0, 0);

    VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
    if (!cap.isOpened()) // openenig the video device failed
    {
        printf("Cannot open video capture device.\n");
        return 0;
    }

    //cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    // Display window
    const char* WIN_SRC = "Src"; //window for the source frame
    namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_SRC, 0, 0);

    // Result window
    const char* WIN_RES = "Res"; //window for the result frame
    namedWindow(WIN_RES, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_RES, 0, 0);

    while(true)
    {
        cap >> frame; // get a new frame from camera
        if (frame.empty())
        {
            printf("End of the video file\n");
            break;
        }

        imshow(WIN_SRC, frame);

        //res_frame = Mat(frame.size(), CV_8UC3);
        frame.copyTo(res_frame);

        cvtColor(res_frame, res_ycrcb, CV_BGR2YCrCb);

        //simple threshold model
        for (i = 0; i < res_frame.rows; i++) {
            for (j = 0; j < res_frame.cols; j++) {
                if (!is_skin_color_rgb(res_frame.at<Vec3b>(i, j))) {
                    res_frame.at<Vec3b>(i, j) = black;
                }
                if (!is_skin_color_crcb(res_ycrcb.at<Vec3b>(i, j))) {
                    res_frame.at<Vec3b>(i, j) = black;
                }
            }
        }

        //gray = fit_gaussian_model(res);

        imshow(WIN_RES, res_frame);

        c = cvWaitKey(10);  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished");
            break;  //ESC pressed
        }
    }

    getchar();
    return 0;
}