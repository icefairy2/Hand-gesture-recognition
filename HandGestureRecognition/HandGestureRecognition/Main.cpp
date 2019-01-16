#include "Main.h"
#include "HandGestureRecognition.h"
#include "Roi.h"
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

/// Global Variables  ///

//Most important ones, all processing is done on these
Mat img;
Mat bw_img;
Rect bounding_rect;

int avgColor[NSAMPLES][3];
int c_lower[NSAMPLES][3];
int c_upper[NSAMPLES][3];

//In case we want to save a video
VideoWriter outputVideo;

vector <Palm_ROI> roi;
vector <Mat> bw_list;

int nr_created_squares = 0;
const int SQUARE_LEN = 20;
const char* WIN_CAM = "Hand gesture";
const char* WIN_HIST = "Histogram";
const char* WIN_BW = "Black and white";
/// end global variables ///

void printText(Mat& src, string text) {
    int fontFace = FONT_HERSHEY_PLAIN;
    putText(src, text, Point(src.cols / 3, src.rows / 10), fontFace, 1.2f, Scalar(200, 0, 0), 2);
}

//Will save the position of the click as the center of the ROI - square to be analyzed for color
void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    if (nr_created_squares < NSAMPLES) {
        if (event == EVENT_LBUTTONDOWN)
        {
            if (x - SQUARE_LEN / 2 > 0 && y - SQUARE_LEN / 2 > 0) {
                if (x + SQUARE_LEN / 2 < img.cols && y + SQUARE_LEN / 2 < img.rows) {
                    roi.push_back(Palm_ROI(Point(x, y), SQUARE_LEN));
                    nr_created_squares++;
                }
            }
        }
    }
    else {
        setMouseCallback(WIN_CAM, NULL, NULL);
    }
}

void createSquares(VideoCapture capture) {
    char c;

    setMouseCallback(WIN_CAM, CallBackFunc, NULL);

    while (nr_created_squares < NSAMPLES) {
        capture >> img;
        flip(img, img, 1);

        for (int j = 0; j < roi.size(); j++) {
            roi[j].draw_rectangle(img);
        }

        imshow(WIN_CAM, img);
        if (OUTPUT_VIDEO) {
            outputVideo << img;
        }

        c = cvWaitKey(10);  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished");
            break;  //ESC pressed
        }
    }
}

bool isStdDevSmall() {
    Mat horiz_concat;
    vector<int>hue_vals;
    Scalar mean, stdv;
    Rect r;

    if (NSAMPLES > 0) {
        r = Rect(roi[0].upper_corner.x, roi[0].upper_corner.y, SQUARE_LEN, SQUARE_LEN);
        horiz_concat = img(r);
    }
    for (int k = 1; k < NSAMPLES; k++) {
        r = Rect(roi[k].upper_corner.x, roi[k].upper_corner.y, SQUARE_LEN, SQUARE_LEN);
        hconcat(horiz_concat, img(r), horiz_concat);
    }

    meanStdDev(horiz_concat, mean, stdv);
    cout << stdv << endl;

    if (stdv[2] < HUE_DEV_THR) {
        return true;
    }
    else {
        return false;
    }
}

void waitForPalmCover(VideoCapture& capture) {
    char c;
    bool palm_covers = false;

    for (int i = 0; i < 300; i++) {
        //Capture and flip the next frame
        capture >> img;
        //Flip the image as if it was a mirror
        flip(img, img, 1);

        cvtColor(img, img, CV_BGR2HSV);
        palm_covers = isStdDevSmall();
        cvtColor(img, img, CV_HSV2BGR);
        for (int j = 0; j < NSAMPLES; j++) {
            if (palm_covers) {
                roi[j].color = Scalar(255, 255, 255); //white
            }
            roi[j].draw_rectangle(img);
        }

        string imgText = string("Cover rectangles with palm");
        printText(img, imgText);

        imshow(WIN_CAM, img);
        if (OUTPUT_VIDEO) {
            outputVideo << img;
        }

        if (palm_covers) {
            break;
        }

        c = cvWaitKey(10);  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished");
            break;  //ESC pressed
        }
    }
}

int getMedian(vector<int> val) {
    int median;
    size_t size = val.size();
    sort(val.begin(), val.end());
    if (size % 2 == 0) {
        median = val[size / 2 - 1];
    }
    else {
        median = val[size / 2];
    }
    return median;
}

void getAvgColor(Palm_ROI roi, int avg[3]) {
    vector<int>hm;
    vector<int>sm;
    vector<int>lm;

    Mat r = img(Rect(roi.upper_corner.x, roi.upper_corner.y, SQUARE_LEN, SQUARE_LEN));

    for (int i = 2; i < r.rows - 2; i++) {
        for (int j = 2; j < r.cols - 2; j++) {
            hm.push_back(r.at<Vec3b>(i, j)[0]);
            sm.push_back(r.at<Vec3b>(i, j)[1]);
            lm.push_back(r.at<Vec3b>(i, j)[2]);
        }
    }
    avg[0] = getMedian(hm);
    avg[1] = getMedian(sm);
    avg[2] = getMedian(lm);
}

//Finds average color on all points within the squares
void average(VideoCapture& capture) {
    char c;
    capture >> img;
    flip(img, img, 1);
    for (int i = 0; i < 30; i++) {
        capture >> img;
        flip(img, img, 1);

        cvtColor(img, img, CV_BGR2HLS);
        for (int j = 0; j < NSAMPLES; j++) {
            getAvgColor(roi[j], avgColor[j]);
            roi[j].draw_rectangle(img);
        }

        cvtColor(img, img, CV_HLS2BGR);
        string imgText = string("Finding average color of hand");
        printText(img, imgText);
        imshow(WIN_CAM, img);

        c = cvWaitKey(10);  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished");
            break;  //ESC pressed
        }
    }
}

void hsl_histogram(Mat& img) {
    vector<Mat> hsl_planes;
    int histSize = 256;

    float range[] = { 0, 256 };
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat h_hist, s_hist, l_hist;

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    split(img, hsl_planes);

    calcHist(&hsl_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&hsl_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&hsl_planes[2], 1, 0, Mat(), l_hist, 1, &histSize, &histRange, uniform, accumulate);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(s_hist, s_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(l_hist, l_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    /// Draw for each channel
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(h_hist.at<float>(i - 1))),
            Point(bin_w*(i), hist_h - cvRound(h_hist.at<float>(i))),
            Scalar(255, 255, 255), 2, 8, 0);
        line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(s_hist.at<float>(i - 1))),
            Point(bin_w*(i), hist_h - cvRound(s_hist.at<float>(i))),
            Scalar(0, 255, 255), 2, 8, 0);
        line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(l_hist.at<float>(i - 1))),
            Point(bin_w*(i), hist_h - cvRound(l_hist.at<float>(i))),
            Scalar(255, 0, 255), 2, 8, 0);
    }

    string imgText = string("H - purple, S - yellow,L - white");
    printText(histImage, imgText);
    imshow(WIN_HIST, histImage);
}

void normalizeColors() {
    // normalize all boundaries so that 
    // threshold is whithin 0-255
    for (int i = 0; i < NSAMPLES; i++) {
        c_lower[i][0] = 7;
        c_upper[i][0] = 12;

        c_lower[i][1] = 30;
        c_upper[i][1] = 40;

        c_lower[i][2] = 80;
        c_upper[i][2] = 80;

        //HSL in 8 bit
        //H is h/2
        if ((avgColor[i][0] - c_lower[i][0]) < 0) {
            c_lower[i][0] = avgColor[i][0];
        }
        //S is S*255
        if ((avgColor[i][1] - c_lower[i][1]) < 0) {
            c_lower[i][1] = avgColor[i][1];
        }
        //L is L*255
        if ((avgColor[i][2] - c_lower[i][2]) < 0) {
            c_lower[i][2] = avgColor[i][2];
        }

        if ((avgColor[i][0] + c_upper[i][0]) > 255) {
            c_upper[i][0] = 255 - avgColor[i][0];
        }
        if ((avgColor[i][1] + c_upper[i][1]) > 255) {
            c_upper[i][1] = 255 - avgColor[i][1];
        }
        if ((avgColor[i][2] + c_upper[i][2]) > 255) {
            c_upper[i][2] = 255 - avgColor[i][2];
        }
    }
}

void thresholdBW(Mat& img) {
    Scalar lowerBound;
    Scalar upperBound;
    Mat foo;

    normalizeColors();
    for (int i = 0; i < NSAMPLES; i++) {
        lowerBound = Scalar(avgColor[i][0] - c_lower[i][0], avgColor[i][1] - c_lower[i][1], avgColor[i][2] - c_lower[i][2]);
        upperBound = Scalar(avgColor[i][0] + c_upper[i][0], avgColor[i][1] + c_upper[i][1], avgColor[i][2] + c_upper[i][2]);
        bw_list.push_back(Mat(img.rows, img.cols, CV_8U));
        inRange(img, lowerBound, upperBound, bw_list[i]);
    }
    bw_list[0].copyTo(bw_img);
    for (int i = 1; i < NSAMPLES; i++) {
        bw_img += bw_list[i];
    }
    medianBlur(bw_img, bw_img, 7);
    imshow(WIN_BW, bw_img);
}

void takeHandImage(Mat& img) {
    Mat bw_high_res;
    int largest_area = 0;
    int largest_contour_index = 0;
    vector<vector<Point>> contours; // Vector for storing contour
    vector<Vec4i> hierarchy;

    pyrUp(bw_img, bw_high_res);
    findContours(bw_high_res, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

    largest_area = 0;
    largest_contour_index = 0;
    for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
    {
        double a = contourArea(contours[i], false);  //  Find the area of contour
        if (a > largest_area) {
            largest_area = a;
            largest_contour_index = i;                //Store the index of largest contour
            bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
        }
    }

    Scalar color(255, 255, 255);
    drawContours(img, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
    rectangle(img, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);

    //imshow(WIN_BW, hand_frame);
}

int main() {
    //Pressed character
    char c;

    //Main camera image capture
    VideoCapture capture(0); // open the deafult camera (i.e. the built in web cam)
    if (!capture.isOpened()) // openenig the video device failed
    {
        printf("Cannot open video capture device.\n");
        return 0;
    }

    Mat frame;
    Mat low_res_frame;
    char class_letter;
    string imgText = string("This is letter ");

    // Display window
    namedWindow(WIN_CAM, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_CAM, 0, 0);

    //Capture video in case property is set
    if (OUTPUT_VIDEO) {
        capture >> frame;
        outputVideo.open("outputVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, frame.size(), true);
    }

    trainNaiveBayes();

    createSquares(capture);
    waitForPalmCover(capture);
    average(capture);

    int counter = 0;
    for (;;) {
        capture >> frame;
        flip(frame, frame, 1);

        pyrDown(frame, low_res_frame);
        blur(low_res_frame, low_res_frame, Size(3, 3));

        cvtColor(low_res_frame, low_res_frame, CV_BGR2HLS);
        hsl_histogram(low_res_frame);
        thresholdBW(low_res_frame);
        cvtColor(low_res_frame, low_res_frame, CV_HLS2BGR);
        takeHandImage(frame);

        //Get gesture every 50 frames to avoid lag
        if (counter % 50 == 0) {
            class_letter = classifyBayes(bw_img);

            imgText = string("This is letter ");
            imgText += class_letter;

            counter = 0;
        }
        printText(frame, imgText);  

        imshow(WIN_CAM, frame);
        if (OUTPUT_VIDEO) {
            outputVideo << frame;
        }

        c = cvWaitKey(10);  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished");
            break;  //ESC pressed
        }

        counter++;
    }

    destroyAllWindows();
    if (OUTPUT_VIDEO) {
        outputVideo.release();
    }
    capture.release();
    return 0;
}