#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
    if (!cap.isOpened()) // openenig the video device failed
    {
        printf("Cannot open video capture device.\n");
        return 0;
    }

    Mat frame;
    char numberStr[256];
    char fileName[256];

    // Display window
    const char* WIN_SRC = "Src"; //window for the source frame
    namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_SRC, 0, 0);

    char c;
    int frameNum = -1;
    int frameCount = 0;

    for (;;)
    {
        cap >> frame; // get a new frame from camera
        if (frame.empty())
        {
            printf("End of the video file\n");
            break;
        }

        ++frameNum;

        imshow(WIN_SRC, frame);

        c = cvWaitKey(10);  // waits a key press to advance to the next frame
        if (c == 27) {
            // press ESC to exit
            printf("ESC pressed - capture finished");
            break;  //ESC pressed
        }
    }
    return 0;
}