#ifndef ROI 
#define ROI


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

using namespace cv;

class Palm_ROI {
public:
    Palm_ROI();
    Palm_ROI(Point center, int edge_length);
    Point upper_corner, lower_corner;
    Point center;
    Scalar color;
    int border_thickness;
    void draw_rectangle(Mat src);
};



#endif