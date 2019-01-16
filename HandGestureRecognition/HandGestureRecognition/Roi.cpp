#include "Roi.h"

using namespace cv;
using namespace std;

Palm_ROI::Palm_ROI() {
    center = Point(0, 0);
    upper_corner = Point(0, 0);
    lower_corner = Point(0, 0);
}

Palm_ROI::Palm_ROI(Point ctr, int e_length) {
    center = ctr;

    upper_corner = Point(
        ctr.x - e_length / 2,
        ctr.y - e_length / 2);
    lower_corner = Point(
        ctr.x + e_length / 2,
        ctr.y + e_length / 2);

    color = Scalar(0, 255, 0);
    border_thickness = 2;
}

void Palm_ROI::draw_rectangle(Mat src) {
    rectangle(src, upper_corner, lower_corner, color, border_thickness);
}