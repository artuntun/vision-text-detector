#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

#define PI 3.14159265

using namespace cv;

/// Global variables


Mat gradient_x, gradient_y;

//Mat SWTimage;
int kernel_size = 3;
char* window_name = "Edge Map";

int strokeMedianFilter(Mat *SWTimage, std::vector<std::vector<Point2d>> *saved_rays);
int strokeWidth(int i, int j, short g_x, short g_y, Mat *SWTimage, std::vector<std::vector<Point2d>> *saved_rays, Mat *edge_image);
int getSteps(int g_x, int g_y, float& step_x, float& step_y);
Mat SWTransform(Mat* edge_image);
/** @function main */

int main(int argc, char** argv)
{
	Mat src, src_gray;
  Mat filtered;
	/// Load an image
	//src = imread("road_signs.png");
	src = imread("australiasigns.jpg");

	if (!src.data)
	{
		std::cout << "Couldn't read the image" << std::endl;
		return -1;
	}

	/// Convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY);
	// Reduce noise with a 3x3 gaussian kernel
	blur(src_gray, filtered, Size(3, 3));
	/// Canny detector
	Mat detected_edges;
	Canny(filtered, detected_edges, 100, 300, kernel_size);
	Sobel(filtered, gradient_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(filtered, gradient_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

	Mat SWTimage = SWTransform(&detected_edges);

	waitKey(0);

	return 0;
}

Mat SWTransform(Mat* edge_image){

	// accept only char and one channel type matrices
	CV_Assert(edge_image->depth() != sizeof(uchar));
	CV_Assert(edge_image->channels() == 1);

	int nRows = edge_image->rows;
	int nCols = edge_image->cols;

	Mat SWTimage = cv::Mat(nRows, nCols, CV_32FC1, cv::Scalar(-1.0));
	std::vector<std::vector<Point2d>> saved_rays;
	/*if (I.isContinuous())
	{
	nCols *= nRows;
	nRows = 1;
	}*/

	int i, j;
	uchar* p;
	int edges = 0;
	short g_x, g_y;
	for (i = 0; i < nRows; ++i)
	{
		p = edge_image->ptr<uchar>(i);
		for (j = 0; j < nCols; ++j)
		{
			//found a edge
			if (p[j] == 255){
				if (i == 22 && j == 138){
						edges++; //Debugging code. Used to access specific pixel
				}
				//gradient on the edge. Negative gradient to find dark text over clear background.
				g_x = -(gradient_x.at<short>(i, j));
				g_y = -(gradient_y.at<short>(i, j));
				strokeWidth(j, i, g_x, g_y, &SWTimage, &saved_rays, edge_image);
			}
		}
	}

	strokeMedianFilter(&SWTimage, &saved_rays);
	return SWTimage;
}

int strokeMedianFilter(Mat *SWTimage, std::vector<std::vector<Point2d>> *saved_rays){
	/*Compute median of each ray. Values higher than the median are set to median value. This function is necessary to avoid
	bad stroke values on corners.*/

	float median;
	for (std::vector<std::vector<Point2d>>::iterator it1 = saved_rays->begin(); it1 != saved_rays->end(); ++it1){
		std::vector<float> swt_values;
		for (std::vector<Point2d>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2){
			swt_values.push_back(SWTimage->at<float>(it2->y, it2->x));
		}
		std::sort(swt_values.begin(), swt_values.end());
		median = swt_values[round(swt_values.size() / 2)];
		for (std::vector<Point2d>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2){
			SWTimage->at<float>(it2->y, it2->x) = std::min(median, SWTimage->at<float>(it2->y, it2->x));
		}
	}

	return 0;
}

int strokeWidth(int j, int i, short g_x, short g_y, Mat *SWTimage,
	std::vector<std::vector<Point2d>> *saved_rays, Mat *edge_image){
	/*This function follow the g_x/g_y gradient direction from the D(j,i) point until finding a another edge pint P.
	Then both gradient direction are compared. If the dD = -dP +- PI/2 all the pixeles that connect D-P are labeld with the DP length*/

	float step_x, step_y;
	//get steps to move on the gradient direction
	getSteps(g_x, g_y, step_x, step_y);
	float sum_x = (float)j;
	float sum_y = (float)i;
	std::vector<Point2d> ray;
	Point2d point;
	point.x = j; point.y = i;
	ray.push_back(point);

	while (true){
		sum_x += step_x;
		sum_y += step_y;

		int x = (int)round(sum_x);
		int y = (int)round(sum_y);

		//pixel out of image
		if (x < 0 || (x >= (*SWTimage).cols) || y < 0 || (y >= (*SWTimage).rows)) {
			return 0;
		}

		point.x = x;
		point.y = y;
		ray.push_back(point);

		//matched a edge
		if ((*edge_image).at<uchar>(y, x) > 0){
			short gx_new = -(gradient_x.at<short>(y, x));
			short gy_new = -(gradient_y.at<short>(y, x));

			//get angles 2PI to 4PI (to have always positive angles)
			float angle_ij = atan2(-(g_y), -(g_x)) + PI +2*PI;
			float angle_yx = atan2(-(gy_new), -(gx_new)) + PI +2*PI;

			// if dp = - dq +- PI/2
			if (angle_ij >= angle_yx) angle_ij = angle_ij - PI;
			else angle_ij = angle_ij + PI;
			if (angle_ij  < angle_yx + PI / 2  && angle_ij  > angle_yx - PI / 2 ){
				//(*SWTimage).at<float>(y, x) = 255;
				//(*SWTimage).at<float>(i, j) = 255;
				float length = sqrt((y-i)*(y-i) + (x-j)*(x-j));
				saved_rays->push_back(ray);
				while (!ray.empty()){
					int x_ray = ray.back().x;
					int y_ray = ray.back().y;
					if ((*SWTimage).at<float>(y_ray, x_ray) < 0)
						(*SWTimage).at<float>(y_ray, x_ray) = length;
					else
						(*SWTimage).at<float>(y_ray, x_ray) = std::min((*SWTimage).at<float>(y_ray, x_ray), length);
					ray.pop_back();
				}
				return 1;
			}
			return 0;
		}
	}
}

int getSteps(int g_x, int g_y, float& step_x, float& step_y){
	float absg_x = abs(g_x);
	float absg_y = abs(g_y);
	if (g_x == 0){
		step_x = 0;
		if (g_y > 0) step_y = 1;
		else step_y = -1;
		return 0;
	}
	if (g_y == 0){
		step_y = 0;
		if (g_x > 0) step_x = 1;
		else step_x = -1;
		return 0;
	}
	if (absg_x > absg_y){
		step_y = g_y / absg_x;
		if (g_x > 0) step_x = 1;
		else step_x = -1;
	}
	else{
		step_x = g_x / absg_y;
		if (g_y > 0) step_y = 1;
		else step_y = -1;
	}
	return 0;
}
