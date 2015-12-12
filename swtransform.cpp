#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <boost/lambda/lambda.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include "DisjointSets.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#define PI 3.14159265

using namespace cv;

/// Global variables


Mat dst, filtered;
Mat gradient_x, abs_grad_x, gradient_y, abs_grad_y;
Mat grad;
//Mat SWTimage;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

int strokeMedianFilter(Mat *SWTimage, std::vector<std::vector<Point2d>> *saved_rays);
int strokeWidth(int i, int j, short g_x, short g_y, Mat *SWTimage, std::vector<std::vector<Point2d>> *saved_rays, Mat *edge_image);
int getSteps(int g_x, int g_y, float& step_x, float& step_y);
Mat SWTransform(Mat* edge_image);
std::vector<std::vector<Point2d>> findConnectedComponents(Mat* SWTimage);
Mat legallyConnectedComponents(Mat* SWTimage);

/** @function main */

int main(int argc, char** argv)
{	
	Mat src, src_gray;
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
	//gradients
	Mat gradxx;
	Sobel(filtered, gradient_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	gradient_x.copyTo(gradxx);
	//Scharr(filtered, gradient_x, CV_16S, 1, 0, 3);
	convertScaleAbs(gradient_x, abs_grad_x); 
	Sobel(filtered, gradient_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	//Scharr(filtered, gradient_y, CV_16S, 0, 1, 3);
	convertScaleAbs(gradient_y, abs_grad_y);
	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	
	Mat SWTimage = SWTransform(&detected_edges);
	//SWTimage.convertTo(SWTimage, CV_8U, 1);

	//std::vector<std::vector<Point2d>> components = findConnectedComponents(&SWTimage);
	//Mat components_image = cv::Mat(SWTimage.rows, SWTimage.cols, CV_8U , cv::Scalar(0));
	Mat components_image = legallyConnectedComponents(&SWTimage);

	//for (std::vector<std::vector<Point2d>>::iterator it1 = components.begin(); it1 != components.end(); ++it1){
	//for (int i = 1; i < 15; i++){
	//	std::vector<Point2d> prueba = components[i];
	//	//for (std::vector<Point2d>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2){
	//	for (std::vector<Point2d>::iterator it1 = prueba.begin(); it1 != prueba.end(); ++it1){
	//		components_image.at<uchar>(it1->y, it1->x) = 255;
	//	}
	//}
	uchar *ptr2;
	for (int i = 0; i < components_image.rows; i++){
		ptr2 = components_image.ptr<uchar>(i);
		for (int j = 0; j < components_image.cols; j++){
			if (ptr2[j] > 0)
				ptr2[j] = 255;
		}
	}
	
	
	//Create windo canvas to show img
	namedWindow("original", CV_WINDOW_AUTOSIZE);
	namedWindow("gray", CV_WINDOW_AUTOSIZE);
	namedWindow("filtered", CV_WINDOW_AUTOSIZE);
	namedWindow("edges", CV_WINDOW_AUTOSIZE);
	namedWindow("gradientx", CV_WINDOW_AUTOSIZE);
	namedWindow("gradienty", CV_WINDOW_AUTOSIZE);
	namedWindow("grad", CV_WINDOW_AUTOSIZE);
	namedWindow("SWT", CV_WINDOW_AUTOSIZE);
	namedWindow("connected_components", CV_WINDOW_AUTOSIZE);
	//Show image in the name of the window
	imshow("original", src);
	imshow("gray", src_gray);
	imshow("filtered", filtered);
	imshow("edges", detected_edges);
	imshow("gradientx", abs_grad_x);
	imshow("gradienty", abs_grad_y);
	imshow("grad", grad);
	imshow("SWT", SWTimage);
	imshow("connected_components", components_image);

	/// Wait until user exit program by pressing a key
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

Mat legallyConnectedComponents(Mat* SWTimage){
	std::vector<std::vector<Point2d>> components;
	Mat labels = cv::Mat(SWTimage->rows, SWTimage->cols, CV_8U , cv::Scalar(0));

	DisjointSets s(1);
	int next_label = 1;
	float* ptr;
	for (int i = 0; i < SWTimage->rows; i++){
		ptr = SWTimage->ptr<float>(i);
		for (int j = 0; j < SWTimage->cols; j++){
			if (ptr[j] > 0){
				std::vector<int> neighbour_labels;
				// check pixel to the west, west-north, north, north-east
				if (j - 1 >= 0) {
					float west = SWTimage->at<float>(i, j - 1);
					if (west > 0){
						if (ptr[j] / west <= 3.0 || west / ptr[j] <= 3.0){
							int west_label = labels.at<uchar>(i, j - 1);
							if (west_label != 0)
								neighbour_labels.push_back(west_label);
						}
					}
				}
				if (i - 1 >= 0) {
					if (j - 1 >= 0){
						float west_north = SWTimage->at<float>(i - 1, j - 1);
						if (west_north > 0){
							if (ptr[j] / west_north <= 3.0 || west_north / ptr[j] <= 3.0){
								int west_north_label = labels.at<uchar>(i - 1, j - 1);
								if (west_north_label != 0)
									neighbour_labels.push_back(west_north_label);
							}
						}
					}
					float north = SWTimage->at<float>(i - 1, j);
					if (north > 0){
						if (ptr[j] / north <= 3.0 || north / ptr[j] <= 3.0){
							int north_label = labels.at<uchar>(i - 1, j);
							if (north_label != 0)
								neighbour_labels.push_back(north_label);
						}
					}
					if (j + 1 < SWTimage->cols){
						float north_east = SWTimage->at<float>(i - 1, j + 1);
						if (north_east > 0){
							if (ptr[j] / north_east <= 3.0 || north_east / ptr[j] <= 3.0){
								int north_east_label = labels.at<uchar>(i - 1, j + 1);
								if (north_east_label != 0)
									neighbour_labels.push_back(north_east_label);
							}
						}
					}	
				}
				//if neighbours have no label. Set new label(component)
				if (neighbour_labels.empty()){
					labels.at<uchar>(i , j ) = next_label;
					next_label++;
					s.AddElements(1);
				}
				else{
					int minimum = 1;
					for (std::vector<int>::iterator it = neighbour_labels.begin(); it != neighbour_labels.end(); it++){
						minimum = std::min(minimum, *it);
						s.Union(s.FindSet(neighbour_labels[0]), s.FindSet(*it));
					}
					labels.at<uchar>(i, j) = minimum;
				}

			}
		}
	}

	uchar * ptr2;
	for (int i = 0; i < labels.rows; i++){
		ptr2 = labels.ptr<uchar>(i);
		for (int j = 0; j < labels.cols; j++){
			if (ptr2[j] != 0){
				ptr2[j] = s.FindSet(ptr2[j]);
			}
		}
	}

	return labels;
}

std::vector<std::vector<Point2d>> findConnectedComponents(Mat* SWTimage){

	boost::unordered_map<int, int> map;
	boost::unordered_map<int, Point2d> revmap;

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
	int num_vertices = 0;
	// Number vertices for graph.  Associate each point with number
	for (int row = 0; row < SWTimage->rows; row++){
		float * ptr = SWTimage->ptr<float>(row);
		for (int col = 0; col < SWTimage->cols; col++){
			if (ptr[col] > 0) {
				map[row * SWTimage->cols + col] = num_vertices;
				Point2d p;
				p.x = col;
				p.y = row;
				revmap[num_vertices] = p;
				num_vertices++;
			}
		}
	}

	Graph g(num_vertices);
	
	for (int row = 0; row < SWTimage->rows; row++){
		float * ptr = SWTimage->ptr<float>(row);
		for (int col = 0; col < SWTimage->cols; col++){
			if (ptr[col] > 0) {
				// check pixel to the right, right-down, down, left-down
				int this_pixel = map[row * SWTimage->cols + col];
				if (col + 1 < SWTimage->cols) {
					float right = SWTimage->at<float>(row, col + 1);
					//float right = CV_IMAGE_ELEM(SWTImage, float, row, col + 1);
					if (right > 0 && (ptr[col] / right <= 3.0 || right / ptr[col] <= 3.0))
						boost::add_edge(this_pixel, map.at(row * SWTimage->cols + col + 1), g);
				}
				if (row + 1 < SWTimage->rows) {
					if (col + 1 < SWTimage->cols) {
						float right_down = SWTimage->at<float>(row + 1, col + 1);
						//float right_down = CV_IMAGE_ELEM(SWTImage, float, row + 1, col + 1);
						if (right_down > 0 && (ptr[col] / right_down <= 3.0 || right_down / ptr[col] <= 3.0))
							boost::add_edge(this_pixel, map.at((row + 1) * SWTimage->cols + col + 1), g);
					}
					float down = SWTimage->at<float>(row + 1, col);
					//float down = CV_IMAGE_ELEM(SWTImage, float, row + 1, col);
					if (down > 0 && (ptr[col] / down <= 3.0 || down / ptr[col] <= 3.0))
						boost::add_edge(this_pixel, map.at((row + 1) * SWTimage->cols + col), g);
					if (col - 1 >= 0) {
						float left_down = SWTimage->at<float>(row + 1, col - 1);
						//float left_down = CV_IMAGE_ELEM(SWTImage, float, row + 1, col - 1);
						if (left_down > 0 && (ptr[col] / left_down <= 3.0 || left_down / ptr[col] <= 3.0))
							boost::add_edge(this_pixel, map.at((row + 1) * SWTimage->cols + col - 1), g);
					}
				}
			}
			ptr++;
		}
	}

	std::vector<int> c(num_vertices);

	int num_comp = connected_components(g, &c[0]);

	std::vector<std::vector<Point2d> > components;
	components.reserve(num_comp);
	std::cout << "Before filtering, " << num_comp << " components and " << num_vertices << " vertices" << std::endl;
	for (int j = 0; j < num_comp; j++) {
		std::vector<Point2d> tmp;
		components.push_back(tmp);
	}
	for (int j = 0; j < num_vertices; j++) {
		Point2d p = revmap[j];
		(components[c[j]]).push_back(p);
	}


	return components;
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
