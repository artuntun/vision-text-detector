#include "opencv\cv.hpp"
#include <iostream>

#define ESCAPE 27

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	Mat img;
	img = imread("data/mandril.jpg", 0);
	if (!img.data)
	{
		printf("Error al cargar la imagen");
		return 1;
	}

	//cvtColor(img, img, CV_BGR2GRAY);

	namedWindow("bla", CV_WINDOW_AUTOSIZE);
	namedWindow("histogram", CV_WINDOW_AUTOSIZE);

	int histSize = 256;
	float range[] = { 0,255 };
	const float *ranges[] = { range };

	Mat hist;
	calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	double total;
	total = img.rows * img.cols;
	/*for (int h = 0; h < histSize; h++)
	{
		float binVal = hist.at<float>(h);
		cout << " " << binVal;
	}*/

	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));

	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(0, 0, 0), 2, 8, 0);
	}

	imshow("bla", img);
	imshow("histogram", histImage);

	waitKey(0);

	destroyAllWindows();

	return 0;

}