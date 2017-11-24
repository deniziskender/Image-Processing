/* Deniz Iskender S000704 Department of Computer Science */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <vector>
#include <math.h>
#include <dirent.h>
#include <iostream>
#include <map>

using namespace std;
using namespace cv;

struct Compare {
	bool operator()(const Rect& first, const Rect& second) const {
		
		int x1 = first.x; int x2 = first.y; int xWidth = first.width; int xHeight = first.height;
		int y1 = second.x; int y2 = second.y; int yWidth = second.width; int yHeight = second.height;

		int allX = x1 + x2 + xWidth + xHeight;
		int allY = y1 + y2 + yWidth + yHeight;

		return allX < allY;
	}
};

Mat applyThresholdAndCanny(Mat src){

	Mat threshold;
	cv::threshold(src, threshold, 10, 250, CV_THRESH_TOZERO | CV_THRESH_OTSU);

	//Find the edges
	Mat edgeDetected;
	cv::Canny(threshold, edgeDetected, 60, 200);
	
	Mat dilated;
	dilate(edgeDetected, dilated, 0, Point(-1, -1), 1, 1, 1);

	return dilated;
}

void writeOutput(map<Rect, vector<Point>, Compare> circleAddedCluster2){

	int outputHistogram[6];

	//create the empty histogram
	for (int i = 0; i < 6; i++){
		outputHistogram[i] = 0;
	}

	//change histograms values according to circleAddedCluster2
	for (auto it = circleAddedCluster2.begin(); it != circleAddedCluster2.end(); it++) {
		int outputValue = it->second.size() - 1;
		if (outputValue >= 6){
			outputValue = 5;
		}
		//increase the value
		outputHistogram[outputValue]++;
	}
	//write the histogram
	cout << "[" << outputHistogram[0] << ", " << outputHistogram[1] << ", "
		<< outputHistogram[2] << ", " << outputHistogram[3] << ", " 
		<< outputHistogram[4] << ", " << outputHistogram[5] <<
		"]" << endl;
}

map<Rect, vector<Point>, Compare> addCirclesToClusters(vector<Point> allCirclesInOneImage,
	vector<Rect> allRectanglesInOneImage,
	map<Rect, vector<Point>, Compare> circlesAddedClusters){
	
	for (vector<Point>::iterator circle = allCirclesInOneImage.begin(); circle != allCirclesInOneImage.end(); circle++) {
		for (vector<Rect>::iterator square = allRectanglesInOneImage.begin(); square != allRectanglesInOneImage.end(); square++) {

			int circleX = circle->x;
			int circleY = circle->y;
			int squareX = square->x;
			int squareY = square->y;
			int squareWidth = square->width;
			int squareHeight = square->height;

			//decide circles are inside of any of rectangle 
			if (circleX >= squareX && circleX <= squareX + squareWidth && circleY >= squareY && circleY <= squareY + squareHeight) {
				circlesAddedClusters[*square].push_back(*circle);
				break;
			}
		}
	}
	return circlesAddedClusters;
}

int main() {

	vector<string> namesOfImages;

	//take all image names and add them to namesOfImages
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir("test")) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			if (string(ent->d_name) == "." || string(ent->d_name) == ".."){
			}
			else{
				namesOfImages.push_back(string(ent->d_name));
			}
		}
		closedir(dir);
	}

	//read these images and make calculations according to these
	for (int i = 0; i < namesOfImages.size(); i++){

		String name = namesOfImages.at(i);
		Mat src = imread("test//" + name, 0);

		//Could not read the image
		if (src.empty()) {
			std::cout << "!!! Failed imread(): image not found" << std::endl;
		}

		//Circles and rects
		vector<Point> allCirclesInOneImage = vector<Point>();
		vector<Rect> allRectanglesInOneImage = vector<Rect>();

		//Initialize for clusters
		map<Rect, vector<Point>, Compare> rectPointAndCompare = map<Rect, vector<Point>, Compare>();
		map<int, vector<Point>> clusters = map <int, vector<Point>>();
		int clusterNumbers = 0;

		//Apply the threshold and canny to make image in a wanted format
		cv::Mat updatedVersionOfImage = applyThresholdAndCanny(src);
		
		//FIND CIRCLES AND RECTANGLES WITH CONTOUR
		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(updatedVersionOfImage.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
		std::vector<cv::Point> approx;
		for (int i = 0; i < contours.size(); i++) {

			cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);

			if (std::fabs(cv::contourArea(contours[i])) < 10 || !cv::isContourConvex(approx))
				continue;
			if (approx.size() >= 4 && approx.size() <= 6) {
				//can be rect, other things
				int vtc = approx.size();
				std::vector<double> cos;
				for (int j = 2; j < vtc + 1; j++){
					
					double dx1 = approx[j%vtc].x - approx[j - 1].x;
					double dy1 = approx[j%vtc].y - approx[j - 1].y;
					double dx2 = approx[j - 2].x - approx[j - 1].x;
					double dy2 = approx[j - 2].y - approx[j - 1].y;
					
					float angle = (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
					cos.push_back(angle);
				}
				
				std::sort(cos.begin(), cos.end());
				double mincos = cos.front();
				double maxcos = cos.back();

				//get the rects if it is not exist
				if (vtc == 4 && mincos >= -0.1 && maxcos <= 0.6) {
					cv::Rect r = cv::boundingRect(contours[i]);
					cv::Point pt(r.x + ((r.width) / 2), r.y + ((r.height) / 2));
					bool exist = false;

					for (vector<Rect>::iterator it = allRectanglesInOneImage.begin(); it != allRectanglesInOneImage.end(); it++) {
						if (norm(Point(it->x, it->y) - Point(r.x, r.y)) < 10){
							exist = true;
							break;
						}
					}
					if (!exist){
						allRectanglesInOneImage.push_back(r);
					}
				}
			}
			else {
				//get the circles if is not exists
				cv::Rect r = cv::boundingRect(contours[i]);
				cv::Point pt(r.x + ((r.width) / 2), r.y + ((r.height) / 2));
				bool exist = false;

				for (vector<Point>::iterator it = allCirclesInOneImage.begin(); it != allCirclesInOneImage.end(); it++) {
					if (norm(*it - pt) < 5)
						exist = true;
				}

				if (!exist)
					allCirclesInOneImage.push_back(pt);
			}
		}

		//decide whether circle is inside of square or not
		//if so, add to the cluster
		map<Rect, vector<Point>, Compare> circleAddedCluster =
			addCirclesToClusters(allCirclesInOneImage, allRectanglesInOneImage, rectPointAndCompare);
		
		//Remove useless circles 
		for (auto iterator1 = circleAddedCluster.begin(); iterator1 != circleAddedCluster.end(); iterator1++) {
			for (auto iterator2 = iterator1->second.begin(); iterator2 != iterator1->second.end(); iterator2++) {
				allCirclesInOneImage.erase(
					remove(allCirclesInOneImage.begin(), allCirclesInOneImage.end(), *iterator2), allCirclesInOneImage.end());
			}
		}

		//sort all circles which are points
		std::sort(allCirclesInOneImage.begin(), allCirclesInOneImage.end(), [](const Point & a, const Point & b) {
			return norm(a) < norm(b);
		});
		for (Point a : allCirclesInOneImage) {
		}

		//Go over rectangles and take the needed one
		Rect rect;
		for (vector<Rect>::iterator iterator1 = allRectanglesInOneImage.begin(); iterator1 != allRectanglesInOneImage.end(); iterator1++) {
			int size = circleAddedCluster[*iterator1].size();
			if ( size > 0) {
				rect = *iterator1;
				break;
			}
		}

		//Cluster
		for (vector<Point>::iterator circle = allCirclesInOneImage.begin(); circle != allCirclesInOneImage.end(); circle++) {
			
			int width = rect.width;
			int height = rect.height;
			
			double thresholdValue = norm(Point(width, height));
			Point CirclePoint = *circle;
			
			bool found = false;
			for (auto ints = clusters.begin(); ints != clusters.end(); ++ints) {

				int sumOfX = 0;
				int sumOfY = 0;
				for (auto vectors = ints->second.begin(); vectors != ints->second.end(); vectors++) {
					sumOfX += vectors->x;
					sumOfY += vectors->y;
				}
				Point p = Point(sumOfX / ints->second.size(), sumOfY / ints->second.size());
				if (cv::norm((p)-CirclePoint) < thresholdValue / 2) {
					clusters[ints->first].push_back(CirclePoint);
					found = true;
					break;
				}
			}
			//If you could not find, add
			if (!found) {
				clusters[clusterNumbers++].push_back(*circle);
			}
		}

		//Add rectangles
		for (auto iterator1 = clusters.begin(); iterator1 != clusters.end(); iterator1++) {
			
			int sumofX = 0;
			int sumOfY = 0;
			
			for (auto iterator2 = iterator1->second.begin(); iterator2 != iterator1->second.end(); iterator2++) {
				sumofX += iterator2->x;
				sumOfY += iterator2->y;
			}

			Rect rectangle;
			rectangle.x = sumofX / iterator1->second.size() - rect.width / 2;
			rectangle.y = sumOfY / iterator1->second.size() - rect.height / 2;
		
			rectangle.width = rect.width;
			rectangle.height = rect.height;

			allRectanglesInOneImage.push_back(rectangle);
		}

		//Put the circles in our new rectangles
		map<Rect, vector<Point>, Compare> circleAddedCluster2 =
			addCirclesToClusters(allCirclesInOneImage, allRectanglesInOneImage, circleAddedCluster);
		
		writeOutput(circleAddedCluster2);

	}
	return 0;
}