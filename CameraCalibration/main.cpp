#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	/******************* Parameters *******************/
	string srcPath = "Image/MI8/";
	string dstPath = "Result/MI8/";
	string pattern = "*.jpg";
	string intrinsicsFilename = "Result/MI8/intrinsics.yml";
	bool showFlag = false;
	bool saveFlag = true;

	Size patternSize = Size(8, 6);  
	Size squareSize = Size(28, 28);  // unit: mm
	int numCorners = patternSize.width * patternSize.height;

	/******************* Load image filenames *******************/
	vector<string> fileList;
	glob(srcPath + pattern, fileList, false);
	int numImages = (int)fileList.size();

	/******************* Extract corners *******************/
	cout << "¡­¡­¡­¡­¡­¡­ Extract corners ¡­¡­¡­¡­¡­¡­" << endl;
	vector<vector<Point2f>> cornersList; 
	vector<string> filenameList;
	Size imageSize = Size();
	for (int i = 0; i < numImages; i++) {
		string filename = fileList[i];
		Mat image = imread(filename);
		if (image.empty()) {
			cout << "Failed to read image " << filename << endl;
			continue;
		}
		if (imageSize.empty())
			imageSize = image.size();
		vector<Point2f> corners;
		if (findChessboardCorners(image, patternSize, corners) == 0) {
			cout << "Failed to find chessboard corners" << endl; 
			continue;
		}
		else {
			Mat gray;
			cvtColor(image, gray, COLOR_BGR2GRAY);
			find4QuadCornerSubpix(gray, corners, Size(5, 5)); 												
			cornersList.push_back(corners);  
			filenameList.push_back(filename);

			if (showFlag) {
				drawChessboardCorners(image, patternSize, corners, true);
				namedWindow("image", 0);
				imshow("image", image);
				waitKey(100);
			}
		}
		cout << "Progress: " << i + 1 << "/" << numImages << endl;
	}
	numImages = (int)filenameList.size();

	/******************* Calibrate camera *******************/
	cout << "¡­¡­¡­¡­¡­¡­ Calibrate camera ¡­¡­¡­¡­¡­¡­" << endl;

	vector<vector<Point3f>> objectPointsList; 
	for (int n = 0; n< numImages; n++) {
		vector<Point3f> objectPoints;
		for (int i = 0; i<patternSize.height; i++) {
			for (int j = 0; j<patternSize.width; j++) {
				Point3f p3d;
				p3d.x = j * squareSize.width;
				p3d.y = i * squareSize.height;
				p3d.z = 0;
				objectPoints.push_back(p3d);
			}
		}
		objectPointsList.push_back(objectPoints);
	}

	Mat cameraMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0));
	Mat distCoeffs = Mat(1, 5, CV_64FC1, Scalar::all(0)); 
	vector<Mat> rvecs; 
	vector<Mat> tvecs;
	calibrateCamera(objectPointsList, cornersList, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0);

	/******************* Evaluate result *******************/
	cout << "¡­¡­¡­¡­¡­¡­ Evaluate result ¡­¡­¡­¡­¡­¡­" << endl;
	double averageError = 0.0;
	for (int n = 0; n< numImages; n++) {
		double error = 0.0;
		vector<Point3f> objectPoints = objectPointsList[n];
		vector<Point2f> corners = cornersList[n];
		vector<Point2f> imagePoints;
		projectPoints(objectPoints, rvecs[n], tvecs[n], cameraMatrix, distCoeffs, imagePoints);
		for (int i = 0; i < numCorners; i++) {
			Vec2f vector = Vec2f(corners[i].x, corners[i].y) - Vec2f(imagePoints[i].x, imagePoints[i].y);
			error += norm(vector);
		}
		error /= numCorners;
		averageError += error;
		cout << "Image: " << n + 1 << "\terror: " << error << endl;
	}
	averageError /= numImages;
	cout << "Average error" << averageError << endl;

	/******************* Save result *******************/
	cout << "¡­¡­¡­¡­¡­¡­ Save result ¡­¡­¡­¡­¡­¡­" << endl;
	cout << "cameraMatrix:\n" << cameraMatrix << endl;
	cout << "distCoeffs:\n" << distCoeffs << endl;
	for (int i = 0; i< numImages; i++) {
		cout << "Image: " << i + 1 << endl;
		cout << "\tRvec: " << rvecs[i].t() << endl;
		cout << "\tTvec: " << tvecs[i].t() << endl;
	}
	FileStorage fs(intrinsicsFilename, FileStorage::WRITE);
	if (fs.isOpened()) {
		fs << "K" << cameraMatrix;
		fs << "D" << distCoeffs;
	}
	fs.release();
	
	/******************* Rectify images *******************/
	cout << "¡­¡­¡­¡­¡­¡­ Rectify images ¡­¡­¡­¡­¡­¡­" << endl;
	for (int i = 0; i < numImages; i++) {
		string filename = filenameList[i];
		Mat image = imread(filename);
		undistort(image.clone(), image, cameraMatrix, distCoeffs);
		if (showFlag) {
			namedWindow("image", 0);
			imshow("image", image);
			waitKey(100);
		}
		if (saveFlag) {
			int pos = filename.find_last_of('\\') + 1;
			filename = dstPath + filename.substr(pos);
			imwrite(filename, image);
		}
		cout << "Progress: " << i + 1 << "/" << numImages << endl;
	}

	return 0;
}