#include <stdlib.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void makeGaussianPyramid(Mat& src, int s, vector<Mat>& pyr);
vector<Mat> centerSurround(vector<Mat>& fmap1, vector<Mat>& fmap2);
void normalizeMap(vector<Mat>& nmap);
Mat make_depth_histogram(Mat data, int width, int height);

VideoCapture capture("rgb(0).avi");
int main()
{
	bool flag=1;
	Mat frame;
	vector<Mat> kernels(4);
	for(int k=0; k<kernels.size(); k++) kernels[k] = getGaborKernel(Size(20,20),1, CV_PI/4*k, 30, 0,CV_PI/2);

	capture >> frame;
	int width = frame.cols;
	int height = frame.rows;
	int cv_type = frame.type();

	VideoWriter video("out.avi",CV_FOURCC('M','J','P','G'),20, Size(width*3,height),true);
	VideoWriter video2("out2.avi",CV_FOURCC('M','J','P','G'),20, Size(width*3,height),true);

	// Time checking start
	int frames = 0;
	float time = 0, fps = 0;
	auto t0 = std::chrono::high_resolution_clock::now();

	while(1)
	{
		if (flag==1)
		{
			capture >> frame;
			if (frame.empty()) break;
			flag = 0;
		}
		else
		{
			// Step 1.
			vector<Mat> GausnPyr(9);
			makeGaussianPyramid(frame,GausnPyr.size(),GausnPyr);

			// Step 2.
			vector<Mat> Pyr_I(GausnPyr.size()); //Pyr_I[#pyr]
			vector<vector<Mat>> Pyr_C(2); //Pyr_C[#BGR][#pyr]
			vector<vector<Mat>> Pyr_O(4); //Pyr_O[#theta][#pyr]
			for(int i=0; i<GausnPyr.size(); i++) {
				vector<Mat> vtemp(3);
				split(GausnPyr[i], vtemp);

				// Make Intensity Map -> #1
				Pyr_I[i] = (vtemp[0]+vtemp[1]+vtemp[2])/3; //Blue

				// Make Color Map -> #2
				Mat B = vtemp[0]-(vtemp[1]+vtemp[2])/2; //Blue
				Mat Y = (vtemp[2]+vtemp[1])/2-abs((vtemp[2]-vtemp[1])/2)-vtemp[0]; //Yellow
				Mat R = vtemp[2]-(vtemp[1]+vtemp[0])/2; //Red
				Mat G = vtemp[1]-(vtemp[0]+vtemp[2])/2; //Green
				Pyr_C[0].push_back((Mat)(B-Y));
				Pyr_C[1].push_back((Mat)(R-G));
				vtemp.clear();

				// Make Orientation Map -> #4
				for(int k=0; k<Pyr_O.size(); k++){
					Mat buf;
					filter2D(Pyr_I[i], buf, CV_32F, kernels[k]);
					Pyr_O[k].push_back(buf);
				}
			}
			GausnPyr.clear();

			// Step 3. Center-Surrounded Difference
			vector<Mat> CSD_I,CSD_C,CSD_O;
			CSD_I = centerSurround(Pyr_I,Pyr_I); // 8->6
			Pyr_I.clear();
			for(int k=0; k<Pyr_C.size(); k++) {
				vector<Mat> inv_Pyr_C(Pyr_C[k].size());
				for(int l=0; l<Pyr_C[k].size(); l++) inv_Pyr_C[l] = -Pyr_C[k][l];
				Pyr_C[k] = centerSurround(Pyr_C[k],inv_Pyr_C); //R-G and G-R, B-Y and Y-B
				for(int l=0; l<Pyr_C[k].size(); l++) CSD_C.push_back(Pyr_C[k][l]);
				Pyr_C[k].clear();
			}
			Pyr_C.clear();
			for(int k=0; k<Pyr_O.size(); k++) {
				Pyr_O[k] = centerSurround(Pyr_O[k],Pyr_O[k]);
				for(int l=0; l<Pyr_O[k].size(); l++) CSD_O.push_back(Pyr_O[k][l]);
				Pyr_O[k].clear();
			}
			Pyr_O.clear();

			// Step 4. Normalization
			normalizeMap(CSD_I);
			normalizeMap(CSD_C);
			normalizeMap(CSD_O);

			// Step 5. Conspicuity Maps
			Mat I = Mat::zeros(Size(CSD_I[0].cols,CSD_I[0].rows),CSD_I[0].type());
			Mat C = Mat::zeros(Size(CSD_C[0].cols,CSD_C[0].rows),CSD_C[0].type());
			Mat O = Mat::zeros(Size(CSD_O[0].cols,CSD_O[0].rows),CSD_O[0].type());
			for(int i=0; i<CSD_I.size(); i++) I += CSD_I[i];
			for(int i=0; i<CSD_C.size(); i++) C += CSD_C[i];
			for(int i=0; i<CSD_O.size(); i++) O += CSD_O[i];
			CSD_I.clear(); CSD_C.clear(); CSD_O.clear();

			// Step 6. Merge
			normalize(I,I,0,255,NORM_MINMAX,CV_8UC1);
			normalize(C,C,0,255,NORM_MINMAX,CV_8UC1);
			normalize(O,O,0,255,NORM_MINMAX,CV_8UC1);
			Mat Salmap = (I+C+O)/3;
			Point maxLoc, minLoc;
			double maxVal, minVal;
			minMaxLoc(Salmap,&minVal,&maxVal,&minLoc,&maxLoc);

			//check for FPS(Frame Per Second)
			auto t1 = std::chrono::high_resolution_clock::now();
			time += std::chrono::duration<float>(t1-t0).count();
			t0 = t1;
			++frames;
			if(time > 0.5f)
			{
				fps = frames / time;
				frames = 0;
				time = 0;
			}
			std::ostringstream ss;
			ss.precision(2);
			ss << " FPS=" << fps;
			putText(frame, ss.str(), Point(10,15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1.2);

			// Step 7. Display and Save result
			Salmap.convertTo(Salmap,CV_16UC1);
		    Mat Salmap_RGB = make_depth_histogram(Salmap,width,height);
		    Salmap.convertTo(Salmap,CV_8UC1);
		    cvtColor(Salmap,Salmap,COLOR_GRAY2BGR);
		    circle(Salmap,maxLoc,4,Scalar(0,255,0),-1);
		    vector<Mat> result={frame,Salmap,Salmap_RGB};
			cvtColor(I,I,COLOR_GRAY2BGR);
			cvtColor(C,C,COLOR_GRAY2BGR);
			cvtColor(O,O,COLOR_GRAY2BGR);
		    vector<Mat> fmap={I,C,O};

			Mat dst(Size(width*3,height),cv_type,Scalar::all(0));
			Mat ICO(Size(width*3,height),cv_type,Scalar::all(0));
			hconcat(result, dst);
			hconcat(fmap, ICO);
			imshow("result",dst);
			imshow("ICO map",ICO);
			video << dst;
			video2 << ICO;

			if(27 == waitKey(10)) break;
			flag = 1;
		}
	}
	kernels.clear();
	return 0;
}

void makeGaussianPyramid(Mat& src, int s, vector<Mat>& pyr){
	pyr[0] = src.clone();
	for(int i=1; i<s; i++){
		pyrDown(pyr[i-1],pyr[i], Size((int)(pyr[i-1].cols/2),(int)(pyr[i-1].rows/2)));
		resize(pyr[i],pyr[i],Size(pyr[0].cols,pyr[0].rows));
	}
}

vector<Mat> centerSurround(vector<Mat>& fmap1, vector<Mat>& fmap2){
	vector<int> center = {2,3,4};
	vector<int> delta = {3,4};
	vector<Mat> CSD;
	for(int c=0; c < center.size(); c++)
		for(int d=0; d<delta.size(); d++)
		{
			Mat ctemp = fmap1[center[c]];
			Mat stemp = fmap2[delta[d]+center[c]];
			Mat temp = abs(ctemp-stemp);
			CSD.push_back(temp);
			temp.release();
		}
	return CSD;
}

void normalizeMap(vector<Mat>& nmap){
	for(int i=0; i<nmap.size(); i++)
	{
		normalize(nmap[i],nmap[i],0,1,NORM_MINMAX,CV_32FC1);
		Scalar meanVal = mean(nmap[i]);
		nmap[i] *= pow((1-(double)meanVal.val[0]),2);
	}
}

Mat make_depth_histogram(Mat data, int width, int height)
{
	Mat rgb(Size(data.cols,data.rows), CV_8UC3);
    static uint32_t histogram[0x10000];
    memset(histogram, 0, sizeof(histogram));

    for(int j = 0; j < height; ++j) for(int i = 0; i < width; ++i) ++histogram[data.at<uint16_t>(j,i)];
    for(int i = 2; i < 0x10000; ++i) histogram[i] += histogram[i-1]; // Build a cumulative histogram for the indices in [1,0xFFFF]
    for(int j = 0; j < height; ++j)
    	for(int i = 0; i < width; ++i){
			if(uint16_t d = data.at<uint16_t>(j,i))
			{
				int f = histogram[d] * 255 / histogram[0xFFFF]; // 0-255 based on histogram location
				rgb.at<Vec3b>(j, i)[0] = 255 - f;
				rgb.at<Vec3b>(j, i)[1] = 0;
				rgb.at<Vec3b>(j, i)[2] = f;
			}
			else
			{
				rgb.at<Vec3b>(j, i)[0] = 0;
				rgb.at<Vec3b>(j, i)[1] = 0;
				rgb.at<Vec3b>(j, i)[2] = 0;
			}
    	}
    return rgb;
}

