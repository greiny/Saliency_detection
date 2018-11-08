#include <stdlib.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace std;

void makeGaussianPyramid(cuda::GpuMat& src, int s, vector<cuda::GpuMat>& pyr);
vector<cuda::GpuMat> centerSurround(vector<cuda::GpuMat>& fmap1, vector<cuda::GpuMat>& fmap2);
void normalizeMap(vector<cuda::GpuMat>& nmap);
Mat make_depth_histogram(Mat data, int width, int height);

VideoCapture capture("rgb(0).avi");
int main()
{
	bool flag=1;
	Mat frame;
	capture >> frame;
	int width = frame.cols;
	int height = frame.rows;
	int cv_type = frame.type();
	VideoWriter video("cuda.avi",CV_FOURCC('M','J','P','G'),20, Size(width*3,height),true);
	VideoWriter video2("cuda2.avi",CV_FOURCC('M','J','P','G'),20, Size(width*3,height),true);

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
			auto t00 = std::chrono::high_resolution_clock::now();

			// Step 1.
			cuda::GpuMat frame_gpu(frame.rows, frame.cols, frame.type());
			frame_gpu.upload(frame);
			vector<cuda::GpuMat> GausnPyr(9);
			makeGaussianPyramid(frame_gpu,GausnPyr.size(),GausnPyr);

			auto t11 = std::chrono::high_resolution_clock::now();
			cout << "Step1: "<< std::chrono::duration<float>(t11-t00).count()<<"[s]" << endl;
			t00 = t11;

			// Step 2.
			vector<cuda::GpuMat> Pyr_I(GausnPyr.size()); //Pyr_I[#pyr]
			vector<vector<cuda::GpuMat>> Pyr_C(2); //Pyr_C[#BGR][#pyr]
			vector<vector<cuda::GpuMat>> Pyr_O(4); //Pyr_O[#theta][#pyr]
			for(int i=0; i<GausnPyr.size(); i++) {
				vector<cuda::GpuMat> vtemp(3);
				cuda::split(GausnPyr[i], vtemp);

				// Make Intensity Map -> #1
				cuda::add(vtemp[0],vtemp[1],Pyr_I[i]);
				cuda::add(Pyr_I[i],vtemp[2],Pyr_I[i]);
				cuda::divide(Pyr_I[i],3,Pyr_I[i]);

				// Make Color Map -> #2
				cuda::GpuMat buf_gpu;
				cuda::GpuMat B;
				cuda::add(vtemp[1],vtemp[2],buf_gpu);
				cuda::divide(buf_gpu,2,buf_gpu);
				cuda::subtract(vtemp[0],buf_gpu, B);
				cuda::GpuMat Y;
				cuda::subtract(vtemp[2],vtemp[1],buf_gpu);
				cuda::divide(buf_gpu,2,buf_gpu);
				cuda::abs(buf_gpu,Y);
				cuda::add(vtemp[1],vtemp[2],buf_gpu);
				cuda::divide(buf_gpu,2,buf_gpu);
				cuda::subtract(buf_gpu,Y,Y);
				cuda::subtract(Y,vtemp[0],Y);
				cuda::GpuMat R;
				cuda::add(vtemp[0],vtemp[1],buf_gpu);
				cuda::divide(buf_gpu,2,buf_gpu);
				cuda::subtract(vtemp[2],buf_gpu,R);
				cuda::GpuMat G;
				cuda::add(vtemp[0],vtemp[2],buf_gpu);
				cuda::divide(buf_gpu,2,buf_gpu);
				cuda::subtract(vtemp[1],buf_gpu,G);
				cuda::subtract(B,Y,buf_gpu);
				Pyr_C[0].push_back(buf_gpu);
				cuda::subtract(R,G,buf_gpu);
				Pyr_C[1].push_back(buf_gpu);
				vtemp.clear();

				// Make Orientation Map -> #4
				for(int k=0; k<Pyr_O.size(); k++){
					Mat kernel = getGaborKernel(Size(20,20),1, CV_PI/4*k, 30, 0,CV_PI/2);
					Mat buf(Pyr_I[i]);
					filter2D(buf, buf, CV_32F, kernel);
					buf_gpu.upload(buf);
					Pyr_O[k].push_back(buf_gpu);
				}
			}
			GausnPyr.clear();

			t11 = std::chrono::high_resolution_clock::now();
			cout << "Step2: "<< std::chrono::duration<float>(t11-t00).count()<<"[s]" << endl;
			t00 = t11;

			// Step 3. Center-Surrounded Difference
			vector<cuda::GpuMat> CSD_I,CSD_C,CSD_O;
			CSD_I = centerSurround(Pyr_I,Pyr_I); // 8->6
			for(int k=0; k<Pyr_C.size(); k++) {
				vector<cuda::GpuMat> inv_Pyr_C(Pyr_C[k].size());
				for(int l=0; l<Pyr_C[k].size(); l++){
					cuda::subtract(Scalar(0),Pyr_C[k][l],inv_Pyr_C[l]);
					cuda::abs(inv_Pyr_C[l],inv_Pyr_C[l]);
				}
				Pyr_C[k] = centerSurround(Pyr_C[k],inv_Pyr_C); //R-G and G-R, B-Y and Y-B
			}
			for(int k=0; k<Pyr_C.size(); k++) for(int l=0; l<Pyr_C[k].size(); l++) CSD_C.push_back(Pyr_C[k][l]);
			for(int k=0; k<Pyr_O.size(); k++) Pyr_O[k] = centerSurround(Pyr_O[k],Pyr_O[k]);
			for(int k=0; k<Pyr_O.size(); k++) for(int l=0; l<Pyr_O[k].size(); l++) CSD_O.push_back(Pyr_O[k][l]);
			Pyr_I.clear(); Pyr_C.clear(); Pyr_O.clear();

			t11 = std::chrono::high_resolution_clock::now();
			cout << "Step3: "<< std::chrono::duration<float>(t11-t00).count()<<"[s]" << endl;
			t00 = t11;

			// Step 4. Normalization
			normalizeMap(CSD_I);
			normalizeMap(CSD_C);
			normalizeMap(CSD_O);

			t11 = std::chrono::high_resolution_clock::now();
			cout << "Step4: "<< std::chrono::duration<float>(t11-t00).count()<<"[s]" << endl;
			t00 = t11;

			// Step 5. Conspicuity Maps
			cuda::GpuMat I(Size(CSD_I[0].cols,CSD_I[0].rows), CSD_I[0].type(),Scalar::all(0));
			cuda::GpuMat C(Size(CSD_C[0].cols,CSD_C[0].rows), CSD_C[0].type(),Scalar::all(0));
			cuda::GpuMat O(Size(CSD_O[0].cols,CSD_O[0].rows), CSD_O[0].type(),Scalar::all(0));

			for(int i=0; i<CSD_I.size(); i++) cuda::add(I,CSD_I[i],I);
			for(int i=0; i<CSD_C.size(); i++) cuda::add(C,CSD_C[i],C);
			for(int i=0; i<CSD_O.size(); i++) cuda::add(O,CSD_O[i],O);
			CSD_I.clear(); CSD_C.clear(); CSD_O.clear();

			t11 = std::chrono::high_resolution_clock::now();
			cout << "Step5: "<< std::chrono::duration<float>(t11-t00).count()<<"[s]" << endl;
			t00 = t11;

			// Step 6. Merge
			cuda::normalize(I,I,0,255,NORM_MINMAX,CV_8UC1);
			cuda::normalize(C,C,0,255,NORM_MINMAX,CV_8UC1);
			cuda::normalize(O,O,0,255,NORM_MINMAX,CV_8UC1);
			cuda::GpuMat Salmap_gpu;
			cuda::add(I,C,Salmap_gpu);
			cuda::add(Salmap_gpu,O,Salmap_gpu);
			cuda::divide(Salmap_gpu,3,Salmap_gpu);
			Mat Salmap;
			Salmap_gpu.download(Salmap);

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
		    vector<Mat> result={frame,Salmap,Salmap_RGB};
			cuda::cvtColor(I,I,COLOR_GRAY2BGR);
			cuda::cvtColor(C,C,COLOR_GRAY2BGR);
			cuda::cvtColor(O,O,COLOR_GRAY2BGR);
			Mat Icpu(I);
			Mat Ccpu(C);
			Mat Ocpu(O);
		    vector<Mat> fmap={Icpu,Ccpu,Ocpu};

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
	return 0;
}

void makeGaussianPyramid(cuda::GpuMat& src, int s, vector<cuda::GpuMat>& pyr){
	src.copyTo(pyr[0]);
	for(int i=1; i<s; i++){
		cuda::pyrDown(pyr[i-1],pyr[i]);
		cuda::resize(pyr[i],pyr[i],Size(pyr[0].cols,pyr[0].rows));
	}
}

vector<cuda::GpuMat> centerSurround(vector<cuda::GpuMat>& fmap1, vector<cuda::GpuMat>& fmap2){
	vector<int> center = {2,3,4};
	vector<int> delta = {3,4};
	vector<cuda::GpuMat> CSD;
	for(int c=0; c < center.size(); c++)
		for(int d=0; d<delta.size(); d++)
		{
			cuda::GpuMat ctemp = fmap1[center[c]];
			cuda::GpuMat stemp = fmap2[delta[d]+center[c]];
			cuda::GpuMat temp;
			cuda::subtract(ctemp,stemp,temp);
			cuda::abs(temp,temp);
			CSD.push_back(temp);
			temp.release();
		}
	return CSD;
}

void normalizeMap(vector<cuda::GpuMat>& nmap){
	for(int i=0; i<nmap.size(); i++)
	{
		cuda::normalize(nmap[i],nmap[i],0,1,NORM_MINMAX,CV_32FC1);
		Scalar meanVal = cuda::sum(nmap[i]);
		double mean = (meanVal[0] / nmap[i].cols*nmap[i].rows);
		cuda::multiply(nmap[i],Scalar(pow((1-mean),2)),nmap[i]);
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
				rgb.at<Vec3b>(j, i)[1] = 5;
				rgb.at<Vec3b>(j, i)[2] = 20;
			}
    	}
    return rgb;
}

