#ifndef  GMM_H
#define GMM_H
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

typedef vector<cv::Vec3f> vecOfPix;

class GMM
{
private:
	int K;	//number of SGM
	vecOfPix* component_points;
	const double gamma = 50;
public:
	double *coeff;
	cv::Vec3f* mean;
	double *covar;
	
	GMM(int K = 5);
	~GMM();
	void addPoint(int component_index, cv::Vec3f& point);
	void update_params();
	double prob(int nk, cv::Vec3f color);
	void calc_dataTerm(cv::Mat& img,cv::Mat& m);
};



#endif // ! GMM_H


