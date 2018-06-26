#ifndef  GMM_H
#define GMM_H
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

//typedef vector<cv::Vec3f> vecOfPix;

class GMM
{
private:
	static const int K = 5;
	//vecOfPix* component_points;
	const double gamma = 50;

	double sum[K][3];
	double prods[K][3][3];
	int n_sample[K];
	int total_points;
	double det[K];
	cv::Mat Inv[K];
public:
	double *coeff;
	cv::Vec3f* mean;
	double *covar;
	
	GMM(int K = 5);
	~GMM();
	void addPoint(int component_index, cv::Vec3d& point);
	void update_params();
	double prob(int nk, cv::Vec3f color);
	void calc_dataTerm(cv::Mat& img,cv::Mat& m);
	void calc_det(int i);
	void calc_inv(int i);
	double get_weight(cv::Vec3b& color);
	int choice(cv::Vec3d color);
	void clear();
};



#endif // ! GMM_H


