#include "GMM.h"
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>


GMM::GMM(int K) {
	coeff = new double[K];
	mean = new cv::Vec3f[K];
	covar = new double[9*K];
	
	for (int i = 0; i < K; i++) {
		coeff[i] = 1.0 / K;
		n_sample[i] = 0;
		Inv[i].create(3, 3, CV_64FC1);
		for (int i1 = 0; i1 < 3; i1++) {
			sum[i][i1] = 0;
			for (int i2 = 0; i2 < 3; i2++) {
				prods[i][i1][i2] = 0.0;
			}
		}
	}
	//erease the memory
	memset(mean, 0, sizeof(mean));	// ??应该为0吗？
	memset(covar, 0, sizeof(covar));   // ??同上
	total_points = 0;
}

double GMM::get_weight(cv::Vec3b& color) {
	double res = 0.0;
	for (int i = 0; i < K; i++) {
		res += coeff[i] * prob(i, color);
	}
	return res;
}

int GMM::choice(cv::Vec3d _color){
	int k = 0;
	double max = 0;
	for (int i = 0; i < K; i++) {
		double p = prob(i, _color);
		if (p > max){
			k = i;
			max = p;
		}
	}
	return k;
}

void GMM::addPoint(int component_index, cv::Vec3d& point) {
	assert(component_index >= 0 && component_index < K);
	for (int i = 0; i < 3; i++) {
		sum[component_index][i] += point[i];
		for (int j = 0; j < 3; j++) {
			prods[component_index][i][j] += point[i] * point[j];
		}
	}
	n_sample[component_index]++;
	total_points++;
}

void GMM::update_params() {//component_points中的数据更新权重、均值、方差这些参数
	for (int i = 0; i < K; i++) {   //更新每个component的参数
		//更新系数
		int n_ = n_sample[i];
		if (!n_) {
			coeff[i] = 0;
			return;
		}
		coeff[i] = 1.0*n_ / total_points;

		//更新均值
		for (int j = 0; j < 3; j++)
			mean[i][j] = sum[i][j] / n_;

		for (int i1 = 0; i1 < 3; i1++) {
			for (int i2 = 0; i2 < 3; i2++) {
				double t = prods[i][i1][i2] / n_ - mean[i][i1] * mean[i][i2];
				covar[i * 9 + i1 * 3 + i2] = t;
			}
		}

		calc_det(i);
		
	}
}


void GMM::calc_inv(int i) {
	cv::Mat m = cv::Mat::ones(3, 3, CV_64FC1);
	for (int i_ = 0; i_ < 9; i_++) {
		m.at<double>(i_ / 3, i_ % 3) = covar[i * 9 + i_];
	}
	Inv[i] = m.inv();//?
}

void GMM :: calc_det(int i) {
	double* thismat = covar + i * 9;
	det[i] = thismat[0] * (thismat[4]*thismat[8] - thismat[5] * thismat[7])
		- thismat[1] * (thismat[3] * thismat[8] - thismat[5] * thismat[6])
		+ thismat[2] * (thismat[3] * thismat[7] - thismat[4] * thismat[6]);
	Inv[i].at<double>(0, 0) = (thismat[4] * thismat[8] - thismat[5] * thismat[7]) / det[i];
	Inv[i].at<double>(1, 0) = -(thismat[3] * thismat[8] - thismat[5] * thismat[6]) / det[i];
	Inv[i].at<double>(2, 0) = (thismat[3] * thismat[7] - thismat[4] * thismat[6]) / det[i];
	Inv[i].at<double>(0, 1) = -(thismat[1] * thismat[8] - thismat[2] * thismat[7]) / det[i];
	Inv[i].at<double>(1, 1) = (thismat[0] * thismat[8] - thismat[2] * thismat[6]) / det[i];
	Inv[i].at<double>(2, 1) = -(thismat[0] * thismat[7] - thismat[1] * thismat[6]) / det[i];
	Inv[i].at<double>(0, 2) = (thismat[1] * thismat[5] - thismat[2] * thismat[4]) / det[i];
	Inv[i].at<double>(1, 2) = -(thismat[0] * thismat[5] - thismat[2] * thismat[3]) / det[i];
	Inv[i].at<double>(2, 2) = (thismat[0] * thismat[4] - thismat[1] * thismat[3]) / det[i];
}

void printMat(cv::Mat& m) {
	for (int i = 0; i < m.rows; i++) {
		cout << "[ ";
		for (int j = 0; j < m.cols; j++)
			cout << m.at<double>(i, j) << ", ";
		cout << "]" << endl;
	}
	cout << endl;
}

double GMM::prob(int nk, cv::Vec3f color) {
	double res = 0;
	if (coeff[nk] > 0) {
		cv::Vec3f diff = color;
		cv::Vec3f m = mean[nk];
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];

		cv::Mat sigma_inv = Inv[nk];

		double det_sigma = det[nk];

		double mult = diff[0] * (diff[0] * sigma_inv.at<double>(0, 0) + diff[1] * sigma_inv.at<double>(1, 0) + diff[2] * sigma_inv.at<double>(2, 0))
			+ diff[1] * (diff[0] * sigma_inv.at<double>(0, 1) + diff[1] * sigma_inv.at<double>(1, 1) + diff[2] * sigma_inv.at<double>(2, 1))
			+ diff[2] * (diff[0] * sigma_inv.at<double>(0, 2) + diff[1] * sigma_inv.at<double>(1, 2) + diff[2] * sigma_inv.at<double>(2, 2));

		res = 1.0f / sqrt(det_sigma)*exp(-1.0*mult / 2);

	}	
	return res;
}

void GMM::clear() {
	total_points = 0;
	for (int i = 0; i < K; i++) {
		det[i] = 0.0;
		n_sample[i] = 0;
		for (int i1 = 0; i1 < 3; i1++) {
			sum[i][i1] = 0.0;
			sum[i][i1] = 0.0;
			for (int i2 = 0; i2 < 3; i2++) {
				prods[i][i1][i] = 0.0;
			}
		}
	}
}


void GMM::calc_dataTerm(cv::Mat& img,cv::Mat& m){
	
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			double data_term = 0.0;

			for (int nk = 0; nk < K; nk++) {
				data_term += prob(nk, (cv::Vec3f)(img.at<cv::Vec3b>(i, j)));
			}
			double t = -log(data_term);
			m.at<double>(i, j) = t;
		}
	}
}


GMM::~GMM()
{
}
