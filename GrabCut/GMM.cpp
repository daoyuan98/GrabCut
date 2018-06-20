#include "GMM.h"
#include <cstring>
#include <iostream>


GMM::GMM(int K) {
	coeff = new double[K];
	mean = new cv::Vec3f[K];
	covar = new double[9*K];
	this->K = K;
	component_points = new vecOfPix[K];
	
	for (int i = 0; i < K; i++) {
		coeff[i] = 1.0 / K;
	}
	//erease the memory
	memset(mean, 0, sizeof(mean));	// ??应该为0吗？
	memset(covar, 0, sizeof(covar));   // ??同上
}

void GMM::addPoint(int component_index, cv::Vec3f& point) {
	assert(component_index >= 0 && component_index < this->K);
	component_points[component_index].push_back(point);
}

void GMM::update_params() {//component_points中的数据更新权重、均值、方差这些参数
	int total_points = 0;
	for (int i = 0; i < K; i++)
		total_points += component_points[i].size();
	//cout << "total_points: " << total_points << endl;
	for (int i = 0; i < K; i++) {   //更新每个component的参数

		int n_points = component_points[i].size();

		//cout << n_points << endl;

		//更新系数
		coeff[i] = 1.0*n_points / total_points;

		//cout << "        " << "Coeffs" << endl;

		//分别更新三个分量的均值
		int a1, a2, a3;
		a1 = a2 = a3 = 0;

		//用于分别计算三个分量的总和，以求均值
		for (int n = 0; n < n_points; n++) {
			a1 += component_points[i][n][0];
			a2 += component_points[i][n][1];
			a3 += component_points[i][n][2];
		}

		//更新均值
		mean[i][0] = 1.0*a1 / n_points;  //?为何乘10？
		mean[i][1] = 1.0*a2 / n_points;
		mean[i][2] = 1.0*a3 / n_points;

		//cout << "        " << "Means" << endl;

		for (int y = 0; y < 3; y++) {
			for (int x = y; x < 3; x++) {
				double var = 0.0;
				int nx = component_points[i].size();
				int ny = nx;

				double sum = 0;
				for (int i1 = 0; i1 < nx; i1++) {
					for (int i2 = 0; i2 < ny; i2++) {
						sum += component_points[i][i1][x] * component_points[i][i2][y];
					}
				}
				var = 1.0*sum / (nx*ny) - mean[i][x] * mean[i][y];
				//cout << "covar: "<<var << endl;
				covar[i*9+ y * 3 + x] = covar[i*9+x * 3 + y] = var;
			}
		}
	}
}


cv::Point find_coordinate(cv::Mat& img, cv::Vec3f& color) {
	cv::Point p(0, 0);
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			if (img.at<cv::Vec3f>(p) == color) {
				return p;
			}
		}
	}
}

void printMat(cv::Mat& m) {
	for (int j = 0; j < m.rows; j++) {
		cout << "[ ";
		for (int i = 0; i < m.cols; i++)
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
		/*
		cout << "before: " << color << endl;
		cout << "miu   : " << m << endl;*/
		diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
		//cout << "after: " << diff << endl;
		
		cv::Mat sigma = cv::Mat::zeros(3, 3, CV_64FC1);
		for (int i = 0; i < 9; i++) {
			sigma.at<double>(i / 3, i % 3) = covar[nk * 9 + i];
		}


		//cout << "sigma: ";
		//printMat(sigma);

		cv::Mat sigma_inv = sigma.inv();
		//cout << endl << "sigma_inv: " << endl;
		//printMat(sigma_inv);

		double det_sigma = cv::determinant(sigma);
		//cout << "Det(sigma): " << det_sigma << endl;

		double mult = diff[0] * (diff[0] * sigma_inv.at<double>(0,0) + diff[1] * sigma_inv.at<double>(1,0) + diff[2] * sigma_inv.at<double>(2, 0))
			+ diff[1] * (diff[0] * sigma_inv.at<double>(0,1) + diff[1] * sigma_inv.at<double>(1, 1) + diff[2] * sigma_inv.at<double>(2, 1))
			+ diff[2] * (diff[0] * sigma_inv.at<double>(0,2) + diff[1] * sigma_inv.at<double>(1, 2) + diff[2] * sigma_inv.at<double>(2, 2));

		//res = mult / 2;
		//cout << "mult: " << mult << endl;

		double upper = exp(-mult / 2);
		double downer = (pow(6.28, 1.5)*sqrt(det_sigma));

		//cout << "upper: " << upper << endl;
		//cout << "downer: " << downer << endl;

		res = 1.5*log(2 * 3.1415)/2.7183 + det_sigma / 2 + mult / 2;
		//cout << "ret val: " << res << endl;  大概e+7,e+8的大小

		//cout << "----------------------" << endl << endl;

	}

	
	return res;


	//cv::Vec3f miu = mean[nk];
	//cout << "miu: " << miu << endl;
	//cv::Vec3f this_color(color);
	//cv::Mat diff(cv::Size(3, 1), CV_64FC1);

	//diff.at<double>(0, 0) = 1.0*color[0] - miu[0];
	//diff.at<double>(0, 1) = 1.0*color[1] - miu[1];
	//diff.at<double>(0, 2) = 1.0*color[2] - miu[2];

	//cv::Mat sigma = MatFromArray(covar, nk);

	//if (diff.rows == 3)//三行一列的话变成一行三列 
 //	     diff = diff.t();
	//cv::Mat tt = diff * sigma.inv();
	//cv::Mat res = tt*diff.t();
	//double prob_term = res.at<double>(0, 0);
	//prob_term = 1.0f / sqrt(cv::determinant(sigma))*exp(-0.5*prob_term);
	//cout << "prob_term : " << prob_term << endl;
	//return prob_term;
}

void GMM::calc_dataTerm(cv::Mat& img,cv::Mat& m){
	//代码上的思路：
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			double data_term = 0.0;

			for (int nk = 0; nk < K; nk++) {
				data_term += coeff[nk] * prob(nk, cv::Vec3f(img.at<cv::Vec3b>(i, j)));
			}

			m.at<double>(i, j) = data_term;
		}
	}

	
	//自己实现的
	//double data_term = 0.0;

	//for (int i = 0; i < K; i++) {
	//	double pi = coeff[i];
	//	cv::Vec3f miu = mean[i];

	//	cout << "miu: " << miu << endl;

	//	cv::Mat sigma = MatFromArray(covar, i);
	//	for (int n = 0; n < component_points[i].size(); n++) {
	//		cv::Mat diff(cv::Size(3, 1), CV_64FC1);

	//		diff.at<double>(0, 0) = component_points[i][n][0] - miu[0];
	//		diff.at<double>(0, 1) = component_points[i][n][1] - miu[1];
	//		diff.at<double>(0, 2) = component_points[i][n][2] - miu[2];

	//		if (diff.rows == 3)//三行一列的话变成一行三列 
	//			diff = diff.t();
	//		cv::Mat tt = diff * sigma.inv();
	//		cv::Mat res = tt*diff.t();
	//		double prob_term = res.at<double>(0, 0) / 2;
	//		
	//		//cout << "probi: " << prob_term << endl;

	//		double t = -log(pi) + log(cv::determinant(sigma)) / 2 + prob_term;
	//		
	//		cv::Point  p = find_coordinate(img, component_points[i][n]);
	//		m.at<double>(p) = t;
	//	}
	//}
	//return data_term;
}


GMM::~GMM()
{
}
