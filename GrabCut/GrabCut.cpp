#include "GrabCut.h"
#include "GMM.h"
#include "graph.h"

using namespace std;
using namespace cv;

const int K = 5;

GrabCut2D::~GrabCut2D(void)
{
}

bool in_rect(int j, int i, Rect& rect ) {
	return
		i >= rect.tl().x && i <= rect.br().x
		&& j >= rect.tl().y && j <= rect.br().y;
}

//确定背景：0， 确定前景：1，可能背景：2，可能前景：3
void SamplePoints(Mat& img, Mat& mask, vector<Vec3f>& bgds,vector<Vec3f>&fgds,Mat&bgd_best_labels, Mat&fgd_best_labels) {
	Point p;
	for (p.y = 0; p.y < img.rows; p.y++) {
		for (p.x = 0; p.x < img.cols; p.x++) {
			if (mask.at<uchar>(p) % 2 == 0) //确定背景和可能背景
				bgds.push_back((Vec3f)img.at<Vec3b>(p));
			else
				fgds.push_back((Vec3f)img.at<Vec3b>(p));
		}
	}
	//cout << bgds.size() <<"   "<< fgds.size() << endl;
	kmeans(bgds, K, bgd_best_labels, TermCriteria(CV_TERMCRIT_ITER, 10, 0.0), 0, KMEANS_PP_CENTERS);
	kmeans(fgds, K, fgd_best_labels, TermCriteria(CV_TERMCRIT_ITER, 10, 0.0), 0, KMEANS_PP_CENTERS);
}

static double calc_beta(cv::Mat img) {
	double total_diff = 0.0;
	Vec3d this_diff;
	int count = 0;
	Point p;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3d it = (Vec3d)img.at<Vec3b>(i, j);
			if (i > 0) {//左边
				this_diff = it - (Vec3d)img.at<Vec3b>(i - 1, j);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i > 0 && j > 0) {//左上
				this_diff = it - (Vec3d)img.at<Vec3b>(i - 1, j - 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i > 0 && j + 1 < img.rows) {//左下
				this_diff = it - (Vec3d)img.at<Vec3b>(i - 1, j + 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (j > 0) {//正上
				this_diff = it - (Vec3d)img.at<Vec3b>(i, j - 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (j + 1 < img.cols) {//正下
				this_diff = it - (Vec3d)img.at<Vec3b>(i, j + 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i + 1 < img.rows) {//正右
				this_diff = it - (Vec3d)img.at<Vec3b>(i + 1, j);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i + 1 < img.rows && j > 0) {//右上
				this_diff = it - (Vec3d)img.at<Vec3b>(i + 1, j - 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i + 1 < img.rows && j + 1 < img.cols) {//右下
				this_diff = it - (Vec3d)img.at<Vec3b>(i + 1, j + 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
		}
	}
	return 1.0/(total_diff / count);
}

double mod(Vec3f& t) {
	return t.dot(t);
}

double mod(Vec3d& t) {
	return t.dot(t);
}

double mod(Vec3b& t) {
	return t.dot(t);
}

void calc_smoothTerm(Mat& img,Mat& leftW, Mat& upleftW,Mat& upW, Mat& uprightW,double beta,double gamma = 500) {
	double dist = sqrt(2);

	leftW.create(img.size(), CV_64FC1);
	upleftW.create(img.size(), CV_64FC1);
	uprightW.create(img.size(), CV_64FC1);
	upW.create(img.size(), CV_64FC1);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (i > 0) {//上一格
				upW.at<double>(i, j) = gamma * \
					exp(-beta * (mod(img.at<Vec3b>(i, j) - img.at<Vec3b>(i - 1, j))));
			}
			else upW.at<double>(i, j) = 0;

			if (i > 0 && j > 0) {//左上
				upleftW.at<double>(i, j) = gamma * \
					exp(-beta * (mod(img.at<Vec3b>(i, j) - img.at<Vec3b>(i - 1, j-1))))/dist;
			}
			else upleftW.at<double>(i, j) = 0;

			if (j > 0) {//左
				leftW.at<double>(i, j) = gamma * \
					exp(-beta * (mod(img.at<Vec3b>(i, j) - img.at<Vec3b>(i, j-1))));
			}
			else leftW.at<double>(i, j) = 0;

			if (i > 0 && j + 1 < img.cols) {//右上
				uprightW.at<double>(i, j) = gamma * \
					exp(-beta * (mod(img.at<Vec3b>(i, j) - img.at<Vec3b>(i - 1, j+1))))/dist;
			}
			else uprightW.at<double>(i, j) = 0;

		}
	}
}

//static void assignGMMS(const Mat& _img, const Mat& _mask, GMM& _bgdGMM,GMM& _fgdGMM, Mat& _partIndex) {
//	Point p;
//	for (int x = 0; x < _img.rows; x++) {
//		for (int y = 0; y < _img.cols; y++) {
//			Vec3d color = (Vec3d)_img.at<Vec3b>(x,y);
//			uchar t = _mask.at<uchar>(x,y);
//			if (t == 0 || t == 2)_partIndex.at<int>(x,y) = _bgdGMM.choice(color);
//			else _partIndex.at<int>(x,y) = _fgdGMM.choice(color);
//		}
//	}
//}

//迭代循环第二步，根据得到的结果计算GMM参数值。
static void learnGMMs(const Mat& _img, const Mat& _mask, GMM& _bgdGMM, GMM& _fgdGMM, const Mat& _partIndex) {
	_bgdGMM.clear();
	_fgdGMM.clear();
	for (int i = 0; i < 5; i++) {
		for (int x = 0; x < _img.rows; x++) {
			for (int y = 0; y < _img.cols; y++) {
				int tmp = _partIndex.at<int>(x,y);
				if (tmp == i) {
					if (_mask.at<uchar>(x,y) == 0 || _mask.at<uchar>(x,y) == 2)
						_bgdGMM.addPoint(tmp, (Vec3d)_img.at<Vec3b>(x,y));
					else
						_bgdGMM.addPoint(tmp, (Vec3d)_img.at<Vec3b>(x,y));
				}
			}
		}
	}
	_bgdGMM.update_params();
	_fgdGMM.update_params();
}

static void initMaskWithRect(Mat& _mask, Size _imgSize, Rect _rect) {
	_mask.create(_imgSize, CV_8UC1);
	_mask.setTo(0);
	_rect.x = _rect.x > 0 ? _rect.x : 0;
	_rect.y = _rect.y > 0 ? _rect.y : 0;
	_rect.width = _rect.x + _rect.width > _imgSize.width ? _imgSize.width - _rect.x : _rect.width;
	_rect.height = _rect.y + _rect.height > _imgSize.height ? _imgSize.height - _rect.y : _rect.height;
	(_mask(_rect)).setTo(Scalar(3));
}

void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
	//1.Load Input Image: 加载输入颜色图像;
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	//cout << "      " << "Load Input Image" << endl;
	
	////2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
	if (mode == GC_WITH_RECT)
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (in_rect(i, j, rect)) 
					mask.at<uchar>(i, j) = 3;   //矩形框内，可能前景：3
				else
					mask.at<uchar>(i, j) = 0;   //矩形框外，确定背景：0
			}
		}

	//if (mode == GC_WITH_RECT)
		//initMaskWithRect(mask, img.size(), rect);

	cout << "      " << "Init Mask" << endl;
	
	//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
	GMM fgd(5), bgd(5);
	cout << "      " << "Init GMM" << endl;

	//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)

	//They match in size.
	vector<Vec3f> bgds, fgds;
	Mat bgd_best_labels;
	Mat fgd_best_labels;

	for (int n_iter = 0; n_iter < iterCount; n_iter++) {
		SamplePoints(img, mask, bgds, fgds, bgd_best_labels, fgd_best_labels);
		cout << "      " << "Sample Points" << endl;

		//5.Learn GMM(根据聚类的样本更新每个GMM component中的均值、协方差等参数）

		//bgd:
		for (int i = 0; i < bgd_best_labels.rows; i++) {
			bgd.addPoint(bgd_best_labels.at<int>(i, 0), (Vec3d)bgds[i]);
		}
		bgd.update_params();
		cout << "      " << "Learn GMM: bgd" << endl;

		//fgd:
		for (int i = 0; i < fgd_best_labels.rows; i++) {
			fgd.addPoint(fgd_best_labels.at<int>(i, 0), (Vec3d)fgds[i]);
		}
		fgd.update_params();
		cout << "      " << "Learn GMM: fgd" << endl;

		//6.Construct Graph（计算t-weight(数据项）和n-weight（平滑项))
		double beta = calc_beta(img);

		cout << "beta: " << beta << endl;


		cout << "      " << "Data Term" << endl;

		Mat leftW, upleftW, upW, uprightW;
		calc_smoothTerm(img, leftW, upleftW, upW, uprightW, beta, 50);

		cout << "      " << "Smooth Term" << endl;
		

		////7.Estimate Segmentation(调用maxFlow库进行分割)
		
		int n_vertex = img.cols*img.rows;
		int n_edges = 2 * (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2);
		Graph<double,double,double>* graph = new Graph<double, double, double>(n_vertex,n_edges);
		double w_max = 1000.0;
		
		//add s-t-p
		//t - links
		for (int x = 0; x < img.rows; x++) {
			//cout << x << endl;
			for (int y = 0; y < img.cols; y++) {
				double w_src, w_snk;
				w_src = w_snk = 0.0;
				int vertex = graph->add_node(); //vertex : newly added index of vertex
				if (mask.at<uchar>(x, y) > 1) {
					w_src = -log(bgd.get_weight(img.at<Vec3b>(x, y)));
					w_snk = -log(fgd.get_weight(img.at<Vec3b>(x, y)));
				}
				else if (mask.at<uchar>(x, y) == 1) {//确定背景
					w_snk = 0;
					w_src = w_max;
				}
				else if (mask.at<uchar>(x, y) == 0) {//确定前景；
					w_src = 0.0;
					w_snk = w_max;
				}
				//s-t-p t-link
				//add two tlin
				graph->add_tweights(vertex, w_src, w_snk);

			}
		}
		cout << "t links done" << endl;

		//add n-links	
		for (int i = 0; i< img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				double w = 0.0;
				int vertex = i * img.cols + j;
				if (i > 0) {
					w = upW.at<double>(i, j);
					graph->add_edge(vertex, vertex - img.cols, w, w);
				}
				if (i > 0 && j > 0) {
					w = upleftW.at<double>(i, j);
					graph->add_edge(vertex, vertex - img.cols - 1, w,w);
				}
				if (j > 0) {
					w = leftW.at<double>(i, j);
					graph->add_edge(vertex, vertex - 1, w, w);
					//cout << w << endl;
				}
				if (i > 0 && i < img.cols - 1 && j>0) {
					w = uprightW.at<double>(i, j);
					graph->add_edge(vertex, vertex - img.cols + 1, w, w);
				}
			}
		}
		cout << "n links " << endl;

		double res = graph->maxflow();
		cout << "res: " << res << endl;
		cout << "graphs done" << endl;
		for (int x = 0; x < img.rows; x++) {
			for (int y = 0; y < img.cols; y++) {
				if (mask.at<uchar>(x, y) > 1) {
					if (graph->what_segment(x*img.cols + y)) {//1:BKG
						mask.at<uchar>(x, y) = 2;
						//cout << "set" << endl;
					}
					else {
						mask.at<uchar>(x, y) = 3;
						//cout << "set!" << endl;
					}
				}
			}
		}

		cout << "      " << "Estimate Segmentation" << endl;
		
		//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
		
	}
}