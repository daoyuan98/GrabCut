#include "GrabCut.h"
#include "GMM.h"
#include "graph.h"
using namespace std;
using namespace cv;

const int K = 5;

GrabCut2D::~GrabCut2D(void)
{
}

bool in_rect(int i, int j, Rect rect) {
	return
		i >= rect.tl().x && i <= rect.br().x
		&& j >= rect.tl().y && j <= rect.br().y;
}

//ȷ��������0�� ȷ��ǰ����1�����ܱ�����2������ǰ����3
void SamplePoints(Mat& img, Mat& mask, vector<Vec3f>& bgds,vector<Vec3f>&fgds,Mat&bgd_best_labels, Mat&fgd_best_labels) {
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (mask.at<uchar>(x, y) % 2 == 0) //ȷ�������Ϳ��ܱ���
				bgds.push_back(Vec3f(img.at<Vec3b>(x, y)));
			else
				fgds.push_back(Vec3f(img.at<Vec3b>(x, y)));
		}
	}
	//cout << bgds.size() <<"   "<< fgds.size() << endl;
	kmeans(bgds, K, bgd_best_labels, TermCriteria(CV_TERMCRIT_ITER, 10, 0.0), 0, KMEANS_PP_CENTERS);
	kmeans(fgds, K, fgd_best_labels, TermCriteria(CV_TERMCRIT_ITER, 10, 0.0), 0, KMEANS_PP_CENTERS);
}

static double calc_beta(cv::Mat img) {
	double total_diff = 0.0;
	Vec3b this_diff;
	int count = 0;
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			Vec3b it = img.at<Vec3b>(i, j);
			if (i > 0) {//���
				this_diff = it - img.at<Vec3b>(i - 1, j);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i > 0 && j > 0) {//����
				this_diff = it - img.at<Vec3b>(i - 1, j - 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i > 0 && j + 1 < img.rows) {//����
				this_diff = it - img.at<Vec3b>(i - 1, j + 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (j > 0) {//����
				this_diff = it - img.at<Vec3b>(i, j - 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (j + 1 < img.rows) {//����
				this_diff = it - img.at<Vec3b>(i, j + 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i + 1 < img.cols) {//����
				this_diff = it - img.at<Vec3b>(i + 1, j);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i + 1 < img.cols && j > 0) {//����
				this_diff = it - img.at<Vec3b>(i + 1, j - 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
			if (i + 1 < img.cols && j + 1 < img.rows) {//����
				this_diff = it - img.at<Vec3b>(i + 1, j + 1);
				total_diff += this_diff.dot(this_diff);
				count++;
			}
		}
	}
	return total_diff / count;
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

void calc_smoothTerm(Mat& img,Mat& leftW, Mat& upleftW,Mat& upW, Mat& uprightW,double beta,double gamma = 50) {
	double dist = sqrt(2);

	leftW.create(img.size(), CV_64FC1);
	upleftW.create(img.size(), CV_64FC1);
	uprightW.create(img.size(), CV_64FC1);
	upW.create(img.size(), CV_64FC1);

	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			if (i > 0) {//��һ��
				leftW.at<double>(i, j) = gamma * \
					exp(-beta * (mod(img.at<Vec3b>(i, j) - img.at<Vec3b>(i - 1, j))));
			}
			else leftW.at<double>(i, j) = 0;

			if (i > 0 && j > 0) {//����
				upleftW.at<double>(i, j) = gamma * \
					exp(-beta * (mod(img.at<Vec3b>(i, j) - img.at<Vec3b>(i - 1, j-1))))/dist;
			}
			else upleftW.at<double>(i, j) = 0;

			if (j > 0) {//����
				upW.at<double>(i, j) = gamma * \
					exp(-beta * (mod(img.at<Vec3b>(i, j) - img.at<Vec3b>(i, j-1))));
			}
			else upW.at<double>(i, j) = 0;

			if (j > 0 && i + 1 < img.cols) {//����
				uprightW.at<double>(i, j) = gamma * \
					exp(-beta * (mod(img.at<Vec3b>(i, j) - img.at<Vec3b>(i +1, j-1))))/dist;
			}
			else uprightW.at<double>(i, j) = 0;

		}
	}
}
#ifndef CUTGRAPH_H_
#define CUTGRAPH_H_
#include "graph.h"
class CutGraph {
private:
	Graph<double, double, double> * graph;
public:
	CutGraph();
	CutGraph(int, int);
	int addVertex();
	double maxFlow();
	void addVertexWeights(int, double, double);
	void addEdges(int, int, double);
	bool isSourceSegment(int);
};
#endif
CutGraph::CutGraph() {}
CutGraph::CutGraph(int _vCount, int _eCount) {
	graph = new Graph<double, double, double>(_vCount, _eCount);
}
int CutGraph::addVertex() {
	return graph->add_node();
}
double CutGraph::maxFlow() {
	return graph->maxflow();
}
void CutGraph::addVertexWeights(int _vNum, double _sourceWeight, double _sinkWeight) {
	graph->add_tweights(_vNum, _sourceWeight, _sinkWeight);
}
void CutGraph::addEdges(int _vNum1, int _vNum2, double _weight) {
	graph->add_edge(_vNum1, _vNum2, _weight, _weight);
}
bool CutGraph::isSourceSegment(int _vNum) {
	if (graph->what_segment(_vNum) == Graph<double, double, double>::SOURCE)return true;
	else return false;
}

static void getGraph(const Mat& _img, const Mat& _mask, const GMM& _bgdGMM, const GMM& _fgdGMM, double _lambda,
	const Mat& _l, const Mat& _ul, const Mat& _u, const Mat& _ur,
	CutGraph& _graph,Mat data_term_s,Mat data_term_t)
{
	int vCount = _img.cols*_img.rows;
	int eCount = 2 * (4 * vCount - 3 * _img.cols - 3 * _img.rows + 2);
	_graph = CutGraph(vCount, eCount);
	Point p;
	for (p.y = 0; p.y < _img.rows; p.y++) {
		for (p.x = 0; p.x < _img.cols; p.x++) {
			int vNum = _graph.addVertex();
			Vec3b color = _img.at<Vec3b>(p);
			double wSource = 0, wSink = 0;
			if (_mask.at<uchar>(p) == 2 || _mask.at<uchar>(p) == 3) {
				wSource = data_term_s.at<double>(p);
				wSink = data_term_t.at<double>(p);
			}
			else if (_mask.at<uchar>(p) == 0) wSink = _lambda;
			else wSource = _lambda;
			_graph.addVertexWeights(vNum, wSource, wSink);
			if (p.x > 0) {
				double w = _l.at<double>(p);
				_graph.addEdges(vNum, vNum - 1, w);
			}
			if (p.x > 0 && p.y > 0) {
				double w = _ul.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols - 1, w);
			}
			if (p.y > 0) {
				double w = _u.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols, w);
			}
			if (p.x < _img.cols - 1 && p.y > 0) {
				double w = _ur.at<double>(p);
				_graph.addEdges(vNum, vNum - _img.cols + 1, w);
			}
		}
	}
}
//���зָ� Done
static void estimateSegmentation(CutGraph& _graph, Mat& _mask) {
	_graph.maxFlow();
	Point p;
	for (p.y = 0; p.y < _mask.rows; p.y++) {
		for (p.x = 0; p.x < _mask.cols; p.x++) {
			if (_mask.at<uchar>(p) == 2 || _mask.at<uchar>(p) == 3) {
				if (_graph.isSourceSegment(p.y*_mask.cols + p.x))
					_mask.at<uchar>(p) = 3;
				else _mask.at<uchar>(p) = 2;
			}
		}
	}
}


void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray& _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
	//1.Load Input Image: ����������ɫͼ��;
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	cout << "      " << "Load Input Image" << endl;
	
	//2.Init Mask: �þ��ο��ʼ��Mask��Labelֵ��ȷ��������0�� ȷ��ǰ����1�����ܱ�����2������ǰ����3��,���ο���������Ϊȷ�����������ο���������Ϊ����ǰ��;
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			if (in_rect(j, i, rect)) 
				mask.at<uchar>(i, j) = 3;   //���ο��ڣ�����ǰ����3
			else
				mask.at<uchar>(i, j) = 0;   //���ο��⣬ȷ��������0
		}
	}
	cout << "      " << "Init Mask" << endl;
	
	//3.Init GMM: ���岢��ʼ��GMM(����ģ����ɷָ�Ҳ�ɵõ�����������GMM��ɻ�ӷ֣�
	GMM fgd(5), bgd(5);
	cout << "      " << "Init GMM" << endl;

	//4.Sample Points:ǰ������ɫ���������о��ࣨ������kmeans���������෽��Ҳ��)

	//They match in size.
	vector<Vec3f> bgds, fgds;
	Mat bgd_best_labels;
	Mat fgd_best_labels;

	for (int n_iter = 0; n_iter < iterCount; n_iter++) {
		SamplePoints(img, mask, bgds, fgds, bgd_best_labels, fgd_best_labels);
		cout << "      " << "Sample Points" << endl;

		//5.Learn GMM(���ݾ������������ÿ��GMM component�еľ�ֵ��Э����Ȳ�����

		//bgd:ÿ���������������������
		for (int i = 0; i < bgd_best_labels.rows; i++) {
			bgd.addPoint(bgd_best_labels.at<int>(i, 0), bgds[i]);
		}
		bgd.update_params();
		cout << "      " << "Learn GMM: bgd" << endl;

		//fgd:ÿ���������ٵ�������
		for (int i = 0; i < fgd_best_labels.rows; i++) {
			fgd.addPoint(fgd_best_labels.at<int>(i, 0), fgds[i]);
		}
		fgd.update_params();
		cout << "      " << "Learn GMM: fgd" << endl;

		//6.Construct Graph������t-weight(�������n-weight��ƽ����))
		double beta = calc_beta(img);

		cout << "beta: " << beta << endl;

		//sǰ�� t����
		Mat data_term_s(img.size(), CV_64FC1), data_term_t(img.size(), CV_64FC1);
		
		bgd.calc_dataTerm(img,data_term_t);
		
		fgd.calc_dataTerm(img,data_term_s);

		//Mat data_term_s = Mat::zeros(img.size(), CV_64FC1);
		//Mat data_term_t = Mat::zeros(img.size(), CV_64FC1);

		cout << "      " << "Data Term" << endl;

		Mat leftW, upleftW, upW, uprightW;
		calc_smoothTerm(img, leftW, upleftW, upW, uprightW, beta, 50);

		cout << "      " << "Smooth Term" << endl;
		

		//7.Estimate Segmentation(����maxFlow����зָ�)
		CutGraph graph;

		getGraph(img, mask, bgd, fgd, 9 * 50, leftW, upleftW, upW, uprightW, graph, data_term_s, data_term_t);
		estimateSegmentation(graph, mask);

		/*for (int yy = 0; yy < mask.cols; yy++) {
			for (int xx = 0; xx < mask.rows; xx++) {
				cout << (int)(mask.at<uchar>(0, 0)) << "  ";
			}
			cout << endl;
		}*/
		
		cout << "      " << "Estimate Segmentation" << endl;
		
		//8.Save Result�������������mask�������mask��ǰ�������Ӧ�Ĳ�ɫͼ�񱣴����ʾ�ڽ��������У�
	}
}