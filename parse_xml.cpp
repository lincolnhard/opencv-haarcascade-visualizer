//Date: 2017/05/01
//Author: Lincoln Hard
//email: lincolnhardabc@gmail.com
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
using namespace cv;
#include <iostream>
using namespace std;

#define SCALEFACTOR 20

class Parse_cascade : public CascadeClassifier
{
public:
    int get_windowsize()
        {
        return oldCascade->orig_window_size.width;
        }

    int get_stage_counts()
        {
        return oldCascade->count;
        }

    int get_weakclassifier_counts(int stage_idx)
        {
        return oldCascade->stage_classifier[stage_idx].count;
        }

    bool is_haar_feature_third_rect_exist(int stage_idx, int classifier_idx)
        {
        if (oldCascade->stage_classifier[stage_idx].classifier[classifier_idx].haar_feature->rect[2].weight == 0.0f)
            {
            return false;
            }
        else
            {
            return true;
            }
        }

    bool get_haar_feature_rect_weight_sign(int stage_idx, int classifier_idx, int rect_idx)
        {
        if (oldCascade->stage_classifier[stage_idx].classifier[classifier_idx].haar_feature->rect[rect_idx].weight > 0)
            {
            return true;
            }
        else
            {
            return false;
            }
        }

    CvRect get_haar_feature_rect(int stage_idx, int classifier_idx, int rect_idx)
        {
        return oldCascade->stage_classifier[stage_idx].classifier[classifier_idx].haar_feature->rect[rect_idx].r;
        }
};

int main
    (
    int ac,
    char** av
    )
{
    Parse_cascade cascade;
    if (!cascade.load(av[1]))
        {
        cerr << "Could not load classifier cascade" << endl;
        return EXIT_FAILURE;
        }
    int detect_winsize = cascade.get_windowsize();
    int num_stages = cascade.get_stage_counts();

    namedWindow("cascade layout", WINDOW_NORMAL);
    Mat im = Mat(detect_winsize * SCALEFACTOR, detect_winsize * SCALEFACTOR, CV_8UC1, Scalar(200));
    int i = 0;
    int j = 0;
    int k = 0;
    char textbuf[128];
    for (i = 0; i < num_stages; ++i)
        {
        int num_weakclassifiers = cascade.get_weakclassifier_counts(i);
        for (j = 0; j < num_weakclassifiers; ++j)
            {
            int rect_num = 2;
            if (cascade.is_haar_feature_third_rect_exist(i, j))
                {
                rect_num = 3;
                }
            for (k = 0; k < rect_num; ++k)
                {
                //stage
                sprintf(textbuf, "stage %d/%d", i + 1, num_stages);
                putText(im, textbuf, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0));
                //weakclassifier
                sprintf(textbuf, "classifier %d/%d", j + 1, num_weakclassifiers);
                putText(im, textbuf, Point(detect_winsize * SCALEFACTOR - 150, detect_winsize * SCALEFACTOR - 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0));
                //rect
                if (cascade.get_haar_feature_rect_weight_sign(i, j, k))
                    {
                    rectangle(im, Rect(cascade.get_haar_feature_rect(i, j, k).x * SCALEFACTOR,
                                       cascade.get_haar_feature_rect(i, j, k).y * SCALEFACTOR,
                                       cascade.get_haar_feature_rect(i, j, k).width * SCALEFACTOR,
                                       cascade.get_haar_feature_rect(i, j, k).height * SCALEFACTOR), Scalar(255), -1);
                    }
                else
                    {
                    rectangle(im, Rect(cascade.get_haar_feature_rect(i, j, k).x * SCALEFACTOR,
                                       cascade.get_haar_feature_rect(i, j, k).y * SCALEFACTOR,
                                       cascade.get_haar_feature_rect(i, j, k).width * SCALEFACTOR,
                                       cascade.get_haar_feature_rect(i, j, k).height * SCALEFACTOR), Scalar(0), -1);
                    }
                }
            imshow("cascade layout", im);
            if (waitKey(0) == 27)
                {
                return EXIT_SUCCESS;
                }
            im.setTo(Scalar(200));
            }
        }

    return EXIT_SUCCESS;
}