/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
//#include "util_tflite.h"
#include "tflite_blazeface.h"
#include "util_debug.h"
#include <list>
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"


//TFlite C API
static const TfLiteTensor*      tensor_input;
static const TfLiteTensor*      tensor_scores;
static const TfLiteTensor*      tensor_bboxes;

TfLiteInterpreter* interpreter_;
TfLiteModel* model_ = nullptr;
TfLiteInterpreterOptions* options_ = nullptr;
TfLiteDelegate* delegate_ = nullptr;

static std::list<fvec2> s_anchors;

/*
 * determine where the anchor points are scatterd.
 *   https://github.com/tensorflow/tfjs-models/blob/master/blazeface/src/face.ts
 */
static int
create_blazeface_anchors(int input_w, int input_h)
{
    /* ANCHORS_CONFIG  */
    int strides[2] = {8, 16};
    int anchors[2] = {2,  6};

    int numtotal = 0;

    for (int i = 0; i < 2; i ++)
    {
        int stride = strides[i];
        int gridCols = (input_w + stride -1) / stride;
        int gridRows = (input_h + stride -1) / stride;
        int anchorNum = anchors[i];

        fvec2 anchor;
        for (int gridY = 0; gridY < gridRows; gridY ++)
        {
            anchor.y = stride * (gridY + 0.5f);
            for (int gridX = 0; gridX < gridCols; gridX ++)
            {
                anchor.x = stride * (gridX + 0.5f);
                for (int n = 0; n < anchorNum; n ++)
                {
                    s_anchors.push_back (anchor);
                    numtotal ++;
                }
            }
        }
    }
    return numtotal;
}



/* -------------------------------------------------- *
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_blazeface(const char *model_buf, size_t model_size, blazeface_config_t *config)
{

    model_ = TfLiteModelCreate(model_buf,model_size);

    options_ = TfLiteInterpreterOptionsCreate();
    delegate_ = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
    TfLiteInterpreterOptionsAddDelegate(options_, delegate_);
    interpreter_ = TfLiteInterpreterCreate(model_, options_);

    TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter_);
    if (status != kTfLiteOk) {
        LOGE("Something went wrong when allocating tensors");
        return 0;
    }

    tensor_input = TfLiteInterpreterGetInputTensor(interpreter_, 0);
    tensor_bboxes = TfLiteInterpreterGetOutputTensor(interpreter_, 0);
    tensor_scores = TfLiteInterpreterGetOutputTensor(interpreter_, 1);


    int det_input_w = tensor_input->dims->data[2];
    int det_input_h = tensor_input->dims->data[1];
    create_blazeface_anchors (det_input_w, det_input_h);

    config->score_thresh = 0.75f;
    config->iou_thresh   = 0.3f;


    return 0;
}

void *
get_blazeface_input_buf (int *w, int *h)
{
    *w = tensor_input->dims->data[2];
    *h = tensor_input->dims->data[1];

    return tensor_input->data.data;
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite (Face detection)
 * -------------------------------------------------- */
static float *
get_bbox_ptr (int anchor_idx)
{
    int idx = 16 * anchor_idx;
    float *bboxes_ptr = (float *)tensor_bboxes->data.data;

    return &bboxes_ptr[idx];
}

static int
decode_bounds (std::list<face_t> &face_list, float score_thresh, int input_img_w, int input_img_h)
{
    face_t face_item;
    float  *scores_ptr = (float *)tensor_scores->data.data;

    int i = 0;
    for (auto itr = s_anchors.begin(); itr != s_anchors.end(); i ++, itr ++)
    {
        fvec2 anchor = *itr;
        float score0 = scores_ptr[i];
        float score = 1.0f / (1.0f + exp(-score0));

        if (score > score_thresh)
        {
            float *p = get_bbox_ptr (i);

            /* boundary box */
            float sx = p[0];
            float sy = p[1];
            float w  = p[2];
            float h  = p[3];

            float cx = sx + anchor.x;
            float cy = sy + anchor.y;

            cx /= (float)input_img_w;
            cy /= (float)input_img_h;
            w  /= (float)input_img_w;
            h  /= (float)input_img_h;

            fvec2 topleft, btmright;
            topleft.x  = cx - w * 0.5f;
            topleft.y  = cy - h * 0.5f;
            btmright.x = cx + w * 0.5f;
            btmright.y = cy + h * 0.5f;

            face_item.score    = score;
            face_item.topleft  = topleft;
            face_item.btmright = btmright;

            /* landmark positions (6 keys) */
            for (int j = 0; j < kFaceKeyNum; j ++)
            {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx += anchor.x;
                ly += anchor.y;
                lx /= (float)input_img_w;
                ly /= (float)input_img_h;

                face_item.keys[j].x = lx;
                face_item.keys[j].y = ly;
            }

            face_list.push_back (face_item);
        }
    }
    return 0;
}

/* -------------------------------------------------- *
 *  Apply NonMaxSuppression:
 *      https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/image_ops.ts
 * -------------------------------------------------- */
static float
calc_intersection_over_union (face_t &face0, face_t &face1)
{
    float sx0 = face0.topleft.x;
    float sy0 = face0.topleft.y;
    float ex0 = face0.btmright.x;
    float ey0 = face0.btmright.y;
    float sx1 = face1.topleft.x;
    float sy1 = face1.topleft.y;
    float ex1 = face1.btmright.x;
    float ey1 = face1.btmright.y;
    
    float xmin0 = std::min (sx0, ex0);
    float ymin0 = std::min (sy0, ey0);
    float xmax0 = std::max (sx0, ex0);
    float ymax0 = std::max (sy0, ey0);
    float xmin1 = std::min (sx1, ex1);
    float ymin1 = std::min (sy1, ey1);
    float xmax1 = std::max (sx1, ex1);
    float ymax1 = std::max (sy1, ey1);
    
    float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
    float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    if (area0 <= 0 || area1 <= 0)
        return 0.0f;

    float intersect_xmin = std::max (xmin0, xmin1);
    float intersect_ymin = std::max (ymin0, ymin1);
    float intersect_xmax = std::min (xmax0, xmax1);
    float intersect_ymax = std::min (ymax0, ymax1);

    float intersect_area = std::max (intersect_ymax - intersect_ymin, 0.0f) *
                           std::max (intersect_xmax - intersect_xmin, 0.0f);
    
    return intersect_area / (area0 + area1 - intersect_area);
}

static bool
compare (face_t &v1, face_t &v2)
{
    if (v1.score > v2.score)
        return true;
    else
        return false;
}

static int
non_max_suppression (std::list<face_t> &face_list, std::list<face_t> &face_sel_list, float iou_thresh)
{
    face_list.sort (compare);

    for (auto itr = face_list.begin(); itr != face_list.end(); itr ++)
    {
        face_t face_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_sel = face_sel_list.rbegin(); itr_sel != face_sel_list.rend(); itr_sel ++)
        {
            face_t face_sel = *itr_sel;

            float iou = calc_intersection_over_union (face_candidate, face_sel);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            face_sel_list.push_back(face_candidate);
            if (face_sel_list.size() >= MAX_FACE_NUM)
                break;
        }
    }

    return 0;
}

static void
pack_face_result (blazeface_result_t *face_result, std::list<face_t> &face_list)
{
    int num_faces = 0;
    for (auto itr = face_list.begin(); itr != face_list.end(); itr ++)
    {
        face_t face = *itr;
        memcpy (&face_result->faces[num_faces], &face, sizeof (face));
        num_faces ++;
        face_result->num = num_faces;

        if (num_faces >= MAX_FACE_NUM)
            break;
    }
}


/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
int
invoke_blazeface (blazeface_result_t *face_result, blazeface_config_t *config)
{

    TfLiteStatus status;

    status = TfLiteInterpreterInvoke(interpreter_);
    if (status != kTfLiteOk) {
        LOGE("Something went wrong when running the TFLite model");
        return 0;
    }


//
//    if (s_detect_interpreter.interpreter->Invoke() != kTfLiteOk)
//    {
//        DBG_LOGE ("ERR: %s(%d)\n", __FILE__, __LINE__);
//        return -1;
//    }

    /* decode boundary box and landmark keypoints */
    float score_thresh = config->score_thresh;
    std::list<face_t> face_list;

    int input_img_w = tensor_input->dims->data[2];
    int input_img_h = tensor_input->dims->data[1];
    decode_bounds (face_list, score_thresh, input_img_w, input_img_h);


#if 1 /* USE NMS */
    float iou_thresh = config->iou_thresh;
    std::list<face_t> face_nms_list;

    non_max_suppression (face_list, face_nms_list, iou_thresh);
    pack_face_result (face_result, face_nms_list);
#else
    pack_face_result (face_result, face_list);
#endif

    return 0;
}

