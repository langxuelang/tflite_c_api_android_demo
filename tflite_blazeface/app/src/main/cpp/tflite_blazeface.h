/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_BLAZEFACE_H_
#define TFLITE_BLAZEFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

/* https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite */
#define BLAZEFACE_MODEL_PATH  "blazeface_model/face_detection_front.tflite"

#define MAX_FACE_NUM  100

enum face_key_id {
    kRightEye = 0,  //  0
    kLeftEye,       //  1
    kNose,          //  2
    kMouth,         //  3
    kRightEar,      //  4
    kLeftEar,       //  5

    kFaceKeyNum
};

typedef struct fvec2
{
    float x, y;
} fvec2;

typedef struct _face_t
{
    float score;
    fvec2 topleft;
    fvec2 btmright;
    fvec2 keys[kFaceKeyNum];
} face_t;

typedef struct _blazeface_result_t
{
    int num;
    face_t faces[MAX_FACE_NUM];
} blazeface_result_t;

typedef struct _blazeface_config_t
{
    float score_thresh;
    float iou_thresh;
} blazeface_config_t;

int init_tflite_blazeface (const char *model_buf, unsigned long model_size, blazeface_config_t *config);
void  *get_blazeface_input_buf (int *w, int *h);

int invoke_blazeface (blazeface_result_t *blazeface_result, blazeface_config_t *config);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_BLAZEFACE_H_ */
