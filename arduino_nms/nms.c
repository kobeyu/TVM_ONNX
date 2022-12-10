#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "nms.h"

float cal_iou(int8_t *ptr, int i, int j)
{ // i = a j = b
    int8_t a_t, a_b, a_l, a_r;
    int8_t b_t, b_b, b_l, b_r;
    a_t = GET_BBX_TOP(ptr,i);
    a_b = GET_BBX_BOTTOM(ptr,i);
    a_l = GET_BBX_LEFT(ptr,i);
    a_r = GET_BBX_RIGHT(ptr,i);

    b_t = GET_BBX_TOP(ptr,j);
    b_b = GET_BBX_BOTTOM(ptr,j);
    b_l = GET_BBX_LEFT(ptr,j);
    b_r = GET_BBX_RIGHT(ptr,j);
    
    //# Overlapping width and height
//    printf("%d %d %d %d", bbx_a->t, bbx_a->b , bbx_a->l, bbx_a->r);
    int8_t r = (a_r < b_r) ? a_r : b_r;
    int8_t l = (a_l > b_l) ? a_l : b_l;
    int8_t b = (a_b < b_b) ? a_b : b_b;
    int8_t t = (a_t > b_t) ? a_t : b_t;
    int8_t w = r - l;
    int8_t h = b - t;
    w = (0 > w)? 0 : w;
    h = (0 > h)? 0 : h;

    // Overlapping area
    int area = h * w;


    // total area of the figure formed by box a and box b
    // except for overlapping area
    int u = (a_r - a_l) * (a_b - a_t) + \
            (b_r - b_l) * (b_b - b_t) - area;

    return (u <= 0) ? 0.0f : (float)area/(float)u;
}


int nms(int8_t *ptr, int valid_count)
{
    uint8_t score = 0;
    float iou = 0;
    float iou_threshold = 0.45;
    uint8_t score_threshold = 64;
//    bbx_t a = {0}, b = {0};

    for (int i = 0; i < valid_count; i++) {
        score = GET_SCORE(ptr, i);
        if (score < score_threshold) {
            GET_SCORE(ptr, i) = -1;
            continue;
        }

//        a = GET_BBX(ptr, i);
        for (int j = i; j < valid_count; j++) {
            if (GET_SCORE(ptr, j) <= 0 || i == j ) continue;
//            b = GET_BBX(ptr, j);
            
            iou = cal_iou(ptr, i, j);
            if (iou >= iou_threshold) GET_SCORE(ptr, j) = -1;
        }
    }
    return 13;
}