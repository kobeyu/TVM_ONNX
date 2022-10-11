#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "nms.h"

float cal_iou(bbx_t *bbx_a, bbx_t *bbx_b)
{
    //# Overlapping width and height
    int r = (bbx_a->r < bbx_b->r) ? bbx_a->r : bbx_b->r;
    int l = (bbx_a->l > bbx_b->l) ? bbx_a->l : bbx_b->l;
    int b = (bbx_a->b < bbx_b->b) ? bbx_a->b : bbx_b->b;
    int t = (bbx_a->t > bbx_b->t) ? bbx_a->t : bbx_b->t;
    int w = r - l;
    int h = b - t;
    w = (0 > w)? 0 : w;
    h = (0 > h)? 0 : h;

    // Overlapping area
    int area = h * w;


    // total area of the figure formed by box a and box b
    // except for overlapping area
    int u = (bbx_a->r - bbx_a->l) * (bbx_a->b - bbx_a->t) + \
            (bbx_b->r - bbx_b->l) * (bbx_b->b - bbx_b->t) - area;

    return (u <= 0) ? 0.0f : (float)area/(float)u;
}


int nms(float *ptr, int valid_count)
{
    float score = 0;
    float iou = 0;
    float iou_threshold = 0.45;
    float score_threshold = 0.7;
    bbx_t a = {0}, b = {0};

    for (int i = 0; i < valid_count; i++) {
        score = GET_SCORE(ptr, i);
        if (score < score_threshold) {
            GET_SCORE(ptr, i) = -1;
            continue;
        }

        a = GET_BBX(ptr, i);
        for (int j = i; j < valid_count; j++) {
            if (GET_SCORE(ptr, j) <= 0 || i == j ) continue;
            b = GET_BBX(ptr, j);
            iou = cal_iou(&a, &b);
            if (iou >= iou_threshold) GET_SCORE(ptr, j) = -1;
        }
    }
    return 13;
}
