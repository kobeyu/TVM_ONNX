#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <nds_intrinsic.h>
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

int packed_data(int8_t data){
  int packed = (int)data;
  int tmp = __nds__swap8(packed);
  packed = __nds__add8(packed,tmp);
  packed = __nds__pkbb16(packed,packed);
  return tmp;
 }


int nms(int8_t *ptr, int valid_count){
    uint8_t score = 0;
    uint8_t score_threshold = 64;
    float iou = 0;
    float iou_threshold = 0.45;

    for (int i =0; i <valid_count; i++){
        score = GET_SCORE(ptr, i);
        if (score < score_threshold) {
            GET_SCORE(ptr, i) = -1;
            continue;
        }
        int a_t = packed_data(GET_BBX_TOP(ptr,i));
        int a_b = packed_data(GET_BBX_BOTTOM(ptr,i));
        int a_l = packed_data(GET_BBX_LEFT(ptr,i));
        int a_r = packed_data(GET_BBX_RIGHT(ptr,i));
        int a_u = (GET_BBX_TOP(ptr,i) - GET_BBX_BOTTOM(ptr,i)) * (GET_BBX_LEFT(ptr,i) - GET_BBX_RIGHT(ptr,i));

        for (int j =i; j <8; j+=4){
            int j_t = *(int*)(GET_BBX_TOP_PTR(ptr,i));
            int j_b = *(int*)(GET_BBX_BOTTOM_PTR(ptr,i));
            int j_l = *(int*)(GET_BBX_LEFT_PTR(ptr,i));
            int j_r = *(int*)(GET_BBX_RIGHT_PTR(ptr,i));
            uint32_t res_t = __nds__umax8(a_t, j_t);
            uint32_t res_b = __nds__umin8(a_b, j_b);
            uint32_t res_l = __nds__umax8(a_l, j_l);
            uint32_t res_r = __nds__umin8(a_r, j_r);
            uint32_t w = __nds__sub8(res_r, res_l);
            uint32_t h = __nds__sub8(res_b, res_t);
            w = __nds__umax8(w,0);
            uint8_t *w_p = (uint8_t*)&w;
            uint8_t w_0 = *(w_p);
            uint8_t w_1 = *(w_p+1);
            uint8_t w_2 = *(w_p+2);
            uint8_t w_3 = *(w_p+3);
            h = __nds__umax8(h,0);
            uint8_t *h_p = (uint8_t*)&h;
            uint8_t h_0 = *(h_p);
            uint8_t h_1 = *(h_p+1);
            uint8_t h_2 = *(h_p+2);
            uint8_t h_3 = *(h_p+3);

            int area_0 = h_0 * w_0;
            int area_1 = h_1 * w_1;
            int area_2 = h_2 * w_2;
            int area_3 = h_3 * w_3;

            int u_0 = a_u + (GET_BBX_RIGHT(ptr,j) - GET_BBX_LEFT(ptr,j)) * (GET_BBX_BOTTOM(ptr,j) - GET_BBX_TOP(ptr,j)) - area_0;
            int u_1 = a_u + (GET_BBX_RIGHT(ptr,j+1) - GET_BBX_LEFT(ptr,j+1)) * (GET_BBX_BOTTOM(ptr,j+1) - GET_BBX_TOP(ptr,j+1)) - area_1;
            int u_2 = a_u + (GET_BBX_RIGHT(ptr,j+2) - GET_BBX_LEFT(ptr,j+2)) * (GET_BBX_BOTTOM(ptr,j+2) - GET_BBX_TOP(ptr,j+2)) - area_2;
            int u_3 = a_u + (GET_BBX_RIGHT(ptr,j+3) - GET_BBX_LEFT(ptr,j+3)) * (GET_BBX_BOTTOM(ptr,j+3) - GET_BBX_TOP(ptr,j+3)) - area_3;
            float iou_threshold = 0.45;
            if ( ((float)area_0/(float)u_0) >= iou_threshold) GET_SCORE(ptr,j) = -1;
            if ( ((float)area_1/(float)u_1) >= iou_threshold) GET_SCORE(ptr,j+1) = -1;
            if ( ((float)area_2/(float)u_2) >= iou_threshold) GET_SCORE(ptr,j+2) = -1;
            if ( ((float)area_3/(float)u_3) >= iou_threshold) GET_SCORE(ptr,j+3) = -1;
        } 
    }
    return 13;

}