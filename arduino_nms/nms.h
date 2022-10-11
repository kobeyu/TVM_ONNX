
#define SZ_OF_BBX 6
#define GET_CLASS(ptr, idx) (int)*((float*)(ptr) + (idx) * SZ_OF_BBX)
#define GET_SCORE(ptr, idx) *((float*)(ptr) + (idx) * SZ_OF_BBX + 1)
#define GET_BBX(ptr, idx) *(bbx_t*)((float*)(ptr) + (idx) * SZ_OF_BBX + 2)

#define GET_BBX_LEFT(ptr, idx) (int)*((float*)(ptr) + (idx) * SZ_OF_BBX + 2)
#define GET_BBX_TOP(ptr, idx) (int)*((float*)(ptr) + (idx) * SZ_OF_BBX + 3)
#define GET_BBX_RIGHT(ptr, idx) (int)*((float*)(ptr) + (idx) * SZ_OF_BBX + 4)
#define GET_BBX_BOTTOM(ptr, idx) (int)*((float*)(ptr) + (idx) * SZ_OF_BBX + 5)


typedef struct bbx {
    float l; float t;
    float r;
    float b;
} bbx_t;



int nms(float *ptr, int valid_count);

