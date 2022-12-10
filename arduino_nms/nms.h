
#define NUMBER_OF_BBX 8
#define GET_CLASS(ptr, idx) (int8_t)*((int8_t*)(ptr) + (idx) )
#define GET_SCORE(ptr, idx) *((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 1)
//#define GET_BBX(ptr, idx) *(bbx_t*)((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 2)

#define GET_BBX_TOP(ptr, idx) (int8_t)*((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 2)
#define GET_BBX_BOTTOM(ptr, idx) (int8_t)*((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 3)
#define GET_BBX_LEFT(ptr, idx) (int8_t)*((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 4)
#define GET_BBX_RIGHT(ptr, idx) (int8_t)*((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 5)

#define GET_BBX_TOP_PTR(ptr, idx) ((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 2)
#define GET_BBX_BOTTOM_PTR(ptr, idx) ((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 3)
#define GET_BBX_LEFT_PTR(ptr, idx) ((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 4)
#define GET_BBX_RIGHT_PTR(ptr, idx) ((int8_t*)(ptr) + (idx) + NUMBER_OF_BBX * 5)

int nms(int8_t *ptr, int valid_count);
