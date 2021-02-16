#include <stdio.h>
#include <stdlib.h>
#include <stdint-gcc.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <nmmintrin.h>
#include <tmmintrin.h>
#include <time.h>

float* fltmatrix(int w, int l)
{
    float *mtx = calloc(w * l, sizeof(float));
    for (int i=0; i<w*l; i++)
    {
        float val = (float)rand()/(float)(RAND_MAX);
        mtx[i]=val;
    }
    return mtx;
}

float* multiplyflt(float* mtx1, float* mtx2, int r1, int c1, int r2, int c2)
{
    float* mtx3 = calloc(r1 * c2, sizeof(float));
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            for (int k = 0; k < r2; k++)
            {
                mtx3[i*c2+j] += mtx1[i*c1+k] * mtx2[k*c2+j];
            }
        }
    }
    return mtx3;
}

int main()
{
    int msec = 0, trigger = 10;
    printf("This version of the code is for comparing SIMD multiplication with traditional mutliplication with no user input for data collection purposes\r\n");

    float* resultflt0; 
    float* resultflt1; 
    float* resultflt2; 
    float* resultflt3;
    clock_t start;
    float endtimer;
    
    float* mtx00_flt = fltmatrix(100,100);
    float* mtx01_flt = fltmatrix(100,100);
    float* mtx1_flt = fltmatrix(1000,1000);
    float* mtx2_flt = fltmatrix(1000,1000);
    float* mtx3_flt = fltmatrix(5000,5000);
    float* mtx4_flt = fltmatrix(5000,5000);
    float* mtx5_flt = fltmatrix(10000,10000);
    float* mtx6_flt = fltmatrix(10000,10000);
    

    //timing traditional float
    printf("Timing Traditional multiplication for float...\r\n");

    start = clock();
    resultflt0= multiplyflt(mtx00_flt,mtx01_flt, 100, 100, 100, 100);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("100x100 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt0);

    start = clock();
    resultflt1= multiplyflt(mtx1_flt,mtx2_flt, 1000, 1000, 1000, 1000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("1000x1000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt1);

    start = clock();
    resultflt2= multiplyflt(mtx3_flt,mtx4_flt, 5000, 5000, 5000, 5000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("5000x5000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt2);

    start = clock();
    resultflt3= multiplyflt(mtx5_flt,mtx6_flt, 10000, 10000, 10000, 10000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("10000x10000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt3);

     printf("\r\n");
    printf("Timing complete");
    


    //freeing all matrices used for multiplication

    free(mtx00_flt);
    free(mtx01_flt);
    free(mtx1_flt);
    free(mtx2_flt);
    free(mtx3_flt);
    free(mtx4_flt);
    free(mtx5_flt);
    free(mtx6_flt);
}
