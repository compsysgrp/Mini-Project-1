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
    float* resultflt4;
    float* resultflt5;
    clock_t start;
    float endtimer;
    
    float* mtx00_flt = fltmatrix(100,100);
    float* mtx01_flt = fltmatrix(100,100);
    float* mtx10_flt = fltmatrix(1000,1000);
    float* mtx11_flt = fltmatrix(1000,1000);
    float* mtx20_flt = fltmatrix(2000,2000);
    float* mtx21_flt = fltmatrix(2000,2000);
    float* mtx30_flt = fltmatrix(3000,3000);
    float* mtx31_flt = fltmatrix(3000,3000);
    float* mtx40_flt = fltmatrix(4000,4000);
    float* mtx41_flt = fltmatrix(4000,4000);
    float* mtx50_flt = fltmatrix(5000,5000);
    float* mtx51_flt = fltmatrix(5000,5000);
    

    //timing traditional float
    printf("Timing Traditional multiplication for float...\r\n");

    start = clock();
    resultflt0= multiplyflt(mtx00_flt,mtx01_flt, 100, 100, 100, 100);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("100x100 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt0);

    start = clock();
    resultflt1= multiplyflt(mtx10_flt,mtx11_flt, 1000, 1000, 1000, 1000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("1000x1000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt1);

    start = clock();
    resultflt2= multiplyflt(mtx20_flt,mtx21_flt, 2000, 2000, 2000, 2000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("2000x2000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt2);

    start = clock();
    resultflt3= multiplyflt(mtx30_flt,mtx31_flt, 3000, 3000, 3000, 3000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("3000x3000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt3);

    start = clock();
    resultflt4= multiplyflt(mtx40_flt,mtx41_flt, 4000, 4000, 4000, 4000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("4000x4000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt4);

    start = clock();
    resultflt5= multiplyflt(mtx50_flt,mtx51_flt, 5000, 5000, 5000, 5000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("5000x5000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt5);

    printf("\r\n");
    printf("Timing complete");
    


    //freeing all matrices used for multiplication

    free(mtx00_flt);
    free(mtx01_flt);
    free(mtx10_flt);
    free(mtx11_flt);
    free(mtx20_flt);
    free(mtx21_flt);
    free(mtx30_flt);
    free(mtx31_flt);
    free(mtx40_flt);
    free(mtx41_flt);
    free(mtx50_flt);
    free(mtx51_flt);
}
