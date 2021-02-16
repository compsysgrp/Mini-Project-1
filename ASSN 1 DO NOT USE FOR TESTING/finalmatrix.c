#include <stdio.h>
#include <stdlib.h>
#include <stdint-gcc.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <nmmintrin.h>
#include <tmmintrin.h>
#include <time.h>


int16_t* intmatrix(int w, int l)
{
    int16_t *mtx = calloc(w * l, sizeof(int16_t));
    for(int i=0; i<w*l; i++)
    {
        int val = rand();
        mtx[i]=val;
    }
    return mtx;
}

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

void printint(int16_t *mtx1, int r,int c)
{
    for(int i=0; i<r; i++)
    {
        for (int j = 0; j<c; j++)
        {
            printf("%d ", mtx1[i*c+j]);
        }
        printf("\r\n");
    }
    printf("\r\n");
}
void printflt(float *mtx1, int r,int c)
{
    for(int i=0; i<r; i++)
    {
        for (int j = 0; j<c; j++)
        {
            printf("%.6f ", mtx1[i*c+j]);
        }
        printf("\r\n");
    }
    printf("\r\n");
}

void simdint(int16_t *mtx1,int16_t *mtx2, int16_t *mtx3, int r1, int c1, int r2, int c2)
{
    for(int i=0; i<r1; i++)
    {
         for (int j = 0; j<c2; j++)
         {
            int k;
            for (int k = 0; k<r2; k++)
            {
                if (k == 0)
                {
                    mtx3[i*c2+j] = 0;
                }
                __m256i ans = _mm256_loadu_si256((__m256i*) &mtx3[i*c2+j]); //answer 
                __m256i nm1 = _mm256_loadu_si256((__m256i*) &mtx1[i*c1+k]); //first val
                __m256i nm2 = _mm256_loadu_si256((__m256i*) &mtx2[k*c2+j]); //second val

                __m256i mlt = _mm256_mullo_epi16 (nm1, nm2); //multiply nm1*nm2

                ans = _mm256_add_epi16(ans,mlt); //add multipled variable to existing mtx3 value

                _mm256_storeu_si256((__m256i*) (mtx3 + i*c2+j),ans); //stored new calculated value
            }
         }
    }
}

void simdflt(float* mtx1, float* mtx2, float* mtx3, int r1, int c1, int r2, int c2)
{
    for(int i=0; i<r1; i++)
    {
         for (int j = 0; j<c2; j++)
         {
            int k;
            for (int k = 0; k<r2; k++)
            {
                if (k == 0)
                {
                    mtx3[i*c2+j] = 0;
                }
                __m256 ans = _mm256_loadu_ps(mtx3 + i*c2+j); //answer 
                __m256 nm1 = _mm256_loadu_ps(mtx1 + i*c1+k); //first val
                __m256 nm2 = _mm256_loadu_ps(mtx2 + k*c2+j); //second val

                ans = _mm256_fmadd_ps(nm1,nm2,ans); //

                _mm256_storeu_ps(mtx3 + i*c2+j,ans); //stored new calculated value
            }
            printf("\r");
           
         }
        
    }
}

int16_t* multiplyint(int16_t* mtx1, int16_t* mtx2, int r1, int c1, int r2, int c2)
{
    int16_t* mtx3 = calloc(r1 * c2, sizeof(int16_t));
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

    //declaring all result matrices
    int16_t* resultint0;
    int16_t* resultint1;
    int16_t* resultint2;
    int16_t* resultint3;
    int16_t *resultint0simd = calloc(100 * 100, sizeof(int16_t));
    int16_t *resultint1simd = calloc(1000 * 1000, sizeof(int16_t));
    int16_t *resultint2simd = calloc(5000 * 5000, sizeof(int16_t));
    int16_t *resultint3simd = calloc(10000 * 10000, sizeof(int16_t));
    float* resultflt0; 
    float* resultflt1; 
    float* resultflt2; 
    float* resultflt3;
    float *resultflt0simd = calloc(100 * 100, sizeof(float));
    float *resultflt1simd = calloc(1000 * 1000, sizeof(float));
    float *resultflt2simd = calloc(5000 * 5000, sizeof(float));
    float *resultflt3simd = calloc(10000 * 10000, sizeof(float));
    clock_t start;
    float endtimer;
    
    //creating all the matrices to be multiplied
    
    int16_t* mtx00_int = intmatrix(100,100);
    int16_t* mtx01_int = intmatrix(100,100);
    
    int16_t* mtx1_int = intmatrix(1000,1000);
    int16_t* mtx2_int = intmatrix(1000,1000);
    
    int16_t* mtx3_int = intmatrix(5000,5000);
    int16_t* mtx4_int = intmatrix(5000,5000);
    
    int16_t* mtx5_int = intmatrix(10000,10000);
    int16_t* mtx6_int = intmatrix(10000,10000);
    

    
    float* mtx00_flt = fltmatrix(100,100);
    float* mtx01_flt = fltmatrix(100,100);
    float* mtx1_flt = fltmatrix(1000,1000);
    float* mtx2_flt = fltmatrix(1000,1000);
    float* mtx3_flt = fltmatrix(5000,5000);
    float* mtx4_flt = fltmatrix(5000,5000);
    float* mtx5_flt = fltmatrix(10000,10000);
    float* mtx6_flt = fltmatrix(10000,10000);
    

    //timing traditional short
    
    printf("Timing Traditional multiplication for short...\r\n");

    start = clock();
    resultint0= multiplyint(mtx00_int,mtx01_int, 100, 100, 100, 100);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("100x100 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint0);

    start = clock();
    resultint1= multiplyint(mtx1_int,mtx2_int, 1000, 1000, 1000, 1000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("1000x1000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint1);
    
    start = clock();
    resultint2= multiplyint(mtx3_int,mtx4_int, 5000, 5000, 5000, 5000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("5000x5000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint2);
    
    
    start = clock();
    resultint2= multiplyint(mtx5_int,mtx6_int, 10000, 10000, 10000, 10000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("10000x10000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint3);
    
    printf("\r\n");
    
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
    

    //timing SIMD int
    
    printf("Timing SIMD multiplication for int...\r\n");
    start = clock();
    simdint(mtx00_int,mtx01_int, resultint0simd, 100, 100, 100, 100);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("100x100 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint0simd);

    start = clock();
    simdint(mtx1_int,mtx2_int, resultint1simd, 1000, 1000, 1000, 1000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("1000x1000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint1simd);

    start = clock();
    simdint(mtx3_int,mtx4_int, resultint2simd, 5000, 5000, 5000, 5000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("5000x5000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint2simd);

    start = clock();
   simdint(mtx5_int,mtx6_int, resultint3simd, 10000, 10000, 10000, 10000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("10000x10000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);  
    free(resultint3simd);
    
    printf("\r\n");

    
    //timing SIMD float
    start = clock();
    simdflt(mtx00_flt,mtx01_flt, resultflt0simd, 100, 100, 100, 100);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("100x100 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt0simd);

    printf("Timing SIMD multiplication for float...\r\n");
    start = clock();
    simdflt(mtx1_flt,mtx2_flt, resultflt1simd, 1000, 1000, 1000, 1000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("1000x1000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt1simd);

    start = clock();
    simdflt(mtx3_flt,mtx4_flt, resultflt2simd, 5000, 5000, 5000, 5000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("5000x5000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt2simd);

    start = clock();
    simdflt(mtx5_flt,mtx6_flt, resultflt3simd, 10000, 10000, 10000, 10000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("10000x100000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt3simd);
    
    printf("Timing complete");
    


    //freeing all matrices used for multiplication
    
    free(mtx00_int);
    free(mtx01_int);
    free(mtx1_int);
    free(mtx2_int);
    free(mtx3_int);
    free(mtx4_int);   
    free(mtx5_int);
    free(mtx6_int);
    free(mtx00_flt);
    free(mtx01_flt);
    free(mtx1_flt);
    free(mtx2_flt);
    free(mtx3_flt);
    free(mtx4_flt);
    free(mtx5_flt);
    free(mtx6_flt);
}
