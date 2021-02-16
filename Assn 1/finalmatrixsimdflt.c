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



    float *resultflt4simd = calloc(4000 * 4000, sizeof(float));
    float *resultflt5simd = calloc(5000 * 5000, sizeof(float));


    float* mtx9_flt = fltmatrix(4000,4000);
    float* mtx10_flt = fltmatrix(4000,4000);
    float* mtx11_flt = fltmatrix(5000,5000);
    float* mtx12_flt = fltmatrix(5000,5000);

    clock_t start = clock();
    float endtimer;
    //timing SIMD int
    
    printf("Timing SIMD multiplication for flt...\r\n");

    start = clock();
    simdflt(mtx9_flt,mtx10_flt, resultflt4simd, 4000, 4000, 4000, 4000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("4000x4000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);  

    start = clock();
    simdflt(mtx11_flt,mtx12_flt, resultflt5simd, 5000, 5000, 5000, 5000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("5000x5000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);  
    
    printf("\r\n");

}
