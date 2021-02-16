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

void intprintmatrix(int16_t* mtx, int w, int l)
{
    for(int i=0; i<w; i++)
    {
        for(int j=0; j<l; j++)
        {
            printf("%d ", mtx[i*l+j]);
        }
        printf("\r\n");
    }
    printf("\r\n");
}

void fltprintmatrix(float* mtx, int w, int l)
{
    for(int i=0; i<w; i++)
    {
        for(int j=0; j<l; j++)
        {
            printf("%f ", mtx[i*l+j]);
        }
        printf("\r\n");
    }
    printf("\r\n");
}


int16_t* multiplyint(int16_t* mtx1, int16_t* mtx2, int r1, int c1, int r2, int c2)
{
    int16_t* mtx3 = calloc(r1 * c2, sizeof(int16_t));

    for (int i = 0; i < r1; i++) // r1 = 2
    {
        for (int j = 0; j < c2; j++) // c1 = 3
        {
            for (int k = 0; k < r2; k++) //c2 = 2
            {
                mtx3[i*c2+j] += mtx1[i*c1+k] * mtx2[k*c2+j];                
            }
            //printf("\r\n");
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

void simdint(int16_t* mtx1, int16_t* mtx2,int16_t* mtx3, int r1, int c1, int r2, int c2)
{
    int width = 256/16; //2 byte int/16 b/c of the data type of _mm256_set1_epi16 etc. If we were using __m128i and epi8, we would do 128/8
    int val = 0;
    
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            int k;
            int k_max = (r2 >> 4) << 4;
            for (int k = 0; k < r2; k++)
            {             
                __m256i ans = _mm256_loadu_si256((__m256i*) (mtx3 + i*c2+k)); //answer 
                __m256i nm1 = _mm256_set1_epi16(mtx1[i*c1+k]); //first val
                __m256i nm2 = _mm256_loadu_si256((__m256i*) (mtx2 + k*c2+j)); //second val

                __m256i mlt = _mm256_mullo_epi16 (nm1, nm2); //multiply nm1*nm2
                ans = _mm256_add_epi16(ans,mlt); //add existing value to the multiplication result

                _mm256_storeu_si256((__m256i*) mtx3 + i*c2+j,ans);
            }
            for (; k < r2; k++)
            {
                mtx3[i*c2+j] += mtx1[i*c1+k] * mtx2[k*c2+j];
            }

        }
    }
    //return mtx3;
}

void simdflt(float* mtx1, float* mtx2,float* mtx3, int r1, int c1, int r2, int c2)
{
    int width = 256/16; //2 byte int/16 b/c of the data type of _mm256_set1_epi16 etc. If we were using __m128i and epi8, we would do 128/8
    int val = 0;
    
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            int k;
            int k_max = (r2 >> 4) << 4;
            for (int k = 0; k < r2; k++)
            {             
                __m256i ans = _mm256_loadu_si256((__m256i*) (mtx3 + i*c2+k)); //answer 
                __m256i nm1 = _mm256_set1_epi16(mtx1[i*c1+k]); //first val
                __m256i nm2 = _mm256_loadu_si256((__m256i*) (mtx2 + k*c2+j)); //second val

                __m256i mlt = _mm256_mullo_epi16 (nm1, nm2); //multiply nm1*nm2
                ans = _mm256_add_epi16(ans,mlt); //add existing value to the multiplication result

                _mm256_storeu_si256((__m256i*) mtx3 + i*c2+j,ans);
            }
            for (; k < r2; k++)
            {
                mtx3[i*c2+j] += mtx1[i*c1+k] * mtx2[k*c2+j];
            }

        }
    }
    //return mtx3;
}


int main()
{
    int r1;
    int c1;
    int r2;
    int c2;
    int datatype;
    int multiplytype;
    int print;
    printf("\r\nEnter # of rows for matrix 1: \r\n");
    scanf("%d", &r1);
    printf("\r\nEnter # of columns for matrix 1: \r\n");
    scanf("%d", &c1);
    printf("\r\nEnter # of rows for matrix 2: \r\n");
    scanf("%d", &r2);
    printf("\r\nEnter # of columns for matrix 2: \r\n");
    scanf("%d", &c2);
    printf("\r\nenter 0 for short, 1 for float: \r\n");
    scanf("%d", &datatype);
    printf("\r\nenter 0 for traditional multiplication, 1 for SIMD multiplication\r\n");
    scanf("%d", &multiplytype);
    printf("\r\nenter 0 for no printing, 1 for printing: \r\n");
    scanf("%d", &print);

    int msec = 0, trigger = 10;
    int16_t* resultint;
    int16_t* mtx1_int;
    int16_t* mtx2_int;
    float* resultflt;
    float* mtx1_flt;
    float* mtx2_flt;


    if (multiplytype == 0)
    {
        printf("Starting timer for traditional multiplication...\r\n");
        clock_t start = clock();
        if (datatype == 0)
        {
            mtx1_int = intmatrix(c1, r1);
            mtx2_int = intmatrix(c1, r1);
            resultint = multiplyint(mtx1_int, mtx2_int, r1, c1, r2, c2);
        }
        else if (datatype == 1)
        {
            mtx1_flt = fltmatrix(c1, r1);
            mtx2_flt = fltmatrix(c1, r1);
            resultflt = multiplyflt(mtx1_flt, mtx2_flt, r1, c1, r2, c2);
        }
        
        float endtimer = clock() - start;
        msec= endtimer *1000 / CLOCKS_PER_SEC;
        printf("Traditional multiplication took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);        
    }
    else if (multiplytype == 1)
    {
        printf("Starting timer for SIMD multiplication...\r\n");
        clock_t start = clock();
        if (datatype == 0)
        {
            int16_t* resultint = calloc(r1 * c2, sizeof(int16_t));
            mtx1_int = intmatrix(c1, r1);
            mtx2_int = intmatrix(c1, r1);
            simdint(mtx1_int, mtx2_int, resultint, r1, c1, r2, c2);
        }
        else if (datatype == 1)
        {
            float* resultflt = calloc(r1 * c2, sizeof(int16_t));
            mtx1_flt = fltmatrix(c1, r1);
            mtx2_flt = fltmatrix(c1, r1);
            //simdflt(mtx1_flt, mtx2_flt, resultflt r1, c1, r2, c2);
        }

        float endtimer = clock() - start;
        msec= endtimer *1000 / CLOCKS_PER_SEC;
        printf("SIMD multiplication took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);    
    }

    if (print == 1)
    {
        printf("Printing... \r\n");
        if (datatype == 0)
        {
            printf("Printing Matrix 1: \r\n");
            intprintmatrix(mtx1_int, r1, c1);
            printf("\r\n Printing Matrix 2: \r\n");
            intprintmatrix(mtx1_int, r2, c2);
            printf("\r\n Printing Result Matrix: \r\n");
            intprintmatrix(resultint, r1, c2);
            free(resultint);
        }
        else if (datatype == 1)
        {
            printf("Printing Matrix 1: \r\n");
            fltprintmatrix(mtx1_flt, r1, c1);
            printf("\r\n Printing Matrix 2: \r\n");
            fltprintmatrix(mtx1_flt, r2, c2);
            printf("\r\n Printing Result Matrix: \r\n");
            fltprintmatrix(resultflt, r1, c2);
            free(resultflt);
        }
    }
}
