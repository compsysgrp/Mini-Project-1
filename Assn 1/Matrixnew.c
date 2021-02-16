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

int16_t* simdint(int16_t* mtx1, int16_t* mtx2, int r1, int c1, int r2, int c2)
{
    int16_t* mtx3 = calloc(r1 * c2, sizeof(int16_t));
    int width = 256/16; //2 byte int/16 b/c of the data type of _mm256_set1_epi16 etc. If we were using __m128i and epi8, we would do 128/8
    int val = 0;
    
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            int k;
            int k_max = (r2 >> 4) << 4;
            for (int k = 0; k < k_max; k+=width)
            {             
                __m256i ans = _mm256_loadu_si256((__m256i*) (mtx3 + i*c2+j)); //answer 
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
    return mtx3;
}

int main()
{
  
}