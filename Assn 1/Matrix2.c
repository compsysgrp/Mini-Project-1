#include <stdio.h>
#include <stdlib.h>
#include <stdint-gcc.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>


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
    //int val = 0;

    for (int i = 0; i < r1; i++) // r1 = 2
    {
        for (int j = 0; j < c2; j++) // c1 = 3
        {
            for (int k = 0; k < r2; k++) //c2 = 2
            {
                mtx3[i*c2+j] += mtx1[i*c1+k] * mtx2[k*c2+j];
                /*
                if (val == 0)
                {
                    val = mtx1[i*c1+k] * mtx2[k*c2+j];
                    printf("%d: %d * %d\r\n",mtx3[i*c2+j], mtx1[i*c1+k],mtx2[k*c2+j]);
                }
                else
                {   
                    printf("%d: %d + (%d * %d)\r\n",mtx3[i*c2+j], val, mtx1[i*c1+k],mtx2[k*c2+j]);
                }
                val = mtx3[i*c2+j], mtx1[i*c1+k],mtx2[k*c2+j];
                */
            }
            //val = 0;
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

int16_t* simdint(int16_t* mtx1, int16_t* mtx2, int r1, int c1, int r2, int c2)
{
    int16_t* mtx3 = calloc(r1 * c2, sizeof(int16_t));
    int width = 256/16; //256/16 b/c of the data type of _mm256_set1_epi16 etc. If we were using __m128i and epi8, we would do 128/8
    
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
        }
        for (;k < r2; k++)
            {
                mtx3[i*c2+k]+=mtx1[ic1+j]*mtx2[j*c2+j];
            }
    }
    return mtx3;
}

int16_t* simdflt(float* mtx1, float* mtx2, int r1, int c1, int r2, int c2)
    {
    int width = 256/32;//256/32 b/c of the data type of float and 256 bit SIMD
    for (int i = 0; i < r1; i++)
    {
        for (int j = 0; j < c2; j++)
        {
            int k_max = (r2 >> 4) << 4;
            for (int k = 0; k < k_max; k+=width)
            {
                        
            }
        }
    }
    return mtx3;        
    }

int main()
{
    //test matrix 1: 3x1
    //1 3 4
    //2 5 1
    //6 0 7

    int16_t *mtx1 = calloc(3 * 1, sizeof(int16_t));

    mtx1[0]=1;
    mtx1[1]=3;
    mtx1[2]=4;
    /*
    mtx1[3]=2;
    mtx1[4]=5;
    mtx1[5]=1;
    mtx1[6]=6;
    mtx1[7]=0;
    mtx1[8]=7;
    */

    intprintmatrix(mtx1,3,1);
    
    //test matrix 2: 1x4
    // 3 8 1 9

    int16_t *mtx2 = calloc(1 * 4, sizeof(int16_t));

    mtx2[0]=3;
    mtx2[1]=8;
    mtx2[2]=1;
    mtx2[3]=9;

    intprintmatrix(mtx2,1,4);

    /*
    int16_t* value1 = intmatrix(5,5);
    intprintmatrix(value1, 5, 5);

    float* value2 = fltmatrix(5,5);
    fltprintmatrix(value2,5,5);
    */

    //int16_t* results = multiplyint(mtx1,mtx2,;

    //answer3,1,1,4)
    //3 8 1 9
    //9 24 3 27
    //12 32 4 36
    int16_t* results = simdint(mtx1,mtx2,3,1,1,4);

    intprintmatrix(results,3,4);

    free(results);
}