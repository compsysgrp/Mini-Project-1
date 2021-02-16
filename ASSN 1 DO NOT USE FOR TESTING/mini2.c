#include <stdio.h>
#include <stdlib.h>
#include <stdint-gcc.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <nmmintrin.h>
#include <tmmintrin.h>

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
            printf("%d ", mtx1[i*c+j]);
        }
        printf("\r\n");
    }
    printf("\r\n");
}

void multiplyint(int16_t *mtx1,int16_t *mtx2, int16_t *mtx3, int r1, int c1, int r2, int c2)
{
    //int val = 0;
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
                
                /*
                if (val == 0)
                {
                    val = mtx1[i*c1+k] * mtx2[k*c2+j];
                    printf("%d: %d * %d\r\n",mtx3[i*c2+j], mtx1[i*c1+k],mtx2[k*c2+j]);
                }
                else
                {   
                    printf("%d: %d + (%d * %d)\r\n",mtx3[i*c2+j], val, mtx1[i*c1+k],mtx2[k*c2+j]);
                    val = mtx3[i*c2+j] + mtx1[i*c1+k]*mtx2[k*c2+j];
                }
                */
            }
            //val = 0;
            printf("\r");
           
         }
        
    }
}

void multiplyflt(float* mtx1, float* mtx2, float* mtx3, int r1, int c1, int r2, int c2)
{
    //int val = 0;
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
                
                /*
                if (val == 0)
                {
                    val = mtx1[i*c1+k] * mtx2[k*c2+j];
                    printf("%d: %d * %d\r\n",mtx3[i*c2+j], mtx1[i*c1+k],mtx2[k*c2+j]);
                }
                else
                {   
                    printf("%d: %d + (%d * %d)\r\n",mtx3[i*c2+j], val, mtx1[i*c1+k],mtx2[k*c2+j]);
                    val = mtx3[i*c2+j] + mtx1[i*c1+k]*mtx2[k*c2+j];
                }
                */
            }
            //val = 0;
            printf("\r");
           
         }
        
    }
}
/*
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
 
            }
        }
    }
    return mtx3;
}
*/

int main ()
{
    int rows = 5;
    int cols = 5;

    //int16_t mtx1[rows*cols];
    //int16_t *mtx1 = calloc(5 * 5, sizeof(float));
    float *mtx1 = calloc(5 * 5, sizeof(float));

    //int16_t mtx2[rows*cols];
    //int16_t *mtx2 = calloc(5 * 5, sizeof(float));
    float *mtx2 = calloc(5 * 5, sizeof(float));

    //test matrix 1: 5 x 5
    //1 3 4 2 5
    //1 6 0 7 9
    //1 3 4 2 5
    //1 6 0 7 9
    //1 3 4 2 5

    mtx1[0]=1;
    mtx1[1]=3;
    mtx1[2]=4;
    mtx1[3]=2;
    mtx1[4]=5;
    mtx1[5]=1;
    mtx1[6]=6;
    mtx1[7]=0;
    mtx1[8]=7;
    mtx1[9]=9;
    mtx1[10]=1;
    mtx1[11]=3;
    mtx1[12]=4;
    mtx1[13]=2;
    mtx1[14]=5;
    mtx1[15]=1;
    mtx1[16]=6;
    mtx1[17]=0;
    mtx1[18]=7;
    mtx1[19]=9;
    mtx1[20]=1;
    mtx1[21]=3;
    mtx1[22]=4;
    mtx1[23]=2;
    mtx1[24]=5;
    mtx1[25]=1;

    mtx2[0]=1;
    mtx2[1]=3;
    mtx2[2]=6;
    mtx2[3]=4;
    mtx2[4]=2;
    mtx2[5]=5;
    mtx2[6]=1;
    mtx2[7]=3;
    mtx2[8]=6;
    mtx2[9]=4;
    mtx2[10]=1;
    mtx2[11]=3;
    mtx2[12]=6;
    mtx2[13]=4;
    mtx2[14]=2;
    mtx2[15]=5;
    mtx2[16]=1;
    mtx2[17]=3;
    mtx2[18]=6;
    mtx2[19]=4;
    mtx2[20]=2;
    mtx2[21]=3;
    mtx2[22]=6;
    mtx2[23]=4;
    mtx2[24]=2;
    mtx2[25]=5;

    //printint(mtx1,rows,cols);
    //printint(mtx2,cols,rows);

    printflt(mtx1,rows,cols);
    printflt(mtx2,cols,rows);

    //int16_t *mtx3 = calloc(5 * 5, sizeof(int16_t));
    float *mtx3 = calloc(5 * 5, sizeof(float));
    
    //multiplyint(mtx1,mtx2,mtx3,rows,cols,cols,rows);
    multiplyflt(mtx1,mtx2,mtx3,rows,cols,cols,rows);

    //simdint(mtx1,mtx3,mtx4,rows,cols,cols,rows);
    //int16_t *mtx4 = multiplyint(mtx1, mtx3,rows,cols,cols,rows);

    //printint(mtx3,rows,rows);
    printflt(mtx3,rows,rows);

    //free(mtx3);
}


