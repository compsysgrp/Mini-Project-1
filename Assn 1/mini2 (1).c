#include <stdio.h>
#include <stdlib.h>
#include <stdint-gcc.h>
#include <immintrin.h>
#include <math.h>

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
    //int val = 0; //variable used for printing matrix math 
    for(int i=0; i<r1; i++)
    {
         for (int j = 0; j<c2; j++)
         {
            int k;
            for (int k = 0; k<r2; k++)
            {   
                //makes sure the starting value of matrix 3 is 0
                if (k == 0)
                {
                    mtx3[i*c2+j] = 0;
                }

                __m256i ans = _mm256_loadu_si256((__m256i*) &mtx3[i*c2+j]); //get the current matrix value 
                __m256i nm1 = _mm256_loadu_si256((__m256i*) &mtx1[i*c1+k]); //get value from matrix 1
                __m256i nm2 = _mm256_loadu_si256((__m256i*) &mtx2[k*c2+j]); //get value from matrix 2

                __m256i mlt = _mm256_mullo_epi16 (nm1, nm2); //multiply value from matrix 1 and matrix 2

                ans = _mm256_add_epi16(ans,mlt); //add multipled variable to existing matrix 3 value

                _mm256_storeu_si256((__m256i*) (mtx3 + i*c2+j),ans); //store new calculated value
                
                //print statements to show matrix math
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
            //val = 0; //resets variable to 0 
            //printf("\r\n"); //formatting
           
         }
        
    }
}

void simdflt(float* mtx1, float* mtx2, float* mtx3, int r1, int c1, int r2, int c2)
{
    //float val = 0; //variable used for printing matrix math 
    for(int i=0; i<r1; i++)
    {
         for (int j = 0; j<c2; j++)
         {
            int k;
            for (int k = 0; k<r2; k++)
            {
                //makes sure the starting value of matrix 3 is 0
                if (k == 0)
                {
                    mtx3[i*c2+j] = 0;
                }
                __m256 ans = _mm256_loadu_ps(mtx3 + i*c2+j); //get the current matrix value  
                __m256 nm1 = _mm256_loadu_ps(mtx1 + i*c1+k); //get value from matrix 1
                __m256 nm2 = _mm256_loadu_ps(mtx2 + k*c2+j); //get value from matrix 2

                ans = _mm256_fmadd_ps(nm1,nm2,ans); //add and multiple variables from matrix 1 and 2 to existing matrix 3 value

                _mm256_storeu_ps(mtx3 + i*c2+j,ans); //store new calculated value
                
                /*
                if (val == 0)
                {
                    val = mtx1[i*c1+k] * mtx2[k*c2+j];
                    printf("%.6f: %.6f * %.6f\r\n",mtx3[i*c2+j], mtx1[i*c1+k],mtx2[k*c2+j]);
                }
                else
                {   
                    printf("%.6f: %.6f + (%.6f * %.6f)\r\n",mtx3[i*c2+j], val, mtx1[i*c1+k],mtx2[k*c2+j]);
                    val = mtx3[i*c2+j] + mtx1[i*c1+k]*mtx2[k*c2+j];
                }
                */
            }
            //val = 0; //resets variable to 0 
            //printf("\r\n"); //formatting
           
         }
        
    }
}

void blockint(int16_t *mtx1,int16_t *mtx2, int16_t *mtx3, int r1, int c1, int r2, int c2, int block_size)
{
    int val = 0;
    for(int k = 0; k<r2; k+= block_size)
    {
        for(int j = 0; j<c2; j+=block_size)
        {
            for(int i = 0; i<r1; i++)
            {
                for(int j2 = j; j2 < fmin(j + block_size, c2); j2++)
                {
                    for(int k2 = k; k2 < fmin(k + block_size, r2); k2++)
                    {
                        //val = mtx3[i*c2+j2];

                        mtx3[i*c2+j2] += mtx1[i*c1+k2]*mtx2[k2*c2+j2];
                            
                        //printf("mtx[%d]: %d = %d + (%d * %d)\r\n",i*c2+j2,mtx3[i*c2+j2], val, mtx1[i*c1+k2],mtx2[k2*c2+j2]);
                            
                    }
                }               
            }
        }
    }
}

void blockflt(float *mtx1,float *mtx2, float *mtx3, int r1, int c1, int r2, int c2, int block_size)
{
    int val = 0;
    for(int k = 0; k<r2; k+= block_size)
    {
        for(int j = 0; j<c2; j+=block_size)
        {
            for(int i = 0; i<r1; i++)
            {
                for(int j2 = j; j2 < fmin(j + block_size, c2); j2++)
                {
                    for(int k2 = k; k2 < fmin(k + block_size, r2); k2++)
                    {
                        //val = mtx3[i*c2+j2];

                        mtx3[i*c2+j2] += mtx1[i*c1+k2]*mtx2[k2*c2+j2];
                            
                        //printf("mtx[%d]: %d = %d + (%d * %d)\r\n",i*c2+j2,mtx3[i*c2+j2], val, mtx1[i*c1+k2],mtx2[k2*c2+j2]);
                            
                    }
                }               
            }
        }
    }
}

int main ()
{
    int16_t *mtx1 = calloc(1000 * 1000, sizeof(float));
    int16_t *mtx2 = calloc(1000 * 1000, sizeof(float));
    int16_t *mtx3 = calloc(2000 * 2000, sizeof(float));
    int16_t *mtx4 = calloc(2000 * 2000, sizeof(float));
    int16_t *mtx5 = calloc(3000 * 3000, sizeof(float));
    int16_t *mtx6 = calloc(3000 * 3000, sizeof(float));
    int16_t *mtx7 = calloc(4000 * 4000, sizeof(float));
    int16_t *mtx8 = calloc(4000 * 4000, sizeof(float));
    int16_t *mtx9 = calloc(5000 * 5000, sizeof(float));
    int16_t *mtx10 = calloc(5000 * 5000, sizeof(float));

    int16_t *mtx1out = calloc(1000 * 1000, sizeof(int16_t));
    int16_t *mtx2out = calloc(2000 * 2000, sizeof(int16_t));
    int16_t *mtx3out = calloc(3000 * 3000, sizeof(int16_t));
    int16_t *mtx4out = calloc(4000 * 4000, sizeof(int16_t));
    int16_t *mtx5out = calloc(5000 * 5000, sizeof(int16_t));

    clock_t start = clock();
    float endtimer;
    
    printf("Timing Cache-optimized (block) multiplication for int...\r\n");

    start = clock();
    blockint(mtx1,mtx2, mtx1out, 1000, 1000, 1000, 1000,block1);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("1000x1000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);  

    start = clock();
    blockint(mtx1,mtx2, mtx2out, 1000, 1000, 1000, 1000,block2);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("2000x2000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);  

    start = clock();
    blockint(mtx1,mtx2, mtx3out, 1000, 1000, 1000, 1000,block3);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("3000x3000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);  

    start = clock();
    blockint(mtx1,mtx2, mtx4out, 1000, 1000, 1000, 1000,block4);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("4000x4000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);  

    start = clock();
    blockint(mtx1,mtx2, mtx5out, 1000, 1000, 1000, 1000,block5);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("5000x5000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);  


    free(mtx1);
    free(mtx2);
    free(mtx3);
    free(mtx4);
    free(mtx5);
    free(mtx6);
    free(mtx7);
    free(mtx8);
    free(mtx9);
    free(mtx10);
    free(mtx1out;
    free(mtx2out);
    free(mtx3out);
    free(mtx4out);
    free(mtx5out);
}


