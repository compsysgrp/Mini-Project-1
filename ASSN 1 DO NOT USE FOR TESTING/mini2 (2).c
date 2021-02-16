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
    // int rows = 5;
    // int cols = 5;

    int rows = 3;
    int cols = 3;

    int16_t *mtx1 = calloc(rows * cols, sizeof(float));
    int16_t *mtx2 = calloc(cols * rows, sizeof(float));

    // float *mtx1 = calloc(rows * cols, sizeof(float));
    // float *mtx2 = calloc(rows * cols, sizeof(float));

    // test matrix 1: 5 x 5
    // 1 3 4 2 5
    // 1 6 0 7 9
    // 1 3 4 2 5
    // 1 6 0 7 9
    // 1 3 4 2 5

    // mtx1[0]=1;
    // mtx1[1]=3;
    // mtx1[2]=4;
    // mtx1[3]=2;
    // mtx1[4]=5;
    // mtx1[5]=1;
    // mtx1[6]=6;
    // mtx1[7]=0;
    // mtx1[8]=7;
    // mtx1[9]=9;
    // mtx1[10]=1;
    // mtx1[11]=3;
    // mtx1[12]=4;
    // mtx1[13]=2;
    // mtx1[14]=5;
    // mtx1[15]=1;
    // mtx1[16]=6;
    // mtx1[17]=0;
    // mtx1[18]=7;
    // mtx1[19]=9;
    // mtx1[20]=1;
    // mtx1[21]=3;
    // mtx1[22]=4;
    // mtx1[23]=2;
    // mtx1[24]=5;
    // mtx1[25]=1;
    
    // mtx2[0]=1;
    // mtx2[1]=3;
    // mtx2[2]=6;
    // mtx2[3]=4;
    // mtx2[4]=2;
    // mtx2[5]=5;
    // mtx2[6]=1;
    // mtx2[7]=3;
    // mtx2[8]=6;
    // mtx2[9]=4;
    // mtx2[10]=1;
    // mtx2[11]=3;
    // mtx2[12]=6;
    // mtx2[13]=4;
    // mtx2[14]=2;
    // mtx2[15]=5;
    // mtx2[16]=1;
    // mtx2[17]=3;
    // mtx2[18]=6;
    // mtx2[19]=4;
    // mtx2[20]=2;
    // mtx2[21]=3;
    // mtx2[22]=6;
    // mtx2[23]=4;
    // mtx2[24]=2;
    // mtx2[25]=5;

    mtx1[0]=1;
    mtx1[1]=2;
    mtx1[2]=3;
    mtx1[3]=4;
    mtx1[4]=5;
    mtx1[5]=6;
    mtx1[6]=7;
    mtx1[7]=8;
    mtx1[8]=9;

    mtx2[0]=1;
    mtx2[1]=2;
    mtx2[2]=3;
    mtx2[3]=4;
    mtx2[4]=5;
    mtx2[5]=6;
    mtx2[6]=7;
    mtx2[7]=8;
    mtx2[8]=9;

    printint(mtx1,rows,cols);
    printint(mtx2,cols,rows);

    // printflt(mtx1,rows,cols);
    // printflt(mtx2,cols,rows);

    int16_t *mtx3 = calloc(rows * rows, sizeof(int16_t));
    //simdint(mtx1,mtx2,mtx3,rows,cols,cols,rows);
    blockint(mtx1,mtx2,mtx3,rows,cols,cols,rows,2);
    printint(mtx3,rows,rows);

    // float *mtx3 = calloc(rows * cols, sizeof(float));
    // simdflt(mtx1,mtx2,mtx3,rows,cols,cols,rows);
    // blockflt(mtx1,mtx2,mtx3,rows,cols,cols,rows,2);
    // printflt(mtx3,rows,rows);

    free(mtx1);
    free(mtx2);
    free(mtx3);
}


