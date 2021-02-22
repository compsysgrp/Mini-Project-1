#include <stdio.h>
#include <stdlib.h>
#include <stdint-gcc.h>
#include <immintrin.h>
#include <math.h>
#include <time.h>


int16_t* intmatrix(int w, int l)
{
    int16_t *mtx = calloc(w * l, sizeof(int16_t));
    for(int i=0; i<w*l; i++)
    {
        int val = rand() % 10;
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


void multiplyint(int16_t* mtx1, int16_t* mtx2, int16_t* mtx3, int r1, int c1, int r2, int c2)
{
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
}

void multiplyflt(float* mtx1, float* mtx2,float* mtx3, int r1, int c1, int r2, int c2)
{
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
}

void simdint(int16_t *mtx1,int16_t *mtx2, int16_t *mtx3, int r1, int c1, int r2, int c2)
{
    for(int i=0; i<r1; i++)
    {
         for (int j = 0; j<c2; j++)
         {
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
            }
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

                        mtx3[i*c2+j2] += mtx1[i*c1+k2]*mtx2[k2*c2+j2];
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
                        mtx3[i*c2+j2] += mtx1[i*c1+k2]*mtx2[k2*c2+j2];         
                    }
                }               
            }
        }
    }
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
    int block_size;
    printf("Enter # of rows for matrix 1: \r\n");
    scanf("%d", &r1);
    printf("Enter # of columns for matrix 1: \r\n");
    scanf("%d", &c1);
    printf("Enter # of rows for matrix 2: \r\n");
    scanf("%d", &r2);
    printf("Enter # of columns for matrix 2: \r\n");
    scanf("%d", &c2);
    printf("enter 0 for short, 1 for float: \r\n");
    scanf("%d", &datatype);
    printf("enter 0 for traditional multiplication, 1 for SIMD multiplication, 2 for block multiplication\r\n");
    scanf("%d", &multiplytype);
    printf("enter 0 for no printing, 1 for printing: \r\n");
    scanf("%d", &print);

    int msec = 0, trigger = 10;
    int16_t* resultint;
    int16_t* intmtx1;
    int16_t* intmtx2;
    float* resultflt;
    float* fltmtx1;
    float* fltmtx2;

    if (multiplytype == 0)
    {
        printf("Starting timer for traditional multiplication...\r\n");
        clock_t start = clock();
        if (datatype == 0)
        {
            intmtx1 = intmatrix(r1, c1);
            intmtx2 = intmatrix(r2, c2);
            resultint = calloc(r1 * c2, sizeof(int16_t*));
            multiplyint(intmtx1, intmtx2, resultint, r1, c1, r2, c2);
        }
        else if (datatype == 1)
        {
            fltmtx1 = fltmatrix(r1, c1);
            fltmtx2 = fltmatrix(r2, c2);
            resultflt = calloc(r1 * c2, sizeof(float));
            multiplyflt(fltmtx1, fltmtx2, resultflt, r1, c1, r2, c2);
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
            resultint = calloc(r1 * c2, sizeof(int16_t));
            intmtx1 = intmatrix(r1, c1);
            intmtx2 = intmatrix(r2, c2);
            simdint(intmtx1, intmtx2, resultint, r1, c1, r2, c2);
        }
        else if (datatype == 1)
        {
            resultflt = calloc(r1 * c2, sizeof(float));
            fltmtx1 = fltmatrix(r1, c1);
            fltmtx2 = fltmatrix(r2, c2);
            simdflt(fltmtx1, fltmtx2, resultflt, r1, c1, r2, c2);
        }

        float endtimer = clock() - start;
        msec= endtimer *1000 / CLOCKS_PER_SEC;
        printf("SIMD multiplication took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);    
    }
    else if (multiplytype ==2)
    {
        printf("Enter block size: \r\n");
        scanf("%d", &block_size);
        printf("starting timer for block multiplication...\r\n");
        clock_t start = clock();
        if (datatype ==0 )
        {
            resultint = calloc(r1 * c2, sizeof(int16_t));
            intmtx1 = intmatrix(r1,c1);
            intmtx2 = intmatrix(r2, c2);
            blockint(intmtx1, intmtx2, resultint, r1, c1, r2, c2, block_size);
        }
        else if (datatype == 1)
        {
            resultflt = calloc(r1 * c2, sizeof(float));
            fltmtx1 = fltmatrix(r1, c1);
            fltmtx2 = fltmatrix(r2, c2);
            blockflt(fltmtx1, fltmtx2, resultflt, r1, c1, r2, c2, block_size);
        }

        float endtimer = clock() - start;
        msec= endtimer *1000 / CLOCKS_PER_SEC;
        printf("Block multiplication took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000); 
    }

    if (print == 1)
        {
            if (datatype == 0)
            {
                printf("Matrix 1:\r\n");
                printint(intmtx1,r1, c1);
                printf("Matrix 2:\r\n");
                printint(intmtx2,r2, c2);
                printf("New Matrix:\r\n");
                printint(resultint, r1, c2);
            }
            else if (datatype == 1)
            {
                printf("Matrix 1:\r\n");
                printflt(fltmtx1,r1, c1);
                printf("Matrix 2:\r\n");
                printflt(fltmtx2,r2, c2);
                printf("New Matrix:\r\n");
                printflt(resultflt, r1, c2);
            }
        }
    
    //this version has frees to remove memory leaks, for some reason causes a core dump (Only sometimes )even though there's no errors or memory leaks according to valgrind
    //and it makes it to line 324
    if (datatype==0)
    {
        free(intmtx1);
        free(intmtx2);
        free(resultint);
    }
    else if (datatype==1)
    {
        free(resultflt);
        free(fltmtx1);
        free(fltmtx2);
    }
}
