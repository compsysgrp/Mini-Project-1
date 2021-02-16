int main()
{
    int msec = 0, trigger = 10;
    printf("This version of the code is for comparing SIMD multiplication with traditional mutliplication with no user input for data collection purposes\r\n");

    //declaring all result matrices
    int16_t* resultint0;
    int16_t* resultint1;
    int16_t* resultint2;
    int16_t* resultint3;
    int16_t* resultint0simd;
    int16_t* resultint1simd;
    int16_t* resultint2simd;
    int16_t* resultint3simd;   
    float* resultflt0; 
    float* resultflt1; 
    float* resultflt2; 
    float* resultflt3;
    float* resultflt0simd;
    float* resultflt1simd; 
    float* resultflt2simd; 
    float* resultflt3simd;
    clock_t start;
    float endtimer;
    
    //creating all the matrices to be multiplied
    mtx00_int = intmatrix(100,100);
    mtx01_int = intmatrix(100,100);
    mtx1_int = intmatrix(1000,1000);
    mtx2_int = intmatrix(1000,1000);
    mtx3_int = intmatrix(5000,5000);
    mtx4_int = intmatrix(5000,5000);
    mtx5_int = intmatrix(10000,10000);
    mtx6_int = intmatrix(10000,10000);

    mtx00_flt = fltmatrix(100,100);
    mtx01_flt = fltmatrix(100,100);
    mtx1_flt = fltmatrix(1000,1000);
    mtx2_flt = fltmatrix(1000,1000);
    mtx3_flt = fltmatrix(5000,5000);
    mtx4_flt = fltmatrix(5000,5000);
    mtx5_flt = fltmatrix(10000,10000);
    mtx6_flt = fltmatrix(10000,10000);
    
    //timing traditional short
    printf("Timing Traditional multiplication for short...\r\n");
    start = clock();
    resultint0= multiplyint(mtx00_int,mtx01_int, 100, 100, 100, 100);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("100x100 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint0)

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
    resultint0simd= simdint(mtx00_int,mtx01_int, 100, 100, 100, 100);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("100x100 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint0simd);

    start = clock();
    resultint1simd= simdint(mtx1_int,mtx2_int, 1000, 1000, 1000, 1000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("1000x1000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint1simd);

    start = clock();
    resultint2simd= simdint(mtx3_int,mtx4_int, 5000, 5000, 5000, 5000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("5000x5000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultint2simd);

    start = clock();
    resultint3simd= simdint(mtx5_int,mtx6_int, 10000, 10000, 10000, 10000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("10000x10000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);  
    free(resultint3simd);

    printf("\r\n");

    //timing SIMD float
    start = clock();
    resultflt0simd= simdflt(mtx00_flt,mtx01_flt, 100, 100, 100, 100);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("100x100 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt0simd);

    printf("Timing SIMD multiplication for float...\r\n");
    start = clock();
    resultflt1simd= simdflt(mtx1_flt,mtx2_flt, 1000, 1000, 1000, 1000);
    endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("1000x1000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt1simd);

    start = clock();
    resultflt2simd= simdflt(mtx3_flt,mtx4_flt, 5000, 5000, 5000, 5000);
    float endtimer = clock() - start;
    msec= endtimer *1000 / CLOCKS_PER_SEC;
    printf("5000x5000 took %d seconds, %d milliseconds: \r\n", msec/1000, msec%1000);
    free(resultflt2simd);

    start = clock();
    resultflt3simd= simdflt(mtx5_flt,mtx6_flt, 10000, 10000, 10000, 10000);
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