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