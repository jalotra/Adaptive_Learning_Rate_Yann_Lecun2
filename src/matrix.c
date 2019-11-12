#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>


void free_matrix(matrix m)
{
    if (m.data) {
        int i;
        if (!m.shallow) for(i = 0; i < m.rows; ++i) free(m.data[i]);
        free(m.data);
    }
}

matrix make_matrix(int rows, int cols)
{
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.shallow = 0;
    m.data = calloc(m.rows, sizeof(double *));
    int i;
    for(i = 0; i < m.rows; ++i) m.data[i] = calloc(m.cols, sizeof(double));
    return m;
}

matrix copy_matrix(matrix m)
{
    int i,j;
    matrix c = make_matrix(m.rows, m.cols);
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            c.data[i][j] = m.data[i][j];
        }
    }
    return c;
}


matrix matrix_mult_matrix(matrix a, matrix b)
{
    assert(a.cols == b.rows);
    int i, j, k;
    matrix p = make_matrix(a.rows, b.cols);
    for(i = 0; i < p.rows; ++i){
        for(j = 0; j < p.cols; ++j){
            for(k = 0; k < a.cols; ++k){
                p.data[i][j] += a.data[i][k]*b.data[k][j];
            }
        }
    }
    return p;
}

matrix matrix_elmult_matrix(matrix a, matrix b)
{
    assert(a.cols == b.cols);
    assert(a.rows == b.rows);
    int i, j;
    matrix p = make_matrix(a.rows, a.cols);
    for(i = 0; i < p.rows; ++i){
        for(j = 0; j < p.cols; ++j){
            p.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }
    return p;
}


matrix axpy_matrix(double a, matrix x, matrix y)
{
    assert(x.cols == y.cols);
    assert(x.rows == y.rows);
    int i, j;
    matrix p = make_matrix(x.rows, x.cols);
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            p.data[i][j] = a*x.data[i][j] + y.data[i][j];
        }
    }
    return p;
}
double box_muller_normal_random(double m, double s)   /* normal random variate generator */
{                       /* mean m, standard deviation s */
    double x1, x2, w, _y1;
    double y2;
    int use_last = 0;

    if (use_last)               /* use value from previous call */
    {
        _y1 = y2;
        use_last = 0;
    }
    else
    {
        do {
            x1 = 2.0 * drand48() - 1.0;
            x2 = 2.0 * drand48() - 1.0;
            w = x1 * x1 + x2 * x2;
        } while ( w >= 1.0 );

        w = sqrt( (-2.0 * log( w ) ) / w );
        _y1 = x1 * w;
        y2 = x2 * w;
        use_last = 1;
    }

        double result = ( m + _y1 * s );

    return result;
}

matrix normal_random_matrix(int rows, int cols, double m, double s)
{
    matrix mat = make_matrix(rows, cols);
    
    int i, j;
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            mat.data[i][j] =  (double) box_muller_normal_random(m , s);    
        }
    }
    return mat;
}


matrix transpose_matrix(matrix m)
{
    matrix t;
    t.rows = m.cols;
    t.cols = m.rows;
    t.data = calloc(t.rows, sizeof(double *));
    t.shallow = 0;
    int i, j;
    for(i = 0; i < t.rows; ++i){
        t.data[i] = calloc(t.cols, sizeof(double));
        for(j = 0; j < t.cols; ++j){
            t.data[i][j] = m.data[j][i];
        }
    }
    return t;
}

matrix scale_matrix(double s, matrix m)
{
    int i, j;
    matrix new_m = make_matrix(m.rows, m.cols);
    for(i = 0; i < m.rows; ++i){
        for(j =0 ; j < m.cols; ++j){
            new_m.data[i][j] = m.data[i][j]*s;
        }
    }

    return new_m;
}


void print_matrix(matrix originalMatrix)
{
    int i,j;

    printf("The number of rows present are : %d \n", originalMatrix.rows);
    printf("The number of cols present are : %d \n", originalMatrix.cols);

    for(i = 0; i < originalMatrix.rows ; i++)
    {
        for(j = 0; j < originalMatrix.cols; j++)
        {
            printf("%f \t", originalMatrix.data[i][j]);
        }
        printf("\n");
    }
}

matrix random_matrix(int rows, int cols, double s)
{
    matrix m = make_matrix(rows, cols);
    int i, j;
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            m.data[i][j] = 2*s*(rand()%1000/1000.0) - s;    
        }
    }
    return m;
}



void test_matrix()
{
    return ;
}

// MY METHODS

matrix matrix_sub_matrix(matrix a, matrix b)
{
    // Matrix will be possible only if the dimensions match up
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);

    matrix m = make_matrix(a.rows, a.cols);
    for_loop(i, a.rows)
    {
        for_loop(j, a.cols)
        {
            m.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }

    return m;
}

matrix matrix_add_matrix(matrix a, matrix b)
{
    // Matrix will be possible only if the dimensions match up
    assert(a.rows == b.rows);
    assert(b.cols == a.cols);

    matrix m = make_matrix(a.rows, a.cols);
    for_loop(i, a.rows)
    {
        for_loop(j, a.cols)
        {
            m.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }

    return m;
}