/*
 * test_dw.c - smoke test for libdw.
 *
 * Runs RROA on a small forward problem and on a small lambda_k
 * estimation problem and prints the objective values. Equivalence
 * with the Python solver is validated in Python; this binary just
 * surfaces a C build break early.
 */

#include "dw.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static unsigned long _rng = 12345UL;
static double urand(void) {
    _rng = _rng * 1103515245UL + 12345UL;
    return (double)((_rng >> 16) & 0x7fffffff) / (double)0x7fffffff;
}
static double nrand(void) {
    double u1 = urand(); if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * urand());
}

static int forward(void)
{
    const int X = 4, Y = 4, I = 40, J = 40;
    double *Phi = malloc(sizeof(double)*X*Y);
    double *eiy = malloc(sizeof(double)*I*Y);
    double *exj = malloc(sizeof(double)*X*J);
    double *ei0 = malloc(sizeof(double)*I);
    double *e0j = malloc(sizeof(double)*J);
    int    *x_i = malloc(sizeof(int)*I);
    int    *y_j = malloc(sizeof(int)*J);
    _rng = 1u;
    for (int p = 0; p < X*Y; ++p) Phi[p] = nrand();
    for (int p = 0; p < I*Y; ++p) eiy[p] = nrand();
    for (int p = 0; p < X*J; ++p) exj[p] = nrand();
    for (int i = 0; i < I; ++i)   ei0[i] = nrand();
    for (int j = 0; j < J; ++j)   e0j[j] = nrand();
    for (int i = 0; i < I; ++i)   x_i[i] = i % X;
    for (int j = 0; j < J; ++j)   y_j[j] = j % Y;

    TUMatching *T = tu_matching_create(I, J, X, Y, Phi, eiy, exj, ei0, e0j, x_i, y_j);
    double obj = 0.0;
    int rc = tu_matching_rroa_solve(T, 200, 1e-6, 0, &obj);
    printf("forward rroa     : obj = %.6f\n", obj);
    tu_matching_free(T);
    free(Phi); free(eiy); free(exj); free(ei0); free(e0j); free(x_i); free(y_j);
    return rc;
}

static int estimation(void)
{
    const int X = 3, Y = 3, K = 2;
    int mu[3*3]  = { 2,1,0, 0,2,1, 1,0,2 };
    int mu_x0[3] = { 1, 1, 1 };
    int mu_0y[3] = { 1, 1, 1 };
    double phi[3*3*2];
    for (int x = 0; x < X; ++x)
        for (int y = 0; y < Y; ++y) {
            phi[(x*Y+y)*K + 0] = (double)(x*y);
            phi[(x*Y+y)*K + 1] = -((double)(x-y)*(x-y));
        }
    int I = 0, J = 0;
    for (int x = 0; x < X; ++x) { I += mu_x0[x]; for (int y = 0; y < Y; ++y) I += mu[x*Y+y]; }
    for (int y = 0; y < Y; ++y) { J += mu_0y[y]; for (int x = 0; x < X; ++x) J += mu[x*Y+y]; }

    double *ei0 = malloc(sizeof(double)*I);
    double *e0j = malloc(sizeof(double)*J);
    double *eiy = malloc(sizeof(double)*I*Y);
    double *exj = malloc(sizeof(double)*X*J);
    _rng = 7u;
    for (int i = 0; i < I; ++i)   ei0[i] = nrand();
    for (int j = 0; j < J; ++j)   e0j[j] = nrand();
    for (int p = 0; p < I*Y; ++p) eiy[p] = nrand();
    for (int p = 0; p < X*J; ++p) exj[p] = nrand();

    TUMatchingEstimation *E = tu_estimation_create(X, Y, K, mu, mu_x0, mu_0y, phi,
                                                    ei0, e0j, eiy, exj);
    double lam[2] = {0,0}, obj = 0.0;
    int rc = tu_estimation_rroa_solve(E, 200, 1e-6, 0, lam, &obj);
    printf("estimation rroa  : obj = %.6f  lambda = [%.6f, %.6f]\n", obj, lam[0], lam[1]);
    tu_estimation_free(E);
    free(ei0); free(e0j); free(eiy); free(exj);
    return rc;
}

int main(void)
{
    if (forward())    return 1;
    if (estimation()) return 1;
    return 0;
}
