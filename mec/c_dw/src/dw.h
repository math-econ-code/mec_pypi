/*
 * dw.h - C port of mec/dw.py
 *
 * Dantzig-Wolfe / RROA column generation for transferable-utility matching.
 * Mirrors the Python TUMatching (forward) and TUMatchingEstimation
 * (lambda_k parametrized) classes.  LP solves use Gurobi's C API; the
 * column-generation pricing step is parallelized with OpenMP when
 * compiled with -fopenmp.
 */

#ifndef MEC_DW_H
#define MEC_DW_H

#include "gurobi_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- *
 *  Forward problem: TUMatching                                     *
 * ---------------------------------------------------------------- */
typedef struct {
    int I, J, X, Y;

    /* primitives */
    double *Phi_x_y;          /* X*Y */
    double *eps_i_y;          /* I*Y */
    double *eta_x_j;          /* X*J */
    double *eps_i_0;          /* I   */
    double *eta_0_j;          /* J   */
    int    *x_i;              /* I   */
    int    *y_j;              /* J   */

    /* derived */
    double *alpha_i_y;        /* I*Y, Phi[x_i,y]/2 + eps_i_y */
    double *gamma_x_j;        /* X*J, Phi[x,y_j]/2 + eta_x_j */
    int    *first_j_by_y;     /* Y, representative j-of-type-y */

    /* RMP active sets */
    char   *Y_i_y;            /* I*Y bool */
    char   *X_j_x;            /* J*X bool */

    /* Gurobi handles */
    GRBenv   *env;
    GRBmodel *model;
} TUMatching;

TUMatching *tu_matching_create(int I, int J, int X, int Y,
                               const double *Phi_x_y,
                               const double *eps_i_y,
                               const double *eta_x_j,
                               const double *eps_i_0,
                               const double *eta_0_j,
                               const int    *x_i,
                               const int    *y_j);

void tu_matching_free(TUMatching *T);

/* Solve via column generation (RROA). Returns 0 on success. */
int tu_matching_rroa_solve(TUMatching *T,
                           int max_iter,
                           double rc_tol,
                           int verbose,
                           double *objval_out);


/* ---------------------------------------------------------------- *
 *  Estimation: Phi_x_y = sum_k lambda_k * phi_x_y_k                *
 * ---------------------------------------------------------------- */
typedef struct {
    int I, J, X, Y, K;

    /* observed counts */
    int *mu_x_y, *mu_x0, *mu_0y;

    /* basis */
    double *phi_x_y_k;        /* X*Y*K */

    /* expanded types */
    int *x_i, *y_j;

    /* shocks */
    double *eps_i_0, *eta_0_j, *eps_i_y, *eta_x_j;

    /* RMP active sets */
    char *Y_i_y, *X_j_x, *active_i_0, *active_0_j;

    /* Gurobi handles */
    GRBenv   *env;
    GRBmodel *model;
} TUMatchingEstimation;

TUMatchingEstimation *tu_estimation_create(int X, int Y, int K,
                                           const int    *mu_x_y,
                                           const int    *mu_x0,
                                           const int    *mu_0y,
                                           const double *phi_x_y_k,
                                           const double *eps_i_0,
                                           const double *eta_0_j,
                                           const double *eps_i_y,
                                           const double *eta_x_j);

void tu_estimation_free(TUMatchingEstimation *E);

/* RROA. lambda_k_out has length K (may be NULL). Returns 0 on success. */
int tu_estimation_rroa_solve(TUMatchingEstimation *E,
                             int max_iter,
                             double rc_tol,
                             int verbose,
                             double *lambda_k_out,
                             double *objval_out);

#ifdef __cplusplus
}
#endif

#endif /* MEC_DW_H */
