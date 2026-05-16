/*
 * dw.c - RROA / Dantzig-Wolfe column generation for TU matching.
 *
 * Constraint ordering inside the RMP (must match the Python reference):
 *   1. linking_x_y[x,y]   (X*Y)        indices [0, X*Y)
 *   2. row_i[i]           (I)          indices [X*Y, X*Y+I)
 *   3. row_j[j]           (J)          indices [X*Y+I, X*Y+I+J)
 *   4. linking_k[k]       (K, est)     indices [X*Y+I+J, X*Y+I+J+K)
 *
 * The pricing step (find_rc) is parallelized with OpenMP.
 */

#include "dw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#define CHECK(call) do { int _e = (call); if (_e) {                            \
    fprintf(stderr, "[dw] Gurobi error %d at %s:%d (%s)\n",                    \
            _e, __FILE__, __LINE__, #call); return _e; } } while (0)

static double *xcalloc_d(size_t n) {
    double *p = (double *)calloc(n, sizeof(double));
    if (!p) { fprintf(stderr, "calloc failed\n"); exit(1); } return p;
}
static int *xcalloc_i(size_t n) {
    int *p = (int *)calloc(n, sizeof(int));
    if (!p) { fprintf(stderr, "calloc failed\n"); exit(1); } return p;
}
static int *xfilled_i(size_t n, int v) {
    int *p = (int *)malloc(n * sizeof(int));
    if (!p) { fprintf(stderr, "malloc failed\n"); exit(1); }
    for (size_t i = 0; i < n; ++i) p[i] = v;
    return p;
}
static char *xcalloc_b(size_t n) {
    char *p = (char *)calloc(n, sizeof(char));
    if (!p) { fprintf(stderr, "calloc failed\n"); exit(1); } return p;
}


/* ================================================================ */
/*                      TUMatching - forward                        */
/* ================================================================ */

TUMatching *tu_matching_create(int I, int J, int X, int Y,
                               const double *Phi_x_y,
                               const double *eps_i_y,
                               const double *eta_x_j,
                               const double *eps_i_0,
                               const double *eta_0_j,
                               const int    *x_i,
                               const int    *y_j)
{
    TUMatching *T = (TUMatching *)calloc(1, sizeof(TUMatching));
    T->I = I; T->J = J; T->X = X; T->Y = Y;

    T->Phi_x_y = xcalloc_d((size_t)X * Y); memcpy(T->Phi_x_y, Phi_x_y, (size_t)X*Y*sizeof(double));
    T->eps_i_y = xcalloc_d((size_t)I * Y); memcpy(T->eps_i_y, eps_i_y, (size_t)I*Y*sizeof(double));
    T->eta_x_j = xcalloc_d((size_t)X * J); memcpy(T->eta_x_j, eta_x_j, (size_t)X*J*sizeof(double));
    T->eps_i_0 = xcalloc_d((size_t)I);     memcpy(T->eps_i_0, eps_i_0, (size_t)I*sizeof(double));
    T->eta_0_j = xcalloc_d((size_t)J);     memcpy(T->eta_0_j, eta_0_j, (size_t)J*sizeof(double));
    T->x_i = xcalloc_i((size_t)I);         memcpy(T->x_i, x_i, (size_t)I*sizeof(int));
    T->y_j = xcalloc_i((size_t)J);         memcpy(T->y_j, y_j, (size_t)J*sizeof(int));

    T->alpha_i_y = xcalloc_d((size_t)I * Y);
#pragma omp parallel for if(I > 64)
    for (int i = 0; i < I; ++i) {
        int x = T->x_i[i];
        for (int y = 0; y < Y; ++y)
            T->alpha_i_y[i*Y + y] = T->Phi_x_y[x*Y + y] * 0.5 + T->eps_i_y[i*Y + y];
    }
    T->gamma_x_j = xcalloc_d((size_t)X * J);
#pragma omp parallel for if(X*J > 256)
    for (int x = 0; x < X; ++x)
        for (int j = 0; j < J; ++j) {
            int yj = T->y_j[j];
            T->gamma_x_j[x*J + j] = T->Phi_x_y[x*Y + yj] * 0.5 + T->eta_x_j[x*J + j];
        }

    T->first_j_by_y = xfilled_i((size_t)Y, -1);
    for (int j = 0; j < J; ++j) {
        int y = T->y_j[j];
        if (T->first_j_by_y[y] == -1) T->first_j_by_y[y] = j;
    }

    T->Y_i_y = xcalloc_b((size_t)I * Y);
    T->X_j_x = xcalloc_b((size_t)J * X);
    return T;
}

void tu_matching_free(TUMatching *T)
{
    if (!T) return;
    free(T->Phi_x_y); free(T->eps_i_y); free(T->eta_x_j);
    free(T->eps_i_0); free(T->eta_0_j);
    free(T->x_i); free(T->y_j);
    free(T->alpha_i_y); free(T->gamma_x_j); free(T->first_j_by_y);
    free(T->Y_i_y); free(T->X_j_x);
    if (T->model) GRBfreemodel(T->model);
    if (T->env)   GRBfreeenv(T->env);
    free(T);
}

static void tu_matching_basic_feasible_solution(TUMatching *T)
{
    for (int y = 0; y < T->Y; ++y) {
        int j = T->first_j_by_y[y];
        if (j < 0) { fprintf(stderr, "[dw] no man of type y=%d\n", y); exit(1); }
        for (int x = 0; x < T->X; ++x) T->X_j_x[j*T->X + x] = 1;
    }
}

static int tu_matching_build_rmp(TUMatching *T, int verbose)
{
    int I = T->I, J = T->J, X = T->X, Y = T->Y;

    if (T->model) { GRBfreemodel(T->model); T->model = NULL; }
    if (T->env)   { GRBfreeenv(T->env);     T->env = NULL; }

    CHECK(GRBloadenv(&T->env, NULL));
    if (!verbose) GRBsetintparam(T->env, "OutputFlag", 0);

    CHECK(GRBnewmodel(T->env, &T->model, "RestrictedMP", 0, NULL, NULL, NULL, NULL, NULL));
    CHECK(GRBsetintattr(T->model, "ModelSense", GRB_MAXIMIZE));
    GRBenv *menv = GRBgetenv(T->model);
    GRBsetintparam(menv, "OutputFlag", verbose > 0);
    GRBsetintparam(menv, "UpdateMode", 1);   /* lazy update -> drop explicit updatemodel calls */

    /* All constraints in one batched call (linking_xy first, then row_i, row_j). */
    int nc = X*Y + I + J;
    char   *sense = (char   *)malloc((size_t)nc);
    double *rhs   = (double *)malloc((size_t)nc * sizeof(double));
    for (int p = 0; p < nc; ++p) sense[p] = GRB_EQUAL;
    for (int p = 0; p < X*Y; ++p)              rhs[p] = 0.0;
    for (int p = X*Y; p < nc; ++p)             rhs[p] = 1.0;
    CHECK(GRBaddconstrs(T->model, nc, 0, NULL, NULL, NULL, sense, rhs, NULL));
    free(sense); free(rhs);

    /* Count active matched columns. */
    int n_iy_act = 0, n_xj_act = 0;
    for (int p = 0; p < I*Y; ++p) if (T->Y_i_y[p]) ++n_iy_act;
    for (int p = 0; p < J*X; ++p) if (T->X_j_x[p]) ++n_xj_act;

    int total = I + n_iy_act + J + n_xj_act;
    int total_nz = I + 2*n_iy_act + J + 2*n_xj_act;

    int    *vbeg = (int    *)malloc((size_t)total    * sizeof(int));
    int    *vind = (int    *)malloc((size_t)total_nz * sizeof(int));
    double *vval = (double *)malloc((size_t)total_nz * sizeof(double));
    double *vobj = (double *)malloc((size_t)total    * sizeof(double));
    double *vlb  = (double *)calloc((size_t)total, sizeof(double));   /* zeros */

    int q = 0, p = 0;
    /* pi_i_0 */
    for (int i = 0; i < I; ++i) {
        vbeg[q] = p;
        vind[p] = X*Y + i;          vval[p++] = 1.0;
        vobj[q++] = T->eps_i_0[i];
    }
    /* active pi_i_y */
    for (int i = 0; i < I; ++i)
        for (int y = 0; y < Y; ++y) if (T->Y_i_y[i*Y + y]) {
            int x = T->x_i[i];
            vbeg[q] = p;
            vind[p] = x*Y + y;      vval[p++] = 1.0;
            vind[p] = X*Y + i;      vval[p++] = 1.0;
            vobj[q++] = T->alpha_i_y[i*Y + y];
        }
    /* pi_0_j */
    for (int j = 0; j < J; ++j) {
        vbeg[q] = p;
        vind[p] = X*Y + I + j;      vval[p++] = 1.0;
        vobj[q++] = T->eta_0_j[j];
    }
    /* active pi_x_j */
    for (int j = 0; j < J; ++j)
        for (int x = 0; x < X; ++x) if (T->X_j_x[j*X + x]) {
            int y = T->y_j[j];
            vbeg[q] = p;
            vind[p] = x*Y + y;      vval[p++] = -1.0;
            vind[p] = X*Y + I + j;  vval[p++] =  1.0;
            vobj[q++] = T->gamma_x_j[x*J + j];
        }

    CHECK(GRBaddvars(T->model, total, total_nz, vbeg, vind, vval,
                     vobj, vlb, NULL, NULL, NULL));
    free(vbeg); free(vind); free(vval); free(vobj); free(vlb);
    return 0;
}

/* Pricing step: parallel scan for best improving y per i and best x per j. */
static int tu_matching_find_rc(TUMatching *T, double rc_tol,
                               int **new_iy, int *n_iy_new,
                               int **new_jx, int *n_jx_new)
{
    int I = T->I, J = T->J, X = T->X, Y = T->Y;
    GRBmodel *m = T->model;

    /* Constraint indices are contiguous: t_xy [0, X*Y), u_i [X*Y, X*Y+I),
       v_j [X*Y+I, X*Y+I+J).  Fetch all duals in one Pi call.            */
    int n_constrs = X*Y + I + J;
    double *all_pi = xcalloc_d((size_t)n_constrs);
    CHECK(GRBgetdblattrarray(m, "Pi", 0, n_constrs, all_pi));
    const double *t_xy = all_pi;
    const double *u_i  = all_pi + X*Y;
    const double *v_j  = all_pi + X*Y + I;

    /* Per-i: best new y or -1.  Dense output -> no synchronization needed. */
    int *best_y = xfilled_i((size_t)I, -1);
#pragma omp parallel for if(I > 64) schedule(static)
    for (int i = 0; i < I; ++i) {
        int    by = -1;
        double br = rc_tol;
        int x = T->x_i[i];
        const double  u     = u_i[i];
        const double *alpha = T->alpha_i_y + (size_t)i*Y;
        const double *t_row = t_xy         + (size_t)x*Y;
        const char   *mask  = T->Y_i_y     + (size_t)i*Y;
        for (int y = 0; y < Y; ++y) {
            if (mask[y]) continue;
            double rc = alpha[y] - t_row[y] - u;
            if (rc > br) { br = rc; by = y; }
        }
        best_y[i] = by;
    }

    int *best_x = xfilled_i((size_t)J, -1);
#pragma omp parallel for if(J > 64) schedule(static)
    for (int j = 0; j < J; ++j) {
        int    bx = -1;
        double br = rc_tol;
        int y = T->y_j[j];
        const double v = v_j[j];
        for (int x = 0; x < X; ++x) {
            if (T->X_j_x[j*X + x]) continue;
            double rc = T->gamma_x_j[x*J + j] + t_xy[x*Y + y] - v;
            if (rc > br) { br = rc; bx = x; }
        }
        best_x[j] = bx;
    }

    /* Serial gather */
    int cap = 16, n = 0;
    int *iy = (int *)malloc((size_t)cap*2*sizeof(int));
    for (int i = 0; i < I; ++i) if (best_y[i] >= 0) {
        T->Y_i_y[i*Y + best_y[i]] = 1;
        if (n + 1 > cap) { cap *= 2; iy = (int *)realloc(iy, (size_t)cap*2*sizeof(int)); }
        iy[2*n + 0] = i; iy[2*n + 1] = best_y[i]; ++n;
    }
    int cap2 = 16, n2 = 0;
    int *jx = (int *)malloc((size_t)cap2*2*sizeof(int));
    for (int j = 0; j < J; ++j) if (best_x[j] >= 0) {
        T->X_j_x[j*X + best_x[j]] = 1;
        if (n2 + 1 > cap2) { cap2 *= 2; jx = (int *)realloc(jx, (size_t)cap2*2*sizeof(int)); }
        jx[2*n2 + 0] = j; jx[2*n2 + 1] = best_x[j]; ++n2;
    }

    *new_iy = iy; *n_iy_new = n;
    *new_jx = jx; *n_jx_new = n2;

    free(all_pi);
    free(best_y); free(best_x);
    return 0;
}

static int tu_matching_update_rmp(TUMatching *T,
                                  int *iy, int n_iy, int *jx, int n_jx)
{
    int X = T->X, Y = T->Y, I = T->I, J = T->J;
    int total = n_iy + n_jx;
    if (total == 0) return 0;

    int total_nz = 2 * total;
    int    *vbeg = (int    *)malloc((size_t)total    * sizeof(int));
    int    *vind = (int    *)malloc((size_t)total_nz * sizeof(int));
    double *vval = (double *)malloc((size_t)total_nz * sizeof(double));
    double *vobj = (double *)malloc((size_t)total    * sizeof(double));
    double *vlb  = (double *)calloc((size_t)total, sizeof(double));

    int q = 0, p = 0;
    for (int u = 0; u < n_iy; ++u) {
        int i = iy[2*u + 0], y = iy[2*u + 1];
        int x = T->x_i[i];
        vbeg[q] = p;
        vind[p] = x*Y + y;          vval[p++] = 1.0;
        vind[p] = X*Y + i;          vval[p++] = 1.0;
        vobj[q++] = T->alpha_i_y[i*Y + y];
    }
    for (int u = 0; u < n_jx; ++u) {
        int j = jx[2*u + 0], x = jx[2*u + 1];
        int y = T->y_j[j];
        vbeg[q] = p;
        vind[p] = x*Y + y;          vval[p++] = -1.0;
        vind[p] = X*Y + I + j;      vval[p++] =  1.0;
        vobj[q++] = T->gamma_x_j[x*J + j];
    }

    CHECK(GRBaddvars(T->model, total, total_nz, vbeg, vind, vval,
                     vobj, vlb, NULL, NULL, NULL));
    free(vbeg); free(vind); free(vval); free(vobj); free(vlb);
    return 0;
}

int tu_matching_rroa_solve(TUMatching *T, int max_iter, double rc_tol,
                           int verbose, double *objval_out)
{
    memset(T->Y_i_y, 0, (size_t)T->I * T->Y);
    memset(T->X_j_x, 0, (size_t)T->J * T->X);

    struct timespec _ts0, _ts1;
#define _T0() clock_gettime(CLOCK_MONOTONIC, &_ts0)
#define _T1() (clock_gettime(CLOCK_MONOTONIC, &_ts1), \
               (_ts1.tv_sec - _ts0.tv_sec) + (_ts1.tv_nsec - _ts0.tv_nsec) * 1e-9)
    double t_build = 0, t_pricing = 0, t_update = 0, t_solve = 0;

    _T0(); tu_matching_basic_feasible_solution(T);
    CHECK(tu_matching_build_rmp(T, verbose));
    t_build = _T1();

    GRBenv *menv = GRBgetenv(T->model);
    GRBsetintparam(menv, "OutputFlag", 0);
    _T0(); CHECK(GRBoptimize(T->model)); t_solve += _T1();

    int status; GRBgetintattr(T->model, "Status", &status);
    if (status != GRB_OPTIMAL) {
        fprintf(stderr, "[dw rroa] initial status %d\n", status);
        return -1;
    }
    GRBsetintparam(menv, "Presolve",    0);
    GRBsetintparam(menv, "LPWarmStart", 2);
    /* Concurrent solver beats primal/dual/barrier alone for warm-started
       column-generation re-solves on this class of LPs. */
    GRBsetintparam(menv, "Method",      3);

    int total_added = 0;
    for (int k = 0; k < max_iter; ++k) {
        int *iy=NULL,*jx=NULL, n_iy=0, n_jx=0;
        _T0(); CHECK(tu_matching_find_rc(T, rc_tol, &iy, &n_iy, &jx, &n_jx)); t_pricing += _T1();
        if (n_iy == 0 && n_jx == 0) {
            free(iy); free(jx);
            if (verbose > 0) fprintf(stderr, "Iter %2d: optimal.\n", k+1);
            break;
        }
        _T0(); CHECK(tu_matching_update_rmp(T, iy, n_iy, jx, n_jx)); t_update += _T1();
        free(iy); free(jx);
        total_added += n_iy + n_jx;
        _T0(); CHECK(GRBoptimize(T->model)); t_solve += _T1();
        if (verbose > 0) {
            double v; GRBgetdblattr(T->model, "ObjVal", &v);
            fprintf(stderr, "Iter %2d: obj = %.6f (added %d L + %d R)\n", k+1, v, n_iy, n_jx);
        }
    }

    if (verbose > 0 || getenv("DW_TIMING")) {
        fprintf(stderr, "[timing] build=%.3fs pricing=%.3fs update=%.3fs solve=%.3fs (added %d cols)\n",
                t_build, t_pricing, t_update, t_solve, total_added);
    }
    if (objval_out) GRBgetdblattr(T->model, "ObjVal", objval_out);
    return 0;
#undef _T0
#undef _T1
}


/* ================================================================ */
/*                TUMatchingEstimation - lambda_k                   */
/* ================================================================ */

TUMatchingEstimation *tu_estimation_create(int X, int Y, int K,
                                           const int    *mu_x_y,
                                           const int    *mu_x0,
                                           const int    *mu_0y,
                                           const double *phi_x_y_k,
                                           const double *eps_i_0,
                                           const double *eta_0_j,
                                           const double *eps_i_y,
                                           const double *eta_x_j)
{
    TUMatchingEstimation *E = (TUMatchingEstimation *)calloc(1, sizeof(TUMatchingEstimation));
    E->X = X; E->Y = Y; E->K = K;

    E->mu_x_y = xcalloc_i((size_t)X*Y); memcpy(E->mu_x_y, mu_x_y, (size_t)X*Y*sizeof(int));
    E->mu_x0  = xcalloc_i((size_t)X);   memcpy(E->mu_x0,  mu_x0,  (size_t)X*sizeof(int));
    E->mu_0y  = xcalloc_i((size_t)Y);   memcpy(E->mu_0y,  mu_0y,  (size_t)Y*sizeof(int));
    E->phi_x_y_k = xcalloc_d((size_t)X*Y*K);
    memcpy(E->phi_x_y_k, phi_x_y_k, (size_t)X*Y*K*sizeof(double));

    int I = 0, J = 0;
    for (int x = 0; x < X; ++x) {
        I += mu_x0[x];
        for (int y = 0; y < Y; ++y) I += mu_x_y[x*Y + y];
    }
    for (int y = 0; y < Y; ++y) {
        J += mu_0y[y];
        for (int x = 0; x < X; ++x) J += mu_x_y[x*Y + y];
    }
    E->I = I; E->J = J;

    E->x_i = xcalloc_i((size_t)I);
    {
        int p = 0;
        for (int x = 0; x < X; ++x) {
            int n = E->mu_x0[x];
            for (int y = 0; y < Y; ++y) n += E->mu_x_y[x*Y + y];
            for (int q = 0; q < n; ++q) E->x_i[p++] = x;
        }
    }
    E->y_j = xcalloc_i((size_t)J);
    {
        int p = 0;
        for (int y = 0; y < Y; ++y) {
            int n = E->mu_0y[y];
            for (int x = 0; x < X; ++x) n += E->mu_x_y[x*Y + y];
            for (int q = 0; q < n; ++q) E->y_j[p++] = y;
        }
    }

    E->eps_i_0 = xcalloc_d((size_t)I);   memcpy(E->eps_i_0, eps_i_0, (size_t)I*sizeof(double));
    E->eta_0_j = xcalloc_d((size_t)J);   memcpy(E->eta_0_j, eta_0_j, (size_t)J*sizeof(double));
    E->eps_i_y = xcalloc_d((size_t)I*Y); memcpy(E->eps_i_y, eps_i_y, (size_t)I*Y*sizeof(double));
    E->eta_x_j = xcalloc_d((size_t)X*J); memcpy(E->eta_x_j, eta_x_j, (size_t)X*J*sizeof(double));

    E->Y_i_y      = xcalloc_b((size_t)I*Y);
    E->X_j_x      = xcalloc_b((size_t)J*X);
    E->active_i_0 = xcalloc_b((size_t)I);
    E->active_0_j = xcalloc_b((size_t)J);
    return E;
}

void tu_estimation_free(TUMatchingEstimation *E)
{
    if (!E) return;
    free(E->mu_x_y); free(E->mu_x0); free(E->mu_0y);
    free(E->phi_x_y_k);
    free(E->x_i); free(E->y_j);
    free(E->eps_i_0); free(E->eta_0_j);
    free(E->eps_i_y); free(E->eta_x_j);
    free(E->Y_i_y); free(E->X_j_x);
    free(E->active_i_0); free(E->active_0_j);
    if (E->model) GRBfreemodel(E->model);
    if (E->env)   GRBfreeenv(E->env);
    free(E);
}

static void tu_estimation_basic_feasible_solution(TUMatchingEstimation *E)
{
    int X = E->X, Y = E->Y, I = E->I, J = E->J, K = E->K;

    int *cnt_x = xcalloc_i((size_t)X);
    for (int i = 0; i < I; ++i) cnt_x[E->x_i[i]]++;
    int *cnt_y = xcalloc_i((size_t)Y);
    for (int j = 0; j < J; ++j) cnt_y[E->y_j[j]]++;

    int **idx_i_by_x = (int **)calloc((size_t)X, sizeof(int *));
    int  *len_i      = xcalloc_i((size_t)X);
    for (int x = 0; x < X; ++x) idx_i_by_x[x] = (int *)malloc((size_t)cnt_x[x]*sizeof(int));
    for (int i = 0; i < I; ++i) { int x = E->x_i[i]; idx_i_by_x[x][len_i[x]++] = i; }
    int **idx_j_by_y = (int **)calloc((size_t)Y, sizeof(int *));
    int  *len_j      = xcalloc_i((size_t)Y);
    for (int y = 0; y < Y; ++y) idx_j_by_y[y] = (int *)malloc((size_t)cnt_y[y]*sizeof(int));
    for (int j = 0; j < J; ++j) { int y = E->y_j[j]; idx_j_by_y[y][len_j[y]++] = j; }

    memset(E->Y_i_y, 0, (size_t)I*Y);
    memset(E->X_j_x, 0, (size_t)J*X);
    memset(E->active_i_0, 0, (size_t)I);
    memset(E->active_0_j, 0, (size_t)J);

    for (int x = 0; x < X; ++x)
        for (int y = 0; y < Y; ++y) {
            int c = E->mu_x_y[x*Y + y];
            for (int q = 0; q < c; ++q) {
                int i = idx_i_by_x[x][--len_i[x]];
                int j = idx_j_by_y[y][--len_j[y]];
                E->Y_i_y[i*Y + y] = 1;
                E->X_j_x[j*X + x] = 1;
            }
        }
    for (int x = 0; x < X; ++x)
        for (int q = 0; q < len_i[x]; ++q) E->active_i_0[idx_i_by_x[x][q]] = 1;
    for (int y = 0; y < Y; ++y)
        for (int q = 0; q < len_j[y]; ++q) E->active_0_j[idx_j_by_y[y][q]] = 1;

    int needed = K + X*Y;
    for (int i = 0; i < I && needed > 0; ++i) if (!E->active_i_0[i]) { E->active_i_0[i] = 1; --needed; }
    for (int j = 0; j < J && needed > 0; ++j) if (!E->active_0_j[j]) { E->active_0_j[j] = 1; --needed; }
    if (needed > 0) {
        fprintf(stderr, "[dw est] not enough singles for initial basis\n"); exit(1);
    }

    for (int x = 0; x < X; ++x) free(idx_i_by_x[x]);
    for (int y = 0; y < Y; ++y) free(idx_j_by_y[y]);
    free(idx_i_by_x); free(idx_j_by_y);
    free(len_i); free(len_j);
    free(cnt_x); free(cnt_y);
}

static int tu_estimation_build_rmp(TUMatchingEstimation *E, int verbose)
{
    int I = E->I, J = E->J, X = E->X, Y = E->Y, K = E->K;

    if (E->model) { GRBfreemodel(E->model); E->model = NULL; }
    if (E->env)   { GRBfreeenv(E->env);     E->env = NULL; }

    CHECK(GRBloadenv(&E->env, NULL));
    if (!verbose) GRBsetintparam(E->env, "OutputFlag", 0);

    CHECK(GRBnewmodel(E->env, &E->model, "RestrictedMP", 0, NULL, NULL, NULL, NULL, NULL));
    CHECK(GRBsetintattr(E->model, "ModelSense", GRB_MAXIMIZE));
    GRBenv *menv = GRBgetenv(E->model);
    GRBsetintparam(menv, "OutputFlag", verbose > 0);
    GRBsetintparam(menv, "UpdateMode", 1);

    /* Batch all constraints in one call. */
    int nc = X*Y + I + J + K;
    char   *sense = (char   *)malloc((size_t)nc);
    double *rhs   = (double *)malloc((size_t)nc * sizeof(double));
    for (int p = 0; p < nc; ++p) sense[p] = GRB_EQUAL;
    for (int p = 0; p < X*Y; ++p)            rhs[p] = 0.0;
    for (int p = X*Y; p < X*Y + I + J; ++p)  rhs[p] = 1.0;
    for (int k = 0; k < K; ++k) {
        double r = 0.0;
        for (int x = 0; x < X; ++x)
            for (int y = 0; y < Y; ++y)
                r += (double)E->mu_x_y[x*Y + y] * E->phi_x_y_k[(x*Y + y)*K + k];
        rhs[X*Y + I + J + k] = r;
    }
    CHECK(GRBaddconstrs(E->model, nc, 0, NULL, NULL, NULL, sense, rhs, NULL));
    free(sense); free(rhs);

    /* Count active vars across all four classes. */
    int n_i0_act = 0, n_iy_act = 0, n_0j_act = 0, n_xj_act = 0;
    for (int i = 0; i < I; ++i) if (E->active_i_0[i]) ++n_i0_act;
    for (int j = 0; j < J; ++j) if (E->active_0_j[j]) ++n_0j_act;
    for (int p = 0; p < I*Y; ++p) if (E->Y_i_y[p]) ++n_iy_act;
    for (int p = 0; p < J*X; ++p) if (E->X_j_x[p]) ++n_xj_act;

    int total = n_i0_act + n_iy_act + n_0j_act + n_xj_act;
    int total_nz = n_i0_act + (2 + K)*n_iy_act + n_0j_act + (2 + K)*n_xj_act;

    int    *vbeg = (int    *)malloc((size_t)total    * sizeof(int));
    int    *vind = (int    *)malloc((size_t)total_nz * sizeof(int));
    double *vval = (double *)malloc((size_t)total_nz * sizeof(double));
    double *vobj = (double *)malloc((size_t)total    * sizeof(double));
    double *vlb  = (double *)calloc((size_t)total, sizeof(double));

    int q = 0, p = 0;
    /* lambda_i_0 */
    for (int i = 0; i < I; ++i) if (E->active_i_0[i]) {
        vbeg[q] = p;
        vind[p] = X*Y + i;          vval[p++] = 1.0;
        vobj[q++] = E->eps_i_0[i];
    }
    /* lambda_i_y */
    for (int i = 0; i < I; ++i)
        for (int y = 0; y < Y; ++y) if (E->Y_i_y[i*Y + y]) {
            int x = E->x_i[i];
            vbeg[q] = p;
            vind[p] = x*Y + y;      vval[p++] = 1.0;
            vind[p] = X*Y + i;      vval[p++] = 1.0;
            for (int k = 0; k < K; ++k) {
                vind[p]    = X*Y + I + J + k;
                vval[p++]  = 0.5 * E->phi_x_y_k[(x*Y + y)*K + k];
            }
            vobj[q++] = E->eps_i_y[i*Y + y];
        }
    /* lambda_0_j */
    for (int j = 0; j < J; ++j) if (E->active_0_j[j]) {
        vbeg[q] = p;
        vind[p] = X*Y + I + j;      vval[p++] = 1.0;
        vobj[q++] = E->eta_0_j[j];
    }
    /* lambda_x_j */
    for (int j = 0; j < J; ++j)
        for (int x = 0; x < X; ++x) if (E->X_j_x[j*X + x]) {
            int y = E->y_j[j];
            vbeg[q] = p;
            vind[p] = x*Y + y;      vval[p++] = -1.0;
            vind[p] = X*Y + I + j;  vval[p++] =  1.0;
            for (int k = 0; k < K; ++k) {
                vind[p]    = X*Y + I + J + k;
                vval[p++]  = 0.5 * E->phi_x_y_k[(x*Y + y)*K + k];
            }
            vobj[q++] = E->eta_x_j[x*J + j];
        }

    CHECK(GRBaddvars(E->model, total, total_nz, vbeg, vind, vval,
                     vobj, vlb, NULL, NULL, NULL));
    free(vbeg); free(vind); free(vval); free(vobj); free(vlb);
    return 0;
}

/* Estimation pricing: parallel scan over i and j.  Singlehood option wins
   over best matched type when its reduced cost is larger (Python logic). */
static int tu_estimation_find_rc(TUMatchingEstimation *E, double rc_tol,
                                 int **new_iy, int *n_iy_new,
                                 int **new_i0, int *n_i0_new,
                                 int **new_jx, int *n_jx_new,
                                 int **new_0j, int *n_0j_new)
{
    int I = E->I, J = E->J, X = E->X, Y = E->Y, K = E->K;
    GRBmodel *m = E->model;

    /* All constraint duals in one Pi call. */
    int n_constrs = X*Y + I + J + K;
    double *all_pi = xcalloc_d((size_t)n_constrs);
    CHECK(GRBgetdblattrarray(m, "Pi", 0, n_constrs, all_pi));
    const double *t_xy = all_pi;
    const double *u_i  = all_pi + X*Y;
    const double *v_j  = all_pi + X*Y + I;
    double *lam = xcalloc_d((size_t)K);
    for (int k = 0; k < K; ++k) lam[k] = -all_pi[X*Y + I + J + k];

    double *Phi_lambda = xcalloc_d((size_t)X*Y);
#pragma omp parallel for if(X*Y > 64) schedule(static)
    for (int xy = 0; xy < X*Y; ++xy) {
        double s = 0.0;
        const double *phi = E->phi_x_y_k + (size_t)xy*K;
        for (int k = 0; k < K; ++k) s += phi[k] * lam[k];
        Phi_lambda[xy] = s;
    }

    /* Per-i: -2 == add singlehood, -1 == nothing, >=0 == add y. */
    int *decision_i = xfilled_i((size_t)I, -1);
#pragma omp parallel for if(I > 64) schedule(static)
    for (int i = 0; i < I; ++i) {
        int x = E->x_i[i];
        double u = u_i[i];
        int by = -1; double br = -1e300;
        const char   *mask    = E->Y_i_y    + (size_t)i*Y;
        const double *eps_row = E->eps_i_y  + (size_t)i*Y;
        const double *Pl_row  = Phi_lambda  + (size_t)x*Y;
        const double *t_row   = t_xy        + (size_t)x*Y;
        for (int y = 0; y < Y; ++y) {
            if (mask[y]) continue;
            double rc = 0.5 * Pl_row[y] - t_row[y] + eps_row[y] - u;
            if (rc > br) { br = rc; by = y; }
        }
        double rc_i0 = E->eps_i_0[i] - u;
        int can_i0 = (!E->active_i_0[i]) && (rc_i0 > br);
        if (!can_i0 && by >= 0 && br > rc_tol)        decision_i[i] = by;
        else if (can_i0 && rc_i0 > rc_tol)            decision_i[i] = -2;
    }

    int *decision_j = xfilled_i((size_t)J, -1);
#pragma omp parallel for if(J > 64) schedule(static)
    for (int j = 0; j < J; ++j) {
        int y = E->y_j[j];
        double v = v_j[j];
        int bx = -1; double br = -1e300;
        for (int x = 0; x < X; ++x) {
            if (E->X_j_x[j*X + x]) continue;
            double rc = 0.5 * Phi_lambda[x*Y + y] + t_xy[x*Y + y] + E->eta_x_j[x*J + j] - v;
            if (rc > br) { br = rc; bx = x; }
        }
        double rc_0j = E->eta_0_j[j] - v;
        int can_0j = (!E->active_0_j[j]) && (rc_0j > br);
        if (!can_0j && bx >= 0 && br > rc_tol)        decision_j[j] = bx;
        else if (can_0j && rc_0j > rc_tol)            decision_j[j] = -2;
    }

    int cap_iy=16,n_iy=0; int *iy = (int *)malloc((size_t)cap_iy*2*sizeof(int));
    int cap_i0=16,n_i0=0; int *i0 = (int *)malloc((size_t)cap_i0*sizeof(int));
    for (int i = 0; i < I; ++i) {
        int d = decision_i[i];
        if (d >= 0) {
            E->Y_i_y[i*Y + d] = 1;
            if (n_iy + 1 > cap_iy) { cap_iy *= 2; iy = (int *)realloc(iy, (size_t)cap_iy*2*sizeof(int)); }
            iy[2*n_iy + 0] = i; iy[2*n_iy + 1] = d; ++n_iy;
        } else if (d == -2) {
            E->active_i_0[i] = 1;
            if (n_i0 + 1 > cap_i0) { cap_i0 *= 2; i0 = (int *)realloc(i0, (size_t)cap_i0*sizeof(int)); }
            i0[n_i0++] = i;
        }
    }
    int cap_jx=16,n_jx=0; int *jx = (int *)malloc((size_t)cap_jx*2*sizeof(int));
    int cap_0j=16,n_0j=0; int *zj = (int *)malloc((size_t)cap_0j*sizeof(int));
    for (int j = 0; j < J; ++j) {
        int d = decision_j[j];
        if (d >= 0) {
            E->X_j_x[j*X + d] = 1;
            if (n_jx + 1 > cap_jx) { cap_jx *= 2; jx = (int *)realloc(jx, (size_t)cap_jx*2*sizeof(int)); }
            jx[2*n_jx + 0] = j; jx[2*n_jx + 1] = d; ++n_jx;
        } else if (d == -2) {
            E->active_0_j[j] = 1;
            if (n_0j + 1 > cap_0j) { cap_0j *= 2; zj = (int *)realloc(zj, (size_t)cap_0j*sizeof(int)); }
            zj[n_0j++] = j;
        }
    }

    *new_iy = iy; *n_iy_new = n_iy;
    *new_i0 = i0; *n_i0_new = n_i0;
    *new_jx = jx; *n_jx_new = n_jx;
    *new_0j = zj; *n_0j_new = n_0j;

    free(all_pi); free(lam); free(Phi_lambda);
    free(decision_i); free(decision_j);
    return 0;
}

static int tu_estimation_update_rmp(TUMatchingEstimation *E,
                                    int *iy, int n_iy, int *i0, int n_i0,
                                    int *jx, int n_jx, int *zj, int n_0j)
{
    int I = E->I, J = E->J, X = E->X, Y = E->Y, K = E->K;
    int total = n_i0 + n_iy + n_0j + n_jx;
    if (total == 0) return 0;

    int total_nz = n_i0 + (2 + K)*n_iy + n_0j + (2 + K)*n_jx;
    int    *vbeg = (int    *)malloc((size_t)total    * sizeof(int));
    int    *vind = (int    *)malloc((size_t)total_nz * sizeof(int));
    double *vval = (double *)malloc((size_t)total_nz * sizeof(double));
    double *vobj = (double *)malloc((size_t)total    * sizeof(double));
    double *vlb  = (double *)calloc((size_t)total, sizeof(double));

    int q = 0, p = 0;
    for (int u = 0; u < n_i0; ++u) {
        int i = i0[u];
        vbeg[q] = p;
        vind[p] = X*Y + i;          vval[p++] = 1.0;
        vobj[q++] = E->eps_i_0[i];
    }
    for (int u = 0; u < n_iy; ++u) {
        int i = iy[2*u + 0], y = iy[2*u + 1];
        int x = E->x_i[i];
        vbeg[q] = p;
        vind[p] = x*Y + y;          vval[p++] = 1.0;
        vind[p] = X*Y + i;           vval[p++] = 1.0;
        for (int k = 0; k < K; ++k) {
            vind[p]    = X*Y + I + J + k;
            vval[p++]  = 0.5 * E->phi_x_y_k[(x*Y + y)*K + k];
        }
        vobj[q++] = E->eps_i_y[i*Y + y];
    }
    for (int u = 0; u < n_0j; ++u) {
        int j = zj[u];
        vbeg[q] = p;
        vind[p] = X*Y + I + j;      vval[p++] = 1.0;
        vobj[q++] = E->eta_0_j[j];
    }
    for (int u = 0; u < n_jx; ++u) {
        int j = jx[2*u + 0], x = jx[2*u + 1];
        int y = E->y_j[j];
        vbeg[q] = p;
        vind[p] = x*Y + y;          vval[p++] = -1.0;
        vind[p] = X*Y + I + j;      vval[p++] =  1.0;
        for (int k = 0; k < K; ++k) {
            vind[p]    = X*Y + I + J + k;
            vval[p++]  = 0.5 * E->phi_x_y_k[(x*Y + y)*K + k];
        }
        vobj[q++] = E->eta_x_j[x*J + j];
    }

    CHECK(GRBaddvars(E->model, total, total_nz, vbeg, vind, vval,
                     vobj, vlb, NULL, NULL, NULL));
    free(vbeg); free(vind); free(vval); free(vobj); free(vlb);
    return 0;
}

int tu_estimation_rroa_solve(TUMatchingEstimation *E, int max_iter, double rc_tol,
                             int verbose, double *lambda_k_out, double *objval_out)
{
    int K = E->K;

    tu_estimation_basic_feasible_solution(E);
    CHECK(tu_estimation_build_rmp(E, verbose));

    GRBenv *menv = GRBgetenv(E->model);
    GRBsetintparam(menv, "OutputFlag", 0);
    CHECK(GRBoptimize(E->model));

    int status; GRBgetintattr(E->model, "Status", &status);
    if (status != GRB_OPTIMAL) {
        fprintf(stderr, "[dw est rroa] initial status %d\n", status); return -1;
    }
    GRBsetintparam(menv, "Presolve",    0);
    GRBsetintparam(menv, "LPWarmStart", 2);
    /* Concurrent solver beats primal/dual/barrier alone for warm-started
       column-generation re-solves on this class of LPs. */
    GRBsetintparam(menv, "Method",      3);

    for (int it = 0; it < max_iter; ++it) {
        int *iy=NULL,*i0=NULL,*jx=NULL,*zj=NULL;
        int n_iy=0,n_i0=0,n_jx=0,n_0j=0;
        CHECK(tu_estimation_find_rc(E, rc_tol, &iy,&n_iy, &i0,&n_i0, &jx,&n_jx, &zj,&n_0j));
        if (n_iy + n_i0 + n_jx + n_0j == 0) {
            free(iy); free(i0); free(jx); free(zj);
            if (verbose > 0) fprintf(stderr, "Iter %2d: optimal.\n", it+1);
            break;
        }
        CHECK(tu_estimation_update_rmp(E, iy,n_iy, i0,n_i0, jx,n_jx, zj,n_0j));
        free(iy); free(i0); free(jx); free(zj);
        CHECK(GRBoptimize(E->model));
    }

    if (objval_out) GRBgetdblattr(E->model, "ObjVal", objval_out);
    if (lambda_k_out) {
        double *raw = xcalloc_d((size_t)K);
        CHECK(GRBgetdblattrarray(E->model, "Pi", E->X*E->Y + E->I + E->J, K, raw));
        for (int k = 0; k < K; ++k) lambda_k_out[k] = -raw[k];
        free(raw);
    }
    return 0;
}
