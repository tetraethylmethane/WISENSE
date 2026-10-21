/* csi_dsp.c -- see csi_dsp.h. Transcribed from wisense/csi.py. */

#include "csi_dsp.h"

#include <math.h>

/* wisense/csi.py KEPT_SUBCARRIERS: 1..26 and 38..63. */
uint32_t csi_kept_to_raw(uint32_t k)
{
    return (k < 26u) ? k + 1u : k + 12u;
}

/* wisense/csi.py PROBLEM_KEPT = (0, 26, 27, 28, 29). */
bool csi_kept_is_good(uint32_t k)
{
    return k != 0u && (k < 26u || k > 29u);
}

void csi_amplitude(const int8_t raw[CSI_RAW_LEN], float out[CSI_N_KEPT])
{
    for (uint32_t k = 0; k < CSI_N_KEPT; k++) {
        const uint32_t sc = csi_kept_to_raw(k);
        const float im = (float)raw[2u * sc];        /* Espressif order: imag first */
        const float re = (float)raw[2u * sc + 1u];
        out[k] = sqrtf(im * im + re * re);
    }
}

void csi_window_init(csi_window_t *w, float *storage)
{
    w->amp = storage;
    w->head = 0;
    w->count = 0;
}

void csi_window_push(csi_window_t *w, const int8_t raw[CSI_RAW_LEN])
{
    csi_amplitude(raw, &w->amp[w->head * CSI_N_KEPT]);
    w->head = (w->head + 1u) % CSI_TIMESTEPS;
    if (w->count < CSI_TIMESTEPS) {
        w->count++;
    }
}

bool csi_window_full(const csi_window_t *w)
{
    return w->count == CSI_TIMESTEPS;
}

/* Population variance of the min-max-normalised column, as
 * minmax(x).var(axis=0) computes it. A flat column normalises to all-zero and
 * so has variance 0. Order of rows does not matter, so the ring is read as is. */
static double normalised_variance(const csi_window_t *w, uint32_t k)
{
    float mn = w->amp[k], mx = w->amp[k];
    for (uint32_t t = 1; t < CSI_TIMESTEPS; t++) {
        const float v = w->amp[t * CSI_N_KEPT + k];
        mn = v < mn ? v : mn;
        mx = v > mx ? v : mx;
    }
    if (mx == mn) {
        return 0.0;
    }
    const double rng = (double)mx - (double)mn;
    double mean = 0.0;
    for (uint32_t t = 0; t < CSI_TIMESTEPS; t++) {
        mean += ((double)w->amp[t * CSI_N_KEPT + k] - mn) / rng;
    }
    mean /= CSI_TIMESTEPS;
    double var = 0.0;
    for (uint32_t t = 0; t < CSI_TIMESTEPS; t++) {
        const double d = ((double)w->amp[t * CSI_N_KEPT + k] - mn) / rng - mean;
        var += d * d;
    }
    return var / CSI_TIMESTEPS;
}

uint32_t csi_select_subcarrier(const csi_window_t *w)
{
    uint32_t best = 1;   /* lowest good index */
    double best_var = normalised_variance(w, best);
    for (uint32_t k = best + 1u; k < CSI_N_KEPT; k++) {
        if (!csi_kept_is_good(k)) {
            continue;
        }
        const double v = normalised_variance(w, k);
        if (v > best_var) {   /* strict: ties keep the lower index */
            best_var = v;
            best = k;
        }
    }
    return best;
}

void csi_extract(const csi_window_t *w, uint32_t kept_sc, float out[CSI_TIMESTEPS])
{
    /* Oldest row is at head once the ring is full. */
    for (uint32_t t = 0; t < CSI_TIMESTEPS; t++) {
        const uint32_t row = (w->head + t) % CSI_TIMESTEPS;
        out[t] = w->amp[row * CSI_N_KEPT + kept_sc];
    }
}

void csi_quantise(const float *x, size_t n, float scale, int32_t zero_point, uint8_t *out)
{
    for (size_t i = 0; i < n; i++) {
        const float q = nearbyintf(x[i] / scale) + (float)zero_point;
        out[i] = (uint8_t)(q < 0.0f ? 0.0f : (q > 255.0f ? 255.0f : q));
    }
}
