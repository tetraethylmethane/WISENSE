/* csi_dsp.h -- raw ESP32 CSI frames -> the model's uint8 input.
 *
 * Device port of wisense/csi.py + wisense/features.py (DEFAULT mode). No
 * ESP-IDF, no allocation: the caller owns every buffer, so this compiles and
 * is tested on a laptop (`mingw32-make -C firmware test`) and builds
 * unmodified into firmware/esp32s3_har.
 *
 * Pipeline, per received packet:
 *   1. csi_window_push(): 128-byte LLTF buffer -> 52 kept-subcarrier
 *      amplitudes, appended to an 850-row ring.
 * Once full, per classification:
 *   2. csi_select_subcarrier(): of the 47 good subcarriers, the one with the
 *      largest variance of its min-max-normalised amplitude over the window
 *      (ties -> lower index). Raw 1 and 38..41 are never chosen: they are
 *      mostly noise, so they win on variance and lose on accuracy -- see
 *      PROBLEM_KEPT in wisense/csi.py.
 *   3. csi_extract(): that subcarrier's 850 amplitudes, oldest first.
 *   4. csi_quantise(): into the model's uint8 input domain.
 *
 * If host and device ever disagree on step 2, they classify different signals
 * while both look healthy -- tests/test_csi_dsp.c checks against a frozen
 * host reference for exactly that reason.
 */

#ifndef WISENSE_CSI_DSP_H
#define WISENSE_CSI_DSP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CSI_RAW_LEN     128u   /* 64 subcarriers x (imag, real) int8 */
#define CSI_N_SC        64u
#define CSI_N_KEPT      52u    /* minus DC null (0) and guard band (27..37) */
#define CSI_TIMESTEPS   850u

/* Bytes the caller must provide for one window. 176,800 -- PSRAM on the S3. */
#define CSI_WINDOW_BYTES (CSI_TIMESTEPS * CSI_N_KEPT * sizeof(float))

typedef struct {
    float   *amp;     /* [CSI_TIMESTEPS][CSI_N_KEPT], caller-owned */
    uint32_t head;    /* row the next push writes */
    uint32_t count;   /* rows filled, saturates at CSI_TIMESTEPS */
} csi_window_t;

/* Kept subcarrier k (0..51) -> raw subcarrier index (1..63). */
uint32_t csi_kept_to_raw(uint32_t k);

/* False for the five problematic kept subcarriers (raw 1, 38..41). */
bool csi_kept_is_good(uint32_t k);

void csi_amplitude(const int8_t raw[CSI_RAW_LEN], float out[CSI_N_KEPT]);

void csi_window_init(csi_window_t *w, float *storage);
void csi_window_push(csi_window_t *w, const int8_t raw[CSI_RAW_LEN]);
bool csi_window_full(const csi_window_t *w);

/* Returns a kept-subcarrier index 0..51, never a problematic one. Window must be full. */
uint32_t csi_select_subcarrier(const csi_window_t *w);

void csi_extract(const csi_window_t *w, uint32_t kept_sc, float out[CSI_TIMESTEPS]);

/* q = clamp(round_half_even(x / scale + zero_point), 0, 255), as TFLite does. */
void csi_quantise(const float *x, size_t n, float scale, int32_t zero_point, uint8_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WISENSE_CSI_DSP_H */
