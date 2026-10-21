/* test_csi_dsp.c -- csi_dsp.c against the frozen host reference. */

#include "../wisense/csi_dsp.h"
#include "csi_reference.h"
#include "harness.h"

static float g_storage[CSI_TIMESTEPS * CSI_N_KEPT];

static void test_kept_map(void)
{
    CHECK_EQ(csi_kept_to_raw(0), 1);
    CHECK_EQ(csi_kept_to_raw(25), 26);
    CHECK_EQ(csi_kept_to_raw(26), 38);    /* guard band 27..37 skipped */
    CHECK_EQ(csi_kept_to_raw(51), 63);
    int good = 0;
    for (uint32_t k = 0; k < CSI_N_KEPT; k++) {
        good += csi_kept_is_good(k);
    }
    CHECK_EQ(good, 47);
    CHECK(!csi_kept_is_good(0) && !csi_kept_is_good(26) && !csi_kept_is_good(29));
    CHECK(csi_kept_is_good(1) && csi_kept_is_good(25) && csi_kept_is_good(30));
}

static void test_amplitude_matches_host(void)
{
    float amp[CSI_N_KEPT];
    csi_amplitude(ref_raw[0], amp);
    for (uint32_t k = 0; k < CSI_N_KEPT; k++) {
        CHECK_NEAR(amp[k], ref_amp0[k], 1e-4);
    }
}

static void test_amplitude_is_imag_real_magnitude(void)
{
    int8_t raw[CSI_RAW_LEN] = {0};
    raw[2] = 3;   /* subcarrier 1: imag 3, real -4 -> 5 */
    raw[3] = -4;
    raw[2 * 63] = -128;  /* extreme value does not overflow */
    float amp[CSI_N_KEPT];
    csi_amplitude(raw, amp);
    CHECK_NEAR(amp[0], 5.0, 1e-6);
    CHECK_NEAR(amp[51], 128.0, 1e-6);
    CHECK_NEAR(amp[1], 0.0, 1e-6);
}

static void test_window_fill_and_wrap(void)
{
    csi_window_t w;
    csi_window_init(&w, g_storage);
    for (uint32_t i = 0; i < CSI_TIMESTEPS - 1; i++) {
        csi_window_push(&w, ref_raw[i]);
    }
    CHECK(!csi_window_full(&w));
    csi_window_push(&w, ref_raw[CSI_TIMESTEPS - 1]);
    CHECK(csi_window_full(&w));
    for (uint32_t i = CSI_TIMESTEPS; i < REF_N_PUSH; i++) {
        csi_window_push(&w, ref_raw[i]);
    }
    CHECK(csi_window_full(&w));
    CHECK_EQ(w.count, CSI_TIMESTEPS);
}

static void test_selection_and_series_match_host(void)
{
    csi_window_t w;
    csi_window_init(&w, g_storage);
    for (uint32_t i = 0; i < REF_N_PUSH; i++) {
        csi_window_push(&w, ref_raw[i]);
    }
    const uint32_t sc = csi_select_subcarrier(&w);
    CHECK_EQ(sc, REF_SELECTED_SC);

    static float series[CSI_TIMESTEPS];
    static uint8_t q[CSI_TIMESTEPS];
    csi_extract(&w, sc, series);
    int mismatches = 0;
    for (uint32_t t = 0; t < CSI_TIMESTEPS; t++) {
        mismatches += fabs((double)series[t] - (double)ref_series[t]) > 1e-4;
    }
    CHECK_EQ(mismatches, 0);   /* order matters: oldest first, after the wrap */

    csi_quantise(series, CSI_TIMESTEPS, REF_SCALE, REF_ZERO, q);
    int qdiff = 0;
    for (uint32_t t = 0; t < CSI_TIMESTEPS; t++) {
        qdiff += q[t] != ref_quant[t];
    }
    CHECK_EQ(qdiff, 0);
}

static void test_problematic_subcarrier_never_selected(void)
{
    /* Raw 38 (kept 26) alternates every packet: normalised variance 0.25, the
     * maximum possible. Raw 5 (kept 4) steps once in ten: variance 0.09. Without
     * the exclusion kept 26 would win; with it, kept 4 must. */
    static int8_t raw[CSI_RAW_LEN];
    csi_window_t w;
    csi_window_init(&w, g_storage);
    for (uint32_t i = 0; i < CSI_TIMESTEPS; i++) {
        memset(raw, 7, sizeof raw);
        raw[2 * 38] = (int8_t)(i % 2u ? 100 : 0);
        raw[2 * 5] = (int8_t)(i % 10u ? 7 : 8);
        csi_window_push(&w, raw);
    }
    CHECK_EQ(csi_select_subcarrier(&w), 4);
}

static void test_flat_window_selects_lowest_index(void)
{
    static int8_t raw[CSI_RAW_LEN];
    memset(raw, 7, sizeof raw);
    csi_window_t w;
    csi_window_init(&w, g_storage);
    for (uint32_t i = 0; i < CSI_TIMESTEPS; i++) {
        csi_window_push(&w, raw);
    }
    CHECK_EQ(csi_select_subcarrier(&w), 1);   /* all variances 0: tie -> lowest good */
}

static void test_quantise_clamps_and_rounds_half_even(void)
{
    const float x[5] = {-10.0f, 0.0625f, 0.1875f, 100.0f, 0.0f};
    uint8_t q[5];
    csi_quantise(x, 5, 0.125f, 0, q);
    CHECK_EQ(q[0], 0);      /* clamp low */
    CHECK_EQ(q[1], 0);      /* 0.5 -> 0 (half to even) */
    CHECK_EQ(q[2], 2);      /* 1.5 -> 2 */
    CHECK_EQ(q[3], 255);    /* clamp high */
    CHECK_EQ(q[4], 0);
}

int main(void)
{
    RUN(test_kept_map);
    RUN(test_amplitude_matches_host);
    RUN(test_amplitude_is_imag_real_magnitude);
    RUN(test_window_fill_and_wrap);
    RUN(test_selection_and_series_match_host);
    RUN(test_problematic_subcarrier_never_selected);
    RUN(test_flat_window_selects_lowest_index);
    RUN(test_quantise_clamps_and_rounds_half_even);
    return harness_report("csi_dsp");
}
