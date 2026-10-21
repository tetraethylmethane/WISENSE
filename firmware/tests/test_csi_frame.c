/* test_csi_frame.c -- the UART frame parser, against hand-built byte streams. */

#include "../wisense/csi_frame.h"
#include "harness.h"

static uint8_t g_stream[4096];
static size_t g_n;

static void put(const void *data, size_t n)
{
    memcpy(&g_stream[g_n], data, n);
    g_n += n;
}

static void put_frame(uint16_t len, int8_t fill)
{
    const uint8_t hdr[2] = {(uint8_t)(len & 0xffu), (uint8_t)(len >> 8)};
    put("CSI_DATA,", 9);
    put(hdr, 2);
    for (uint16_t i = 0; i < len; i++) {
        const int8_t v = (int8_t)(fill + (int8_t)(i % 7u));
        put(&v, 1);
    }
}

static int feed_all(csi_frame_parser_t *p, int8_t *last_first_byte)
{
    int frames = 0;
    for (size_t i = 0; i < g_n; i++) {
        if (csi_frame_feed(p, g_stream[i])) {
            frames++;
            *last_first_byte = p->frame[0];
        }
    }
    return frames;
}

static void test_single_frame(void)
{
    csi_frame_parser_t p;
    int8_t first = 0;
    csi_frame_init(&p);
    g_n = 0;
    put_frame(128, -5);
    CHECK_EQ(feed_all(&p, &first), 1);
    CHECK_EQ(first, -5);
    CHECK_EQ(p.frame[127], (int8_t)(-5 + 127 % 7));
}

static void test_log_noise_between_frames(void)
{
    csi_frame_parser_t p;
    int8_t first = 0;
    csi_frame_init(&p);
    g_n = 0;
    put("I (1234) wifi: CSI_DAT connected\n", 33);   /* partial marker, then noise */
    put_frame(128, 1);
    put("CCCSI_DATA", 10);                              /* restart on repeated 'C' */
    put(",", 1);
    const uint8_t hdr[2] = {128, 0};
    put(hdr, 2);
    for (int i = 0; i < 128; i++) {
        const int8_t v = 9;
        put(&v, 1);
    }
    CHECK_EQ(feed_all(&p, &first), 2);
    CHECK_EQ(first, 9);
    CHECK_EQ(p.resyncs, 0);
}

static void test_ht_frames_skipped_not_delivered(void)
{
    csi_frame_parser_t p;
    int8_t first = 0;
    csi_frame_init(&p);
    g_n = 0;
    put_frame(384, 0);
    put_frame(128, 3);
    CHECK_EQ(feed_all(&p, &first), 1);
    CHECK_EQ(p.skipped, 1);
    CHECK_EQ(first, 3);
}

static void test_false_marker_resyncs(void)
{
    csi_frame_parser_t p;
    int8_t first = 0;
    csi_frame_init(&p);
    g_n = 0;
    put("CSI_DATA,", 9);
    const uint8_t bad[2] = {0x34, 0x12};   /* 0x1234: not a CSI length */
    put(bad, 2);
    put_frame(128, 4);
    CHECK_EQ(feed_all(&p, &first), 1);
    CHECK_EQ(p.resyncs, 1);
    CHECK_EQ(first, 4);
}

int main(void)
{
    RUN(test_single_frame);
    RUN(test_log_noise_between_frames);
    RUN(test_ht_frames_skipped_not_delivered);
    RUN(test_false_marker_resyncs);
    return harness_report("csi_frame");
}
