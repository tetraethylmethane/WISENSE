/* csi_frame.c -- see csi_frame.h. */

#include "csi_frame.h"

#include <string.h>

static const char MARKER[] = "CSI_DATA,";
#define MARKER_LEN ((uint16_t)(sizeof MARKER - 1u))

enum { ST_SYNC, ST_LEN, ST_DATA };

void csi_frame_init(csi_frame_parser_t *p)
{
    memset(p, 0, sizeof *p);
}

bool csi_frame_feed(csi_frame_parser_t *p, uint8_t byte)
{
    switch (p->state) {
    case ST_SYNC:
        if (byte == (uint8_t)MARKER[p->pos]) {
            if (++p->pos == MARKER_LEN) {
                p->state = ST_LEN;
                p->pos = 0;
                p->len = 0;
            }
        } else {
            /* The marker's only repeated prefix is "C", so restart cheaply. */
            p->pos = (byte == (uint8_t)MARKER[0]) ? 1u : 0u;
        }
        return false;

    case ST_LEN:
        p->len = (uint16_t)(p->len | ((uint16_t)byte << (8u * p->pos)));
        if (++p->pos == 2u) {
            if (p->len == 128u || p->len == 256u || p->len == 384u) {
                p->state = ST_DATA;
                p->pos = 0;
            } else {
                p->resyncs++;
                p->state = ST_SYNC;
                p->pos = 0;
            }
        }
        return false;

    case ST_DATA:
        if (p->pos < CSI_RAW_LEN) {
            p->frame[p->pos] = (int8_t)byte;
        }
        if (++p->pos < p->len) {
            return false;
        }
        p->state = ST_SYNC;
        p->pos = 0;
        if (p->len == CSI_RAW_LEN) {
            p->delivered++;
            return true;
        }
        p->skipped++;
        return false;

    default:
        csi_frame_init(p);
        return false;
    }
}
