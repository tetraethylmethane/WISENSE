/* csi_frame.h -- byte-at-a-time parser for the collecting tool's binary stream.
 *
 * Port of host/csi_capture.py BinaryParser for the configuration the S3
 * firmware expects: in the tool's web form tick only "CSI length" and
 * "CSI data", binary data mode, informer mode "Serial communication". Each
 * frame on the wire is then:
 *
 *     "CSI_DATA,"  len:u16 little-endian  len bytes of int8 CSI
 *
 * Frames whose length is not 128 (HT-LTF present) are consumed and counted
 * but not delivered -- the model was trained on LLTF-only frames. A length
 * outside {128, 256, 384} means we synced on a false marker inside payload;
 * the parser drops back to searching.
 */

#ifndef WISENSE_CSI_FRAME_H
#define WISENSE_CSI_FRAME_H

#include <stdbool.h>
#include <stdint.h>

#include "csi_dsp.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t  state;
    uint16_t pos;        /* bytes matched/collected in the current state */
    uint16_t len;
    int8_t   frame[CSI_RAW_LEN];
    uint32_t delivered;
    uint32_t skipped;    /* well-formed but not 128 bytes */
    uint32_t resyncs;    /* implausible length after a marker */
} csi_frame_parser_t;

void csi_frame_init(csi_frame_parser_t *p);

/* Feed one byte. Returns true when p->frame holds a complete 128-byte frame. */
bool csi_frame_feed(csi_frame_parser_t *p, uint8_t byte);

#ifdef __cplusplus
}
#endif

#endif /* WISENSE_CSI_FRAME_H */
