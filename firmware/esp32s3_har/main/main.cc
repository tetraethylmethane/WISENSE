// WiSense HAR -- Wi-Fi CSI activity recognition on an ESP32-S3 N16R8.
//
// Reproduces the deployment in Armenta-Garcia et al. (Sensors 2025): the
// uint8 LSTM-DenseNet under TFLite Micro, tensor arena in PSRAM.
//
// Two stages, both configurable in `idf.py menuconfig` -> WiSense HAR:
//
//   1. Benchmark. Classifies held-out-subject recordings compiled into flash
//      (generated/test_vectors.h) and reports accuracy and latency -- the
//      paper's 232 ms / 127 kB figures are checked here, with no radio.
//   2. Live. The paper's two-board layout: an ESP32 running the authors'
//      collecting tool as Rx forwards binary CSI frames over UART; this board
//      windows them (csi_dsp.c), picks the subcarrier, and classifies once per
//      WISENSE_HOP packets.
//
// generated/ is written by `python -m wisense.export <run>`; build after it.

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <new>

#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "csi_dsp.h"
#include "csi_frame.h"
#include "model_data.h"
#include "model_ops.h"
#include "test_vectors.h"

namespace {

const char *TAG = "wisense";
const char *kActivities[] = {"Fall", "Get up", "Lie down", "Sit down", "Walk"};
constexpr int kNumClasses = 5;
constexpr size_t kArenaBytes = CONFIG_WISENSE_ARENA_KB * 1024;

tflite::MicroInterpreter *g_interp = nullptr;
TfLiteTensor *g_input = nullptr;
TfLiteTensor *g_output = nullptr;

bool setup_model()
{
    const tflite::Model *model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "model schema %" PRIu32 " != runtime %d", model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    static ModelOpResolver resolver;
    if (RegisterModelOps(resolver) != kTfLiteOk) {
        ESP_LOGE(TAG, "op registration failed");
        return false;
    }
    auto *arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, kArenaBytes, MALLOC_CAP_SPIRAM));
    if (!arena) {
        ESP_LOGE(TAG, "no PSRAM for a %u-byte arena -- is octal PSRAM enabled?", (unsigned)kArenaBytes);
        return false;
    }
    alignas(tflite::MicroInterpreter) static uint8_t interp_mem[sizeof(tflite::MicroInterpreter)];
    g_interp = new (interp_mem) tflite::MicroInterpreter(model, resolver, arena, kArenaBytes);
    if (g_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed: arena too small, or an op missing from model_ops.h");
        return false;
    }
    g_input = g_interp->input(0);
    g_output = g_interp->output(0);
    if (g_input->type != kTfLiteUInt8 || g_input->bytes != CSI_TIMESTEPS) {
        ESP_LOGE(TAG, "unexpected input tensor (type %d, %u bytes)", g_input->type, (unsigned)g_input->bytes);
        return false;
    }
    ESP_LOGI(TAG, "model %u B in flash, arena used %u B of %u B (PSRAM)",
             (unsigned)g_model_data_len, (unsigned)g_interp->arena_used_bytes(), (unsigned)kArenaBytes);
    return true;
}

// Returns the class index; *ms gets the Invoke() time only, as the paper times it.
int classify(const uint8_t *x, float *ms)
{
    memcpy(g_input->data.uint8, x, CSI_TIMESTEPS);
    const int64_t t0 = esp_timer_get_time();
    if (g_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        return -1;
    }
    *ms = (float)(esp_timer_get_time() - t0) / 1000.0f;
    int best = 0;
    for (int k = 1; k < kNumClasses; k++) {
        if (g_output->data.uint8[k] > g_output->data.uint8[best]) {
            best = k;
        }
    }
    return best;
}

void run_benchmark()
{
    if (g_input->params.scale != kInputScale || g_input->params.zero_point != kInputZeroPoint) {
        ESP_LOGW(TAG, "test vectors were quantised for a different model; regenerate with wisense.export");
    }
    int correct = 0;
    float total = 0, lo = 1e9f, hi = 0;
    for (int i = 0; i < kNumTestVectors; i++) {
        float ms = 0;
        const int p = classify(kTestVectors[i], &ms);
        correct += (p == kTestLabels[i]);
        total += ms;
        lo = ms < lo ? ms : lo;
        hi = ms > hi ? ms : hi;
        printf("bench %2d  truth %-8s  pred %-8s  %7.1f ms\n", i, kActivities[kTestLabels[i]],
               p >= 0 ? kActivities[p] : "ERROR", ms);
        vTaskDelay(1);  // let the idle task feed the watchdog
    }
    printf("BENCH accuracy %d/%d = %.3f   latency mean %.1f ms (min %.1f, max %.1f)   "
           "arena %u B   model %u B\n",
           correct, kNumTestVectors, (double)correct / kNumTestVectors, (double)(total / kNumTestVectors),
           (double)lo, (double)hi, (unsigned)g_interp->arena_used_bytes(), (unsigned)g_model_data_len);
}

void run_live()
{
    constexpr uart_port_t port = UART_NUM_1;
    const uart_config_t cfg = {
        .baud_rate = CONFIG_WISENSE_UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk = UART_SCLK_DEFAULT,
        .flags = {},
    };
    ESP_ERROR_CHECK(uart_driver_install(port, 8192, 0, 0, nullptr, 0));
    ESP_ERROR_CHECK(uart_param_config(port, &cfg));
    ESP_ERROR_CHECK(uart_set_pin(port, UART_PIN_NO_CHANGE, CONFIG_WISENSE_UART_RX_PIN,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    auto *storage = static_cast<float *>(heap_caps_malloc(CSI_WINDOW_BYTES, MALLOC_CAP_SPIRAM));
    static float series[CSI_TIMESTEPS];
    static uint8_t q[CSI_TIMESTEPS];
    static uint8_t rx[512];
    if (!storage) {
        ESP_LOGE(TAG, "no PSRAM for the CSI window");
        return;
    }
    csi_window_t win;
    csi_window_init(&win, storage);
    csi_frame_parser_t parser;
    csi_frame_init(&parser);

    ESP_LOGI(TAG, "live: UART%d RX=GPIO%d @ %d baud; filling %u-packet window...", (int)port,
             CONFIG_WISENSE_UART_RX_PIN, CONFIG_WISENSE_UART_BAUD, (unsigned)CSI_TIMESTEPS);
    uint32_t since = 0;
    int64_t last_report = esp_timer_get_time();
    for (;;) {
        const int n = uart_read_bytes(port, rx, sizeof rx, pdMS_TO_TICKS(100));
        for (int i = 0; i < n; i++) {
            if (!csi_frame_feed(&parser, rx[i])) {
                continue;
            }
            csi_window_push(&win, parser.frame);
            if (!csi_window_full(&win) || ++since < CONFIG_WISENSE_HOP) {
                continue;
            }
            since = 0;
            const int64_t t0 = esp_timer_get_time();
            const uint32_t sc = csi_select_subcarrier(&win);
            csi_extract(&win, sc, series);
            csi_quantise(series, CSI_TIMESTEPS, g_input->params.scale, g_input->params.zero_point, q);
            const float pre_ms = (float)(esp_timer_get_time() - t0) / 1000.0f;
            float ms = 0;
            const int p = classify(q, &ms);
            if (p >= 0) {
                printf("LIVE %-8s  conf %.2f  sc %2u  pre %.1f ms  infer %.1f ms\n", kActivities[p],
                       (double)g_output->data.uint8[p] / 255.0, (unsigned)csi_kept_to_raw(sc),
                       (double)pre_ms, (double)ms);
            }
        }
        if (esp_timer_get_time() - last_report > 5000000) {
            last_report = esp_timer_get_time();
            ESP_LOGI(TAG, "frames %" PRIu32 " delivered, %" PRIu32 " non-LLTF skipped, %" PRIu32
                     " resyncs, window %" PRIu32 "/%u",
                     parser.delivered, parser.skipped, parser.resyncs, win.count, (unsigned)CSI_TIMESTEPS);
        }
    }
}

}  // namespace

extern "C" void app_main(void)
{
    if (!setup_model()) {
        return;
    }
#if CONFIG_WISENSE_RUN_BENCH
    run_benchmark();
#endif
#if CONFIG_WISENSE_RUN_LIVE
    run_live();
#endif
}
