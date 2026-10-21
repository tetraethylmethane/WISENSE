#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "driver/gpio.h"
#include "esp_log.h"

#include "reset_button.h"
#include "../tasks/wifi_app.h"

static const char TAG[] = "Reset Button";

//Binary semaphore for the button task
SemaphoreHandle_t reset_semaphore = NULL; 

/*
 * ISR handler for the reset button
 */
void IRAM_ATTR reset_button_isr_handler (void * parameters) {
    //Notify the reset button task by increasing the semaphore 
    xSemaphoreGiveFromISR(reset_semaphore, NULL); 
}

/*
 * Wi-Fi reset button task that reacts to a BOOT button event by sending a message
 * to the Wi-Fi task for changing to AP mode and erase the NVS storage
 */
void reset_button_task (void * parameters) {
    while(1) {
        //Block until semaphore is released
        if (xSemaphoreTake(reset_semaphore, portMAX_DELAY) == pdTRUE) {
            /* WISENSE patch: the BOOT button shares GPIO0 with the USB-serial
             * auto-reset circuit, so opening the serial port (or plugging the
             * cable) can produce a falling edge -- which used to erase the
             * saved configuration instantly. Require a deliberate 3 s hold. */
            int held_ms = 0;
            while (gpio_get_level(WIFI_RESET_BUTTON) == 0 && held_ms < 3000) {
                vTaskDelay(50 / portTICK_PERIOD_MS);
                held_ms += 50;
            }
            if (held_ms < 3000) {
                ESP_LOGI(TAG, "GPIO0 low for %d ms -- ignored (hold BOOT 3 s to reset)", held_ms);
                continue;
            }
            ESP_LOGI(TAG, "starting ISR");
            //Send a message to the Wi-Fi app to start as access point
            wifi_app_message_t n_msg = {
                .msgID = MSG_RESET_WIFI,
                .data = NULL
            };
            wifi_app_send_message(n_msg); 
            //Add a delay to this task so it gives time before a further push event
            vTaskDelay(3000 / portTICK_PERIOD_MS);
        }
    }
}

void reset_button_config(void) {
    //Create the binary semaphore
    reset_semaphore = xSemaphoreCreateBinary(); 

    gpio_set_direction(WIFI_RESET_BUTTON, GPIO_MODE_INPUT); 

    //Enable input on negative edge
    gpio_set_intr_type(WIFI_RESET_BUTTON, GPIO_INTR_NEGEDGE);

    //Create reset button task
    xTaskCreatePinnedToCore(&reset_button_task, "reset_task", RESET_BUTTON_TASK_STACK_SIZE, NULL, RESET_BUTTON_TASK_PRIORITY, NULL, tskNO_AFFINITY);

    //Install the ISR and attach it to the gpio
    gpio_install_isr_service(ESP_INTR_FLAG_DEFAULT);
    gpio_isr_handler_add(WIFI_RESET_BUTTON, reset_button_isr_handler, NULL);
}