#include "amber_ble.h"

#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "host/ble_hs.h"
#include "host/ble_hs_id.h"
#include "host/ble_uuid.h"
#include "host/util/util.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "nvs_flash.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

static const char* TAG = "AMBER_BLE";

// These are the only knobs you should need to change when tuning the radio policy.
static const uint32_t kSafeBufferedRecords = 300;           // Operational queue budget before we should wake the radio.
static const uint32_t kFlushStartRecords = (kSafeBufferedRecords * 70) / 100;  // Start draining above 70% of the safe budget.
static const uint32_t kFlushStopRecords = (kSafeBufferedRecords * 10) / 100;   // Stop once we are back near 10%.
static const uint32_t kMaxBufferedAgeMs = 5 * 60 * 1000;    // Hard cap: wake radio at least every 5 minutes.
static const uint32_t kAdvertiseWindowMs = 20 * 1000;       // How long each discovery window stays open.
static const uint32_t kAdvertiseCooldownMs = 10 * 1000;     // Sleep between discovery windows if no base station appears.
static const uint32_t kNotifyIntervalMs = 35;               // Pace BLE drain so it does not starve acquisition.
static const uint32_t kDisconnectGraceMs = 1500;            // Disconnect after flush completes.

#define AMBER_BLE_PAYLOAD_MAX_BYTES 128
#define AMBER_BLE_QUEUE_CAPACITY 1024

typedef struct {
    uint32_t enqueue_ms;
    char payload[AMBER_BLE_PAYLOAD_MAX_BYTES];
} amber_ble_record_t;

static bool s_ble_ready = false;
static bool s_ble_synced = false;
static bool s_advertising = false;
static bool s_notify_enabled = false;
static uint8_t s_own_addr_type = 0;
static uint16_t s_status_value_handle = 0;
static uint16_t s_conn_handle = BLE_HS_CONN_HANDLE_NONE;
static char s_status_payload[AMBER_BLE_PAYLOAD_MAX_BYTES] = "amber_booting";

static amber_ble_record_t* s_queue = NULL;
static size_t s_head = 0;
static size_t s_tail = 0;
static size_t s_count = 0;
static uint32_t s_oldest_enqueue_ms = 0;
static uint32_t s_last_notify_ms = 0;
static uint32_t s_last_disconnect_candidate_ms = 0;
static uint32_t s_advertise_started_ms = 0;
static uint32_t s_next_advertise_ms = 0;
static uint32_t s_dropped_records = 0;
static uint32_t s_sent_records = 0;
static uint32_t s_last_status_log_ms = 0;
static SemaphoreHandle_t s_queue_mutex = NULL;
static TaskHandle_t s_flush_task_handle = NULL;

static const ble_uuid128_t s_service_uuid =
    BLE_UUID128_INIT(0x60, 0x37, 0xb8, 0x1a, 0x84, 0x93, 0x4f, 0x8d,
                     0x86, 0xb5, 0x1d, 0x90, 0x51, 0x5b, 0x2e, 0x01);
static const ble_uuid128_t s_status_uuid =
    BLE_UUID128_INIT(0x60, 0x37, 0xb8, 0x1a, 0x84, 0x93, 0x4f, 0x8d,
                     0x86, 0xb5, 0x1d, 0x90, 0x51, 0x5b, 0x2e, 0x02);

static uint32_t amber_ble_now_ms(void) {
    return (uint32_t)(esp_timer_get_time() / 1000);
}

static void amber_ble_advertise_start(void);
static int amber_ble_gap_event(struct ble_gap_event* event, void* arg);

static bool amber_ble_lock(TickType_t timeout_ticks) {
    return s_queue_mutex != NULL && xSemaphoreTake(s_queue_mutex, timeout_ticks) == pdTRUE;
}

static void amber_ble_unlock(void) {
    if (s_queue_mutex != NULL) {
        xSemaphoreGive(s_queue_mutex);
    }
}

static int amber_ble_status_access(uint16_t conn_handle,
                                   uint16_t attr_handle,
                                   struct ble_gatt_access_ctxt* ctxt,
                                   void* arg)
{
    (void)conn_handle;
    (void)arg;

    if (ctxt->op != BLE_GATT_ACCESS_OP_READ_CHR || attr_handle != s_status_value_handle) {
        return BLE_ATT_ERR_UNLIKELY;
    }

    const int rc = os_mbuf_append(ctxt->om, s_status_payload, strlen(s_status_payload));
    return rc == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
}

static const struct ble_gatt_svc_def s_amber_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &s_service_uuid.u,
        .includes = NULL,
        .characteristics =
            (struct ble_gatt_chr_def[]) {
                {
                    .uuid = &s_status_uuid.u,
                    .access_cb = amber_ble_status_access,
                    .arg = NULL,
                    .descriptors = NULL,
                    .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
                    .min_key_size = 0,
                    .val_handle = &s_status_value_handle,
                    .cpfd = NULL,
                },
                {0},
            },
    },
    {0},
};

static void amber_ble_advertise_stop(void)
{
    if (!s_advertising) {
        return;
    }

    const int rc = ble_gap_adv_stop();
    if (rc != 0 && rc != BLE_HS_EALREADY) {
        ESP_LOGW(TAG, "ble_gap_adv_stop rc=%d", rc);
        return;
    }

    s_advertising = false;
    ESP_LOGI(TAG, "advertising stopped");
}

static void amber_ble_advertise_start(void)
{
    if (!s_ble_synced || s_advertising || s_conn_handle != BLE_HS_CONN_HANDLE_NONE) {
        return;
    }

    struct ble_hs_adv_fields fields;
    memset(&fields, 0, sizeof(fields));
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;

    const char* name = ble_svc_gap_device_name();
    fields.name = (uint8_t*)name;
    fields.name_len = strlen(name);
    fields.name_is_complete = 1;

    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_set_fields failed: rc=%d", rc);
        return;
    }

    struct ble_gap_adv_params adv_params;
    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;

    rc = ble_gap_adv_start(
        s_own_addr_type, NULL, BLE_HS_FOREVER, &adv_params, amber_ble_gap_event, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_start failed: rc=%d", rc);
        return;
    }

    s_advertising = true;
    s_advertise_started_ms = amber_ble_now_ms();
    ESP_LOGI(TAG, "advertising started");
}

static int amber_ble_gap_event(struct ble_gap_event* event, void* arg)
{
    (void)arg;

    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        ESP_LOGI(TAG, "connect status=%d handle=%d", event->connect.status, event->connect.conn_handle);
        if (event->connect.status == 0) {
            s_conn_handle = event->connect.conn_handle;
            s_advertising = false;
            s_last_disconnect_candidate_ms = 0;
        } else {
            s_conn_handle = BLE_HS_CONN_HANDLE_NONE;
            s_notify_enabled = false;
            s_advertising = false;
            s_next_advertise_ms = amber_ble_now_ms() + kAdvertiseCooldownMs;
        }
        return 0;

    case BLE_GAP_EVENT_DISCONNECT:
        ESP_LOGI(TAG, "disconnect reason=%d", event->disconnect.reason);
        s_conn_handle = BLE_HS_CONN_HANDLE_NONE;
        s_notify_enabled = false;
        s_advertising = false;
        s_last_disconnect_candidate_ms = 0;
        s_next_advertise_ms = amber_ble_now_ms() + kAdvertiseCooldownMs;
        return 0;

    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == s_status_value_handle) {
            s_notify_enabled = event->subscribe.cur_notify != 0;
        }
        ESP_LOGI(TAG,
                 "subscribe handle=%d attr=%d notify=%d indicate=%d",
                 event->subscribe.conn_handle,
                 event->subscribe.attr_handle,
                 event->subscribe.cur_notify,
                 event->subscribe.cur_indicate);
        return 0;

    case BLE_GAP_EVENT_ADV_COMPLETE:
        s_advertising = false;
        s_next_advertise_ms = amber_ble_now_ms() + kAdvertiseCooldownMs;
        ESP_LOGI(TAG, "advertising complete reason=%d", event->adv_complete.reason);
        return 0;

    default:
        return 0;
    }
}

static void amber_ble_on_reset(int reason)
{
    ESP_LOGE(TAG, "nimble reset reason=%d", reason);
}

static void amber_ble_on_sync(void)
{
    int rc = ble_hs_util_ensure_addr(0);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_hs_util_ensure_addr failed: rc=%d", rc);
        return;
    }

    rc = ble_hs_id_infer_auto(0, &s_own_addr_type);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_hs_id_infer_auto failed: rc=%d", rc);
        return;
    }

    uint8_t addr_val[6] = {0};
    rc = ble_hs_id_copy_addr(s_own_addr_type, addr_val, NULL);
    if (rc == 0) {
        ESP_LOGI(TAG,
                 "device address %02X:%02X:%02X:%02X:%02X:%02X",
                 addr_val[5],
                 addr_val[4],
                 addr_val[3],
                 addr_val[2],
                 addr_val[1],
                 addr_val[0]);
    }

    s_ble_synced = true;
}

static void amber_ble_host_task(void* param)
{
    (void)param;
    ESP_LOGI(TAG, "NimBLE host task started");
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static void amber_ble_flush_task(void* param)
{
    (void)param;
    ESP_LOGI(TAG, "BLE flush task started");

    while (true) {
        uint32_t now_ms = amber_ble_now_ms();

        bool should_advertise = false;
        bool has_pending = false;
        bool can_drain = false;
        bool should_disconnect = false;
        amber_ble_record_t next_record;
        bool have_record = false;

        if (amber_ble_lock(pdMS_TO_TICKS(5))) {
            has_pending = s_count > 0;
            const uint32_t oldest_age_ms =
                (s_count > 0 && s_oldest_enqueue_ms != 0) ? (now_ms - s_oldest_enqueue_ms) : 0;
            should_advertise =
                has_pending && (s_count >= kFlushStartRecords || oldest_age_ms >= kMaxBufferedAgeMs);

            if (s_conn_handle != BLE_HS_CONN_HANDLE_NONE && s_notify_enabled && s_count > 0 &&
                (now_ms - s_last_notify_ms) >= kNotifyIntervalMs) {
                next_record = s_queue[s_tail];
                s_tail = (s_tail + 1) % AMBER_BLE_QUEUE_CAPACITY;
                s_count--;
                s_oldest_enqueue_ms = (s_count > 0) ? s_queue[s_tail].enqueue_ms : 0;
                s_last_notify_ms = now_ms;
                s_sent_records++;
                have_record = true;
                can_drain = true;
                snprintf(s_status_payload, sizeof(s_status_payload), "%s", next_record.payload);
                if (s_count <= kFlushStopRecords) {
                    if (s_last_disconnect_candidate_ms == 0) {
                        s_last_disconnect_candidate_ms = now_ms;
                    } else if ((now_ms - s_last_disconnect_candidate_ms) >= kDisconnectGraceMs) {
                        should_disconnect = true;
                        s_last_disconnect_candidate_ms = 0;
                    }
                } else {
                    s_last_disconnect_candidate_ms = 0;
                }
            } else if (s_conn_handle != BLE_HS_CONN_HANDLE_NONE && s_count <= kFlushStopRecords) {
                if (s_last_disconnect_candidate_ms == 0) {
                    s_last_disconnect_candidate_ms = now_ms;
                } else if ((now_ms - s_last_disconnect_candidate_ms) >= kDisconnectGraceMs) {
                    should_disconnect = true;
                    s_last_disconnect_candidate_ms = 0;
                }
            } else {
                s_last_disconnect_candidate_ms = 0;
            }
            amber_ble_unlock();
        }

        if (should_advertise && !s_advertising && s_conn_handle == BLE_HS_CONN_HANDLE_NONE &&
            now_ms >= s_next_advertise_ms) {
            amber_ble_advertise_start();
            now_ms = amber_ble_now_ms();
        }

        if (s_advertising && s_conn_handle == BLE_HS_CONN_HANDLE_NONE &&
            (now_ms - s_advertise_started_ms) >= kAdvertiseWindowMs) {
            amber_ble_advertise_stop();
            s_next_advertise_ms = now_ms + kAdvertiseCooldownMs;
        }

        if (can_drain && have_record && s_status_value_handle != 0) {
            ble_gatts_chr_updated(s_status_value_handle);
        }

        if (should_disconnect && s_conn_handle != BLE_HS_CONN_HANDLE_NONE) {
            const int rc = ble_gap_terminate(s_conn_handle, BLE_ERR_REM_USER_CONN_TERM);
            if (rc != 0) {
                ESP_LOGW(TAG, "ble_gap_terminate rc=%d", rc);
            } else {
                ESP_LOGI(TAG, "buffer drained, disconnecting to save power");
            }
        }

        if (has_pending && (now_ms - s_last_status_log_ms) >= 30000) {
            s_last_status_log_ms = now_ms;
            ESP_LOGD(TAG,
                     "buffer status pending=%u sent=%u dropped=%u advertising=%d connected=%d notify=%d",
                     (unsigned)s_count,
                     (unsigned)s_sent_records,
                     (unsigned)s_dropped_records,
                     s_advertising ? 1 : 0,
                     s_conn_handle != BLE_HS_CONN_HANDLE_NONE ? 1 : 0,
                     s_notify_enabled ? 1 : 0);
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

bool amber_ble_init(const char* device_name)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nvs_flash_init failed: %s", esp_err_to_name(ret));
        return false;
    }

    s_queue_mutex = xSemaphoreCreateMutex();
    if (s_queue_mutex == NULL) {
        ESP_LOGE(TAG, "failed to create BLE queue mutex");
        return false;
    }

    s_queue = (amber_ble_record_t*)heap_caps_malloc(
        sizeof(amber_ble_record_t) * AMBER_BLE_QUEUE_CAPACITY, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (s_queue == NULL) {
        s_queue = (amber_ble_record_t*)heap_caps_malloc(
            sizeof(amber_ble_record_t) * AMBER_BLE_QUEUE_CAPACITY, MALLOC_CAP_8BIT);
    }
    if (s_queue == NULL) {
        ESP_LOGE(TAG, "failed to allocate BLE queue");
        return false;
    }
    memset(s_queue, 0, sizeof(amber_ble_record_t) * AMBER_BLE_QUEUE_CAPACITY);

    ret = nimble_port_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nimble_port_init failed: %s", esp_err_to_name(ret));
        return false;
    }

    ble_hs_cfg.reset_cb = amber_ble_on_reset;
    ble_hs_cfg.sync_cb = amber_ble_on_sync;
    ble_hs_cfg.gatts_register_cb = NULL;
    ble_hs_cfg.store_status_cb = NULL;

    ble_svc_gap_init();
    ble_svc_gatt_init();

    int rc = ble_gatts_count_cfg(s_amber_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_count_cfg failed: rc=%d", rc);
        return false;
    }

    rc = ble_gatts_add_svcs(s_amber_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_add_svcs failed: rc=%d", rc);
        return false;
    }

    rc = ble_svc_gap_device_name_set(device_name != NULL ? device_name : "amber-stage1");
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_svc_gap_device_name_set failed: rc=%d", rc);
        return false;
    }

    if (xTaskCreatePinnedToCore(
            amber_ble_flush_task, "amber_ble_flush", 4096, NULL, 2, &s_flush_task_handle, 0) != pdPASS) {
        ESP_LOGE(TAG, "failed to create BLE flush task");
        return false;
    }

    s_ble_ready = true;
    nimble_port_freertos_init(amber_ble_host_task);
    ESP_LOGI(TAG,
             "NimBLE initialized safe=%u start=%u stop=%u max_age_ms=%u queue=%u payload=%u",
             (unsigned)kSafeBufferedRecords,
             (unsigned)kFlushStartRecords,
             (unsigned)kFlushStopRecords,
             (unsigned)kMaxBufferedAgeMs,
             (unsigned)AMBER_BLE_QUEUE_CAPACITY,
             (unsigned)AMBER_BLE_PAYLOAD_MAX_BYTES);
    return true;
}

void amber_ble_publish_status(const char* status_text)
{
    if (!s_ble_ready || status_text == NULL || s_queue == NULL) {
        return;
    }

    const uint32_t now_ms = amber_ble_now_ms();
    if (!amber_ble_lock(pdMS_TO_TICKS(2))) {
        return;
    }

    // Keep the latest payload readable even before the radio wakes up.
    snprintf(s_status_payload, sizeof(s_status_payload), "%s", status_text);

    if (s_count == AMBER_BLE_QUEUE_CAPACITY) {
        s_tail = (s_tail + 1) % AMBER_BLE_QUEUE_CAPACITY;
        s_count--;
        s_dropped_records++;
    }

    amber_ble_record_t* slot = &s_queue[s_head];
    slot->enqueue_ms = now_ms;
    snprintf(slot->payload, sizeof(slot->payload), "%s", status_text);
    s_head = (s_head + 1) % AMBER_BLE_QUEUE_CAPACITY;
    s_count++;
    s_oldest_enqueue_ms = (s_count > 0) ? s_queue[s_tail].enqueue_ms : 0;

    amber_ble_unlock();
}
