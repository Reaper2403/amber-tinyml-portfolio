#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool amber_ble_init(const char* device_name);
void amber_ble_publish_status(const char* status_text);

#ifdef __cplusplus
}
#endif
