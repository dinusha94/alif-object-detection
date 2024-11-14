/* Copyright (C) 2024 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */
#ifndef OSPI_FLASH_H_
#define OSPI_FLASH_H_

#include <stdint.h>
#include "Driver_Flash.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Use this in my custom flash driver */
extern ARM_DRIVER_FLASH* const ptrDrvFlash;

/**
 * @brief Initialize OSPI driver and flash chip. Enables XIP mode.
 *
 */
int32_t ospi_flash_init(void);

/**
 * @brief Initialize OSPI driver and flash chip. program data mode.
 *
 */
// int32_t ospi_flash_send(void);
int32_t ospi_flash_send(void);

/**
 * @brief Initialize OSPI driver and flash chip. read data mode.
 *
 */
int32_t ospi_flash_read(void);

#ifdef __cplusplus
}
#endif
#endif
