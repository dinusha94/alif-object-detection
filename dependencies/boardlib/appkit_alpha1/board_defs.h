/* Copyright (C) 2022 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/******************************************************************************
 * @file     board_defs.h
 * @brief    BOARD definitions for the Alif Application Kit (rev. Alpha1)
 ******************************************************************************/

#define BOARD_CAMERA_I2C_INSTANCE               30 /* Signifying I3C interface */

#define BOARD_I2C_INSTANCE                      0

#define BOARD_I2S_INSTANCE                      0

#define BOARD_UART1_INSTANCE                    4

#define BOARD_UART2_INSTANCE                    2

#define BOARD_LED_COUNT                         2
#define BOARD_LEDRGB_COUNT                      0
#define BOARD_BUTTON_COUNT                      0

// <o> User LED1 (ds1) GPIO port number and pin number
#define BOARD_LED1_GPIO_PORT                    2
#define BOARD_LED1_PIN_NO                       29

// <o> User LED2 (ds2) GPIO port number and pin number
#define BOARD_LED2_GPIO_PORT                    1
#define BOARD_LED2_PIN_NO                       14

// <o> LCD panel reset GPIO port number and pin number
#define BOARD_LCD_RESET_GPIO_PORT               4
#define BOARD_LCD_RESET_PIN_NO                  6

// <o> LCD panel backlight GPIO port number and pin number
#define BOARD_LCD_BACKLIGHT_GPIO_PORT           4
#define BOARD_LCD_BACKLIGHT_PIN_NO              4

// <o> Touch screen reset GPIO port number and pin number
#define BOARD_TOUCH_RESET_GPIO_PORT             4
#define BOARD_TOUCH_RESET_PIN_NO                2

// <o> Touch screen interrupt GPIO port number and pin number
#define BOARD_TOUCH_INT_GPIO_PORT               2
#define BOARD_TOUCH_INT_PIN_NO                  20

// <o> Camera module reset GPIO port number and pin number
#define BOARD_CAMERA_RESET_GPIO_PORT            4
#define BOARD_CAMERA_RESET_PIN_NO               5

// <o> Camera module power supply GPIO port number and pin number
#define BOARD_CAMERA_POWER_GPIO_PORT            2
#define BOARD_CAMERA_POWER_PIN_NO               7
