/* Copyright (C) 2023 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/******************************************************************************
 * @file     board.h
 * @brief    BOARD API
 ******************************************************************************/

#define BOARD_CAMERA_I2C_INSTANCE               1

#define BOARD_I2S_INSTANCE                      3

#define BOARD_PDM_INSTANCE                      2

#define BOARD_UART1_INSTANCE                    2

#define BOARD_UART2_INSTANCE                    4

#define BOARD_LED_COUNT                         2
#define BOARD_LEDRGB_COUNT                      2
#define BOARD_BUTTON_COUNT                      2

// <o> User BUTTON1 (JOY_SW4 - down) GPIO port number and pin number
#define BOARD_BUTTON1_GPIO_PORT                 LP
#define BOARD_BUTTON1_PIN_NO                    3

// <o> User BUTTON2 (JOY_SW5 - click) GPIO port number and pin number
#define BOARD_BUTTON2_GPIO_PORT                 LP
#define BOARD_BUTTON2_PIN_NO                    4

// <o> User LEDRGB0 (D4) GPIO port numbers and pin numbers
#define BOARD_LEDRGB0_R_GPIO_PORT               12
#define BOARD_LEDRGB0_R_PIN_NO                  3

#define BOARD_LEDRGB0_G_GPIO_PORT               7
#define BOARD_LEDRGB0_G_PIN_NO                  4

#define BOARD_LEDRGB0_B_GPIO_PORT               12
#define BOARD_LEDRGB0_B_PIN_NO                  0

// <o> User LEDRGB1 (D6) GPIO port numbers and pin numbers
#define BOARD_LEDRGB1_R_GPIO_PORT               6
#define BOARD_LEDRGB1_R_PIN_NO                  2

#define BOARD_LEDRGB1_G_GPIO_PORT               6
#define BOARD_LEDRGB1_G_PIN_NO                  5

#define BOARD_LEDRGB1_B_GPIO_PORT               6
#define BOARD_LEDRGB1_B_PIN_NO                  6

// <o> User LED1 (D4-R) GPIO port number and pin number
#define BOARD_LED1_GPIO_PORT                    12
#define BOARD_LED1_PIN_NO                       3

// <o> User LED2 (D6-G) GPIO port number and pin number
#define BOARD_LED2_GPIO_PORT                    6
#define BOARD_LED2_PIN_NO                       5

// <o> LCD panel reset GPIO port number and pin number
#define BOARD_LCD_RESET_GPIO_PORT               LP
#define BOARD_LCD_RESET_PIN_NO                  5

// <o> LCD panel backlight GPIO port number and pin number
#define BOARD_LCD_BACKLIGHT_GPIO_PORT           6
#define BOARD_LCD_BACKLIGHT_PIN_NO              1

// <o> Touch screen reset GPIO port number and pin number
#define BOARD_TOUCH_RESET_GPIO_PORT             4
#define BOARD_TOUCH_RESET_PIN_NO                0

// <o> Touch screen interrupt GPIO port number and pin number
#define BOARD_TOUCH_INT_GPIO_PORT               9
#define BOARD_TOUCH_INT_PIN_NO                  4

// <o> Camera module power GPIO port number and pin number
#define BOARD_CAMERA_POWER_GPIO_PORT            7
#define BOARD_CAMERA_POWER_PIN_NO               5

// <o> Camera module reset GPIO port number and pin number
#define BOARD_CAMERA_RESET_GPIO_PORT            9
#define BOARD_CAMERA_RESET_PIN_NO               1

// <o> Camera module flash GPIO port number and pin number
#define BOARD_CAMERA_FLASH_GPIO_PORT            9
#define BOARD_CAMERA_FLASH_PIN_NO               2

// <o> Camera module trigger GPIO port number and pin number
#define BOARD_CAMERA_TRIGGER_GPIO_PORT          9
#define BOARD_CAMERA_TRIGGER_PIN_NO             3
