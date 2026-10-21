# ESP32-S3 N16R8: on-device activity recognition from Wi-Fi CSI.
# Build with ESP-IDF 5.4 (the version the authors used for the collecting tool):
#   idf.py set-target esp32s3 && idf.py build flash monitor
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(wisense_har)
