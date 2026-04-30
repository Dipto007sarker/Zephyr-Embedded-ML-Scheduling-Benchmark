/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #include "accelerometer_handler.hpp"

// #include <zephyr/device.h>
// #include <zephyr/drivers/sensor.h>
// #include <stdio.h>
// #include <string.h>
// #include <zephyr/kernel.h>

// #define BUFLEN 300
// int begin_index = 0;
// const struct device *const sensor = DEVICE_DT_GET_ONE(adi_adxl345);
// int current_index = 0;

// float bufx[BUFLEN] = { 0.0f };
// float bufy[BUFLEN] = { 0.0f };
// float bufz[BUFLEN] = { 0.0f };

// bool initial = true;

// TfLiteStatus SetupAccelerometer()
// {
// 	if (!device_is_ready(sensor)) {
// 		printk("%s: device not ready.\n", sensor->name);
// 		return kTfLiteApplicationError;
// 	}

// 	MicroPrintf("Got accelerometer, name: %s\n", sensor->name);

// 	return kTfLiteOk;
// }

// bool ReadAccelerometer(float *input, int length)
// {
// 	int rc;
// 	struct sensor_value accel[3];
// 	int samples_count;

// 	rc = sensor_sample_fetch(sensor);
// 	if (rc < 0) {
// 		MicroPrintf("Fetch failed\n");
// 		return false;
// 	}
// 	/* Skip if there is no data */
// 	if (!rc) {
// 		return false;
// 	}

// 	samples_count = rc;
// 	for (int i = 0; i < samples_count; i++) {
// 		rc = sensor_channel_get(sensor, SENSOR_CHAN_ACCEL_XYZ, accel);
// 		if (rc < 0) {
// 			MicroPrintf("ERROR: Update failed: %d\n", rc);
// 			return false;
// 		}
// 		bufx[begin_index] = (float)sensor_value_to_double(&accel[0]);
// 		bufy[begin_index] = (float)sensor_value_to_double(&accel[1]);
// 		bufz[begin_index] = (float)sensor_value_to_double(&accel[2]);
// 		begin_index++;
// 		if (begin_index >= BUFLEN) {
// 			begin_index = 0;
// 		}
// 	}

// 	if (initial && begin_index >= 100) {
// 		initial = false;
// 	}

// 	if (initial) {
// 		return false;
// 	}

// 	int sample = 0;
// 	for (int i = 0; i < (length - 3); i += 3) {
// 		int ring_index = begin_index + sample - length / 3;
// 		if (ring_index < 0) {
// 			ring_index += BUFLEN;
// 		}
// 		input[i] = bufx[ring_index];
// 		input[i + 1] = bufy[ring_index];
// 		input[i + 2] = bufz[ring_index];
// 		sample++;
// 	}
// 	return true;
// }



#include "accelerometer_handler.hpp"

#include <cmath>
#include <tensorflow/lite/micro/micro_log.h>

int begin_index = 0;

namespace {
constexpr int BUFLEN = 300;
constexpr float kPi = 3.14159265358979323846f;

float bufx[BUFLEN] = {0.0f};
float bufy[BUFLEN] = {0.0f};
float bufz[BUFLEN] = {0.0f};

bool initial = true;
int sample_counter = 0;
}

TfLiteStatus SetupAccelerometer()
{
    begin_index = 0;
    sample_counter = 0;
    initial = true;

    MicroPrintf("Using synthetic accelerometer input\n");
    return kTfLiteOk;
}

bool ReadAccelerometer(float *input, int length)
{
    if (input == nullptr || length <= 0) {
        return false;
    }

    // Simulate several new accelerometer samples each call
    const int samples_count = 5;

    for (int i = 0; i < samples_count; i++) {
        float t = static_cast<float>(sample_counter) * 0.08f;

        bufx[begin_index] = 0.8f * std::sin(t);
        bufy[begin_index] = 0.6f * std::cos(1.2f * t);
        bufz[begin_index] = 0.5f * std::sin(0.7f * t + kPi / 4.0f);

        begin_index++;
        if (begin_index >= BUFLEN) {
            begin_index = 0;
        }

        sample_counter++;
    }

    if (initial && sample_counter >= 100) {
        initial = false;
    }

    if (initial) {
        return false;
    }

    int sample = 0;
    for (int i = 0; i < (length - 3); i += 3) {
        int ring_index = begin_index + sample - length / 3;
        if (ring_index < 0) {
            ring_index += BUFLEN;
        }

        input[i] = bufx[ring_index];
        input[i + 1] = bufy[ring_index];
        input[i + 2] = bufz[ring_index];
        sample++;
    }

    return true;
}