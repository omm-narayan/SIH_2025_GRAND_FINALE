#include <Arduino.h>

#include <driver/i2s.h>

// Pins
#define I2S_WS  15
#define I2S_SCK 14
#define I2S_SD  32

// Adjustable Parameters
float sensitivity = 0.07;      // Lower = less sensitivity
int noiseGate = 50;            // Ignore anything below this amplitude
float smooth1 = 0.04;          // First smoothing filter
float smooth2 = 0.03;          // Second smoothing filter

int32_t f1 = 0, f2 = 0;        // Filter states

void setup() {
  Serial.begin(115200);

  const i2s_config_t i2s_config = {
    mode: (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    sample_rate: 44100,
    bits_per_sample: I2S_BITS_PER_SAMPLE_32BIT,
    channel_format: I2S_CHANNEL_FMT_ONLY_LEFT,
    communication_format: I2S_COMM_FORMAT_I2S_MSB,
    dma_buf_count: 4,
    dma_buf_len: 256,
    use_apll: true
  };

  const i2s_pin_config_t pin_config = {
    bck_io_num: I2S_SCK,
    ws_io_num: I2S_WS,
    data_out_num: I2S_PIN_NO_CHANGE,
    data_in_num: I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

void loop() {
  int32_t raw = 0;
  size_t readBytes;

  i2s_read(I2S_NUM_0, &raw, sizeof(raw), &readBytes, portMAX_DELAY);

  if (readBytes > 0) {

    // --- 1. Convert & reduce amplitude ---
    int32_t sample = raw >> 14;
    sample = sample * sensitivity;

    // --- 2. Apply noise gate ---
    if (abs(sample) < noiseGate) sample = 0;

    // --- 3. Smooth filter 1 ---
    f1 = f1 + smooth1 * (sample - f1);

    // --- 4. Smooth filter 2 (refinement) ---
    f2 = f2 + smooth2 * (f1 - f2);

    Serial.println(f2);
  }
}
