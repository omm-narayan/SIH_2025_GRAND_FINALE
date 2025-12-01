#include <Arduino.h>
int radarPin = 23;  // HLK-LD2410 OUT pin

void setup() {
  Serial.begin(115200);
  pinMode(radarPin, INPUT_PULLDOWN);

  Serial.println("Human Presence Sensor - Ready");
}

void loop() {
  int val = digitalRead(radarPin);

  if (val == LOW) {
    Serial.println("Presence Detected");   // LOW = detection
  } else {
    Serial.println("No Presence");         // HIGH = no detection
  }

  delay(150);
}
