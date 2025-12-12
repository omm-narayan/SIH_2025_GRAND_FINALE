#include <Arduino.h>

// Human detection (YOUR ORIGINAL CODE)
const int radarPin = 23;
const int ledPin = 2;
const int windowSize = 100;
const int sampleInterval = 100;

// CO2 on pin 34
const int co2Pin = 34;

int samples[windowSize];
int indexSample = 0;

void setup() {
  Serial.begin(115200);
  pinMode(radarPin, INPUT);
  pinMode(ledPin, OUTPUT);
  pinMode(co2Pin, INPUT);
  for(int i = 0; i < windowSize; i++) samples[i] = 1;
}

bool checkBreathing() {
  int transitions = 0;
  int prev = samples[0];
  for (int i = 1; i < windowSize; i++) {
    if (samples[i] != prev) transitions++;
    prev = samples[i];
  }
  return (transitions >= 3 && transitions <= 10);
}

void loop() {
  int radarVal = digitalRead(radarPin);
  samples[indexSample] = radarVal;
  indexSample = (indexSample + 1) % windowSize;
  
  int co2Raw = analogRead(co2Pin);
  int co2ppm = map(co2Raw, 0, 4095, 400, 2000);
  
  static unsigned long lastCheck = 0;
  if (millis() - lastCheck > windowSize * sampleInterval) {
    lastCheck = millis();
    
    if (checkBreathing()) {
      Serial.print("HUMAN,");
      digitalWrite(ledPin, HIGH);
    } else {
      Serial.print("NO HUMAN,");
      digitalWrite(ledPin, LOW);
    }
    
    Serial.println(co2ppm);
  }
  
  delay(sampleInterval);
}