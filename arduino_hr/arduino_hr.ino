
#include <Wire.h>
#include <MAX30105.h>
#include <heartRate.h>

MAX30105 particleSensor;

long lastBeat = 0;  // Time at which the last beat occurred.

float beatsPerMinute = 0;
const unsigned long WINDOW_MS = 10000;

const byte MAX_BEATS = 100;

unsigned long beatTimestamps[MAX_BEATS];

byte beatCount = 0;
byte beatIndex = 0;

float beatAvg = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("Initializing...");

  // Initialize sensor
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 was not found. Please check wiring/power. ");
    while (1)
      ;
  }

  Serial.println("Place your index finger on the sensor with steady pressure.");

  particleSensor.setup();                     // Configure sensor with default settings
  particleSensor.setPulseAmplitudeRed(0x0A);  // Turn Red LED to low to indicate sensor is running
  particleSensor.setPulseAmplitudeGreen(0);   // Turn off Green LED

  Serial.println("timestamp,ir,avg_bpm");
}

void loop() {
  long irValue = particleSensor.getIR();

  // Timestamp for every sensor reading
  long now = millis();

  if (checkForBeat(irValue)) {

      if (lastBeat != 0) {

          unsigned long delta = now - lastBeat;

          beatsPerMinute = 60000.0 / delta;

          if (beatsPerMinute >= 20 &&
              beatsPerMinute <= 255) {

              beatTimestamps[beatIndex] = now;

              beatIndex = (beatIndex + 1) % MAX_BEATS;

              if (beatCount < MAX_BEATS) {
                  beatCount++;
              }
          }
      }

      lastBeat = now;
  }

  // Calculate BPM using beats within the last 40 seconds

  unsigned long windowStart = now - WINDOW_MS;

  unsigned long oldestBeat = 0;
  unsigned long newestBeat = 0;

  byte validBeats = 0;

  for (byte i = 0; i < beatCount; i++) {

      unsigned long t = beatTimestamps[i];

      if (now - t <= WINDOW_MS) {

          if (validBeats == 0 || t < oldestBeat) {
              oldestBeat = t;
          }

          if (validBeats == 0 || t > newestBeat) {
              newestBeat = t;
          }

          validBeats++;
      }
  }

  if (validBeats >= 2 && newestBeat > oldestBeat) {

      beatAvg =
          (validBeats - 1) * 60000.0 /
          (newestBeat - oldestBeat);

  } else {

      beatAvg = 0;
  }

  // Print timestamp, IR and average BPM
  Serial.print(now);
  Serial.print(",");
  Serial.print(irValue);
  // Serial.print(",");
  // Serial.print(beatsPerMinute);
  Serial.print(",");
  Serial.println(beatAvg, 2);
}
