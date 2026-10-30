
#include <Wire.h>
#include <MAX30105.h>
#include <heartRate.h>

MAX30105 particleSensor;

const byte RATE_SIZE = 20;  // Number of measurements to average.
float rates[RATE_SIZE] = {0};     // Array of heart rates.
byte rateSpot = 0;
long lastBeat = 0;  // Time at which the last beat occurred.
byte rateCount = 0;

float beatsPerMinute = 0;
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

  if (checkForBeat(irValue) == true) {
    // Only calculate BPM if a previous beat exists
    if (lastBeat != 0) {
      // Sensed a beat.
      long delta = now - lastBeat;
      beatsPerMinute = 60 / (delta / 1000.0);

      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = beatsPerMinute;
        rateSpot %= RATE_SIZE;

        // Count only valid measurements
        if (rateCount < RATE_SIZE) {
          rateCount++;
        }

        // Calculate average.
        beatAvg = 0;
        for (byte x = 0; x < rateCount; x++)
          beatAvg += rates[x];

        if (rateCount > 0) {
            beatAvg /= rateCount;
        }
      }
    }

    lastBeat = now;
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
