
import serial
import numpy as np
import time
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

SERIAL_PORT = "/dev/ttyACM0"       # Change to your Arduino port
BAUD_RATE = 115200

SAMPLE_RATE = 50.12           # Hz
WINDOW_SIZE = 2000          # Approximately 20 seconds
UPDATE_INTERVAL = 1.0       # Update every second

MIN_HR = 42
MAX_HR = 210

OUTPUT_FILE = "hr_data.txt"

# Keep recent BPM estimates for smoothing
hr_history = deque(maxlen=5)

# Store incoming IR samples
ir_buffer = deque(maxlen=WINDOW_SIZE)

# ==========================================
# FFT HEART RATE CALCULATION
# ==========================================

def calculate_heart_rate(signal):

    signal = np.asarray(signal, dtype=float)

    # Remove DC component
    signal = signal - np.mean(signal)

    # Remove linear trend
    x = np.arange(len(signal))
    slope, intercept = np.polyfit(x, signal, 1)
    signal = signal - (slope * x + intercept)

    # Apply Hann window
    window = np.hanning(len(signal))
    signal = signal * window

    # Perform FFT
    fft_result = np.fft.rfft(signal)

    magnitude = np.abs(fft_result)

    # Calculate frequency bins
    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / SAMPLE_RATE
    )

    # Inspect possible fundamental frequency
    fundamental_indices = np.where(
        (frequencies >= 1.0) &
        (frequencies <= 1.8)
    )[0]

    # Inspect possible higher harmonics
    harmonic_indices = np.where(
        (frequencies >= 2.0) &
        (frequencies <= 3.5)
    )[0]

    fundamental_index = fundamental_indices[
        np.argmax(magnitude[fundamental_indices])
    ]

    harmonic_index = harmonic_indices[
        np.argmax(magnitude[harmonic_indices])
    ]

    fundamental_freq = frequencies[fundamental_index]
    harmonic_freq = frequencies[harmonic_index]

    print("\nFundamental frequency analysis:")

    print(
        f"Possible fundamental: {fundamental_freq:.3f} Hz "
        f"({fundamental_freq * 60:.2f} BPM)"
    )

    print(
        f"Possible harmonic: {harmonic_freq:.3f} Hz "
        f"({harmonic_freq * 60:.2f} BPM)"
    )

    print(
        f"Fundamental magnitude: {magnitude[fundamental_index]:.2f}"
    )

    print(
        f"Harmonic magnitude: {magnitude[harmonic_index]:.2f}"
    )


    valid_indices = np.where(
        (frequencies >= 0.5) &
        (frequencies <= 3.5)
    )[0]

    top_indices = valid_indices[
    np.argsort(magnitude[valid_indices])[-5:][::-1]
]

    print("\nStrongest frequency components:")

    for idx in top_indices:
        print(
            f"Frequency: {frequencies[idx]:.3f} Hz | "
            f"BPM: {frequencies[idx] * 60:.2f} | "
            f"Magnitude: {magnitude[idx]:.2f}"
        )

    if len(valid_indices) == 0:
        return None

    # Find dominant frequency
    peak_index = valid_indices[
        np.argmax(magnitude[valid_indices])
    ]

    # Parabolic interpolation for better frequency resolution
    if 0 < peak_index < len(magnitude) - 1:

        alpha = magnitude[peak_index - 1]
        beta = magnitude[peak_index]
        gamma = magnitude[peak_index + 1]

        denominator = alpha - 2 * beta + gamma

        if abs(denominator) > 1e-12:
            offset = 0.5 * (alpha - gamma) / denominator
        else:
            offset = 0.0

        offset = np.clip(offset, -0.5, 0.5)

    else:
        offset = 0.0

    # Refined frequency
    peak_frequency = (
        peak_index + offset
    ) * SAMPLE_RATE / WINDOW_SIZE

    # Convert frequency to BPM
    bpm = peak_frequency * 60.0

    return bpm


# ==========================================
# SAVE HEART RATE
# ==========================================

def save_heart_rate(bpm):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(OUTPUT_FILE, "a") as file:
        file.write(f"{timestamp},{bpm:.2f}\n")


def plot_ppg(signal):

    signal = np.asarray(signal)

    time_axis = np.arange(len(signal)) / SAMPLE_RATE

    plt.figure(figsize=(12, 4))

    plt.plot(time_axis, signal)

    plt.xlabel("Time (seconds)")
    plt.ylabel("IR Value")
    plt.title("MAX30102 Raw PPG Signal")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fft(signal):

    signal = np.asarray(signal, dtype=float)

    signal = signal - np.mean(signal)

    window = np.hanning(len(signal))

    fft_result = np.fft.rfft(signal * window)

    magnitude = np.abs(fft_result)

    frequencies = np.fft.rfftfreq(
        len(signal),
        d=1.0 / SAMPLE_RATE
    )

    plt.figure(figsize=(12, 4))

    plt.plot(frequencies, magnitude)

    plt.xlim(0.5, 3.5)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("MAX30102 FFT Spectrum")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN PROGRAM
# ==========================================
plots_shown = False

def main():

    # This instantly wipes the file empty
    with open(OUTPUT_FILE, "w") as file:
        pass

    print("Connecting to MAX30102...")

    ser = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=2
    )

    # Allow Arduino to reset
    time.sleep(2)
    ser.reset_input_buffer()

    print("Connected!")
    print("Collecting IR samples...")

    last_update = time.time()

    try:

        # Plot raw PPG and FFT spectrum once
        global plots_shown

        while True:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            # Ignore initialization messages
            if not line[0].isdigit():
                continue

            try:
                parts = line.split(",")

                if len(parts) != 3:
                    print(f"Invalid serial data: {line}")
                    continue

                timestamp_ms = int(parts[0])
                ir_value = int(parts[1])
                td_bpm = float(parts[2])

            except (ValueError, IndexError) as e:
                print(f"Parsing error: {e} | Received: {line}")
                continue

            # Check finger presence
            if ir_value < 50000:
                print("Please place your finger on the sensor.")
                ir_buffer.clear()
                hr_history.clear()
                plots_shown = False
                continue

            ir_buffer.append(ir_value)

            # Wait until the buffer is full
            if len(ir_buffer) < WINDOW_SIZE:
                continue

            # Calculate BPM periodically
            if time.time() - last_update >= UPDATE_INTERVAL:

                bpm = calculate_heart_rate(ir_buffer)

                if bpm is not None:

                    if not plots_shown:
                        plot_ppg(ir_buffer)
                        plot_fft(ir_buffer)
                        plots_shown = True

                    hr_history.append(bpm)

                    # Median filtering
                    stable_bpm = np.median(hr_history)

                    print(
                        f"FFT BPM: {bpm:.2f} | "
                        f"Stable BPM: {stable_bpm:.2f} | "
                        f"Time-domain BPM: {td_bpm:.2f}"
                    )

                    save_heart_rate(stable_bpm)

                last_update = time.time()

    except KeyboardInterrupt:

        print("\nStopping measurement...")

    finally:

        ser.close()


if __name__ == "__main__":
    main()
