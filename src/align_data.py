
import numpy as np
import csv
import os

CSI_FILE = "../data/training_data.txt"
CSI_TIME_FILE = "../data/csi_timestamps.csv"

HR_FILE = "../data/hr_data.txt"
HR_TIME_FILE = "../data/hr_timestamps.csv"

OUTPUT_CSI = "../data/training_data_aligned.txt"
OUTPUT_HR = "../data/hr_data_aligned.txt"


def load_values(filename):
    return np.atleast_1d(
        np.loadtxt(filename, dtype=float)
    )


def main():

    print("Loading CSI data...")

    csi_times = load_values(CSI_TIME_FILE)

    with open(CSI_FILE, "r") as file:
        csi_data = [
            line.strip()
            for line in file
            if line.strip()
        ]

    print("Loading heart-rate data...")

    hr_times = load_values(HR_TIME_FILE)
    hr_values = load_values(HR_FILE)

    # Check that each measurement has a timestamp.
    if len(csi_data) != len(csi_times):
        raise ValueError(
            f"CSI mismatch: {len(csi_data)} rows but "
            f"{len(csi_times)} timestamps"
        )

    if len(hr_values) != len(hr_times):
        raise ValueError(
            f"HR mismatch: {len(hr_values)} values but "
            f"{len(hr_times)} timestamps"
        )

    if len(hr_times) < 2:
        raise ValueError("Not enough HR estimates to align.")

    # Sort HR data by timestamp.
    order = np.argsort(hr_times)
    hr_times = hr_times[order]
    hr_values = hr_values[order]

    # Remove duplicate timestamps, if any.
    hr_times, unique_indices = np.unique(
        hr_times,
        return_index=True
    )
    hr_values = hr_values[unique_indices]

    # Estimate CSI sampling rate.
    csi_diffs = np.diff(csi_times)

    valid_diffs = csi_diffs[csi_diffs > 0]

    if len(valid_diffs) == 0:
        raise ValueError("Invalid CSI timestamps.")

    csi_rate = 1.0 / np.median(valid_diffs)

    print(f"\nCSI samples: {len(csi_data)}")
    print(f"HR estimates: {len(hr_values)}")
    print(f"Estimated CSI sampling rate: {csi_rate:.2f} Hz")

    # Find the overlapping time interval.
    start_time = max(
        np.min(csi_times),
        np.min(hr_times)
    )

    end_time = min(
        np.max(csi_times),
        np.max(hr_times)
    )

    if end_time <= start_time:
        raise ValueError(
            "CSI and HR recordings have no overlapping time interval."
        )

    # Select CSI samples within the HR time range.
    valid_indices = np.where(
        (csi_times >= start_time) &
        (csi_times <= end_time)
    )[0]

    aligned_times = csi_times[valid_indices]

    if len(aligned_times) < 101:
        raise ValueError(
            "Insufficient overlapping CSI samples for LSTM training."
        )

    # Interpolate HR values onto CSI timestamps.
    aligned_hr = np.interp(
        aligned_times,
        hr_times,
        hr_values
    )

    # Save matching CSI and HR rows.
    with open(OUTPUT_CSI, "w") as csi_file, \
         open(OUTPUT_HR, "w") as hr_file:

        for index, bpm in zip(valid_indices, aligned_hr):

            csi_file.write(csi_data[index] + "\n")
            hr_file.write(f"{bpm:.2f}\n")

    print("\nAlignment completed!")

    print(f"Aligned CSI rows: {len(valid_indices)}")
    print(f"Aligned HR rows: {len(aligned_hr)}")

    print(
        f"Usable recording duration: "
        f"{end_time - start_time:.2f} seconds"
    )

    print(f"\nSaved: {OUTPUT_CSI}")
    print(f"Saved: {OUTPUT_HR}")


if __name__ == "__main__":
    main()
