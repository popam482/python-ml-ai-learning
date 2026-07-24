"""

SENSOR PULSE ANALYTICS using NumPy

100x6 matrix: 6 sensors over 100 time steps

"""
from logging import critical

import numpy as np


def generate_data():
    data_matrix = np.random.uniform(low=10, high=100, size=(100, 6))
    # synthetic noise and outliers
    indices = np.random.choice(data_matrix.size, size=30, replace=False)
    rows, cols = np.unravel_index(indices, data_matrix.shape)
    data_matrix[rows, cols] *= -1
    print("Generated raw values: \n", data_matrix)
    return data_matrix


def clean_data(raw_data):

    column_means = np.mean(raw_data, axis=0)

    invalid = raw_data <= 0

    rows, cols = np.where(invalid)

    raw_data[rows, cols] = column_means[cols]

    min_values = np.min(raw_data, axis=0)
    max_values = np.max(raw_data, axis=0)

    normalized_data = (raw_data - min_values) / (max_values - min_values)

    print("\nNormalized data: \n", normalized_data)

    return normalized_data


def biggest_sensor_variation(data):
    stdevs = np.std(data, axis=0)
    sensor_index = np.argmax(stdevs)
    return sensor_index


def check_composite_health_index(data, threshold_low, threshold_high):
    weights = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10])

    health_index = np.sum(data * weights, axis=1)

    critical = np.where(
        (health_index < threshold_low) |
        (health_index > threshold_high)
    )[0]

    print("\nComposite Health Index:")
    print(health_index)

    print("\nCritical ticks:")
    print(critical)

    print("\nTotal critical ticks:", len(critical))





raw_data = generate_data()
normalized_data = clean_data(raw_data)
most_volatile_sensor = biggest_sensor_variation(normalized_data)
critical_threshold_low = 0.3
critical_threshold_high = 0.85
check_composite_health_index(normalized_data, critical_threshold_low, critical_threshold_high)
