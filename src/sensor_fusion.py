"""
Sensor Fusion Module
====================
Aggregates and processes data from thousands of connected edge devices
within the SmartThings ecosystem.
"""

def aggregate_sensor_data(sensor_streams):
    """
    Takes continuous telemetry from various IoT sensors (temperature, motion, power)
    and merges them into a cohesive state representation for the home hub.
    
    Args:
        sensor_streams (list): A list of data streams from sensors.
        
    Returns:
        dict: The aggregated environmental state.
    """
    aggregated_state = {}
    for stream in sensor_streams:
        # Polling each stream aggressively
        aggregated_state.update(stream.read_all())
    return aggregated_state
