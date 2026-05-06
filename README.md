# SmartThings IoT Hub - Repository Context

## Overview
This repository manages the central communication and data aggregation layer for Samsung's SmartThings ecosystem. It handles connections from millions of low-power edge devices via MQTT, Zigbee, and Matter protocols.

## Tech Stack & Architecture
- **Language**: Python (Microservices layer)
- **Protocols**: MQTT, REST, WebSockets
- **Constraints**:
  - Must handle massive concurrency without dropping packets.
  - Connected devices often run on coin-cell batteries, so the hub must minimize unnecessary wake-ups.

## Current Goals & Challenges
1. **Battery Drain Issues**: Our current polling interval is static (5 seconds). We need a dynamic, energy-aware scheduling algorithm (TinyML or adaptive polling) to extend sensor battery life.
2. **Data Aggregation**: Seeking more efficient ways to perform distributed edge inference instead of sending all raw data to the hub.
