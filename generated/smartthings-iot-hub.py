"""
Implementation of a core decision-making algorithm for home automation systems,
inspired by the framework proposed in the paper "Home automation in the wild:
challenges and opportunities".

Source Paper:
    Title: Home automation in the wild: challenges and opportunities
    URL: https://www.semanticscholar.org/paper/51b3125609a84cf13900bcabfcdfd707d4c7c41b

Mathematical Idea (in plain English):
The paper describes a conceptual framework for home automation systems,
emphasizing a "Data Processing and Analytics Layer" that includes Context
Awareness, Predictive Analytics, and Decision-Making modules. This
implementation translates that framework into a rule-based inference engine.
The core mathematical idea is a form of finite state machine or expert system
where the system's "state" is defined by inferred contexts (e.g., user presence,
time of day). Decisions are made by evaluating a set of predefined rules. Each
rule consists of a condition (a boolean function of sensor data, current
context, and simple predictions) and an action (a function that generates
actuator commands). The system iteratively checks these rules against the
current environmental and contextual data to determine appropriate actions.
A simplified Haversine formula is used for geographical distance calculation
to infer user presence (home/away).

Key Hyperparameters and Default Values:
- `preferred_temperature_home` (float): Default 22.0 (Celsius).
- `