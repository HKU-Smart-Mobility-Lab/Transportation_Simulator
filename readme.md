### An Open-Sourced Network-Based Large-Scale Simulation Platform for Shared Mobility Operations

![效果图](https://img.shields.io/static/v1?label=build&message=passing&color=green) ![](https://img.shields.io/static/v1?label=python&message=3&color=blue) ![](https://img.shields.io/static/v1?label=release&message=2.0&color=green) ![](https://img.shields.io/static/v1?label=license&message=MIT&color=blue)

### Background

Establish an open-sourced network-based simulation platform for shared mobility operations. The simulation explicitly characterizes drivers’ movements on road networks for trip delivery, idle cruising, and en-route pick-up. 

### Install

1. Download the code.

`git clone git@github.com:HKU-Smart-Mobility-Lab/Transpotation_Simulator.git`

2. Download the dependencies and libraries.

`pip install -r requirements.txt`

### File Structure

```
-simulator
	--input
    ---graph.graphml
    ---order.pickle
    ---driver_info.pickle
  --output
  	---
 	--driver_generation.py
 	--handle_raw_data.py
 	--find_closest_point.py
 	--simulator_pattern.py
 	--simulator_env.py
 	--A2C.py
 	--sarsa.py
 	--main.py
 	--config.py
 	--LICENSE.md
 	--readme.md
 	
```

### Tutorials







### Technical questions



### Citing

If you use this simulator for academic research, you are highly encouraged to cite our paper:

An Open-Sourced Network-Based Large-Scale Simulation Platform for Shared Mobility Operations



### Contributors

This simulator is supported by the [Smart Mobility Lab](	https://github.com/HKU-Smart-Mobility-Lab) at The Univerisity of Hong Kong.







##### 