# Preprocessing Data

The goal of this directory is to preprocess the Open X-Embodiment (OXE) dataset for pretraining CLIP-RT. We use the OXE dataset curated by [OpenVLA](https://github.com/openvla/openvla).

## General Steps

1. Follow the installation guide in [OpenVLA](https://github.com/openvla/openvla) 
2. Download the OXE data (also in [OpenVLA](https://github.com/openvla/openvla))
3. Run `python preprocess.py`


## How to Extract Natural Language Supervision

We extract natural language supervision from 7D end-effector commands (the first six for position and rotation / the last one is gripper open and close). Specifically, we find the dominant axis in the end-effector commands and map it to natural language supervision. For example, the dominant axis is the z-axis with a value of -0.2, it is converted into: "lower the arm by twenty centimeters".

One problem is that each robotic dataset has different definitions about the Cartesian coordinate system and units for each value (e.g., meters, degrees, radians, etc). So, we manually check each robotic data in the OXE dataset and standardize those definitions (see preprocess.py).

Technically speaking, there can be an infinite number of natural language supervisions (e.g., "lower the arm by 11.52 centimeters"). However,to make VLA models more generalizable action representations, we predefine the granularities of robotic actions. For example, each axis for position control is divided into 8 granularities (-20cm, -10cm, -5cm, -1cm, 1cm, 5cm, 10cm, 20cm). Rotational control has 8 granularities (-90°, -45°, -15°, -5°, 5°, 15°, 45°, 90°). This means that the dominant axis values are discretized into one of those values. Finally, the discretized values are mapped into natural language supervisions. There are 50 types of natural language supervisions (8 * 6 dimensions + gripper open + gripper close commands). See `docs/action_to_label_pretrain.json` for more details. 

We prepare nearly 900 natural langauge supervisions to make VLA models learn diverse natural language supervision. We use GPT-4 to prepare those supervisions.
