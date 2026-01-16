# Learning in Real-World Partially Observable Environments: Revisiting Efficient Deep Reinforcement Learning DVFS Optimization for Resource-Constrained Embedded Devices

Moayadeldin Hussain, Man Lin


<img width="2008" height="978" alt="solution_architecture-1-1" src="https://github.com/user-attachments/assets/7bd005fd-a5c8-46ec-ba47-f3eed1c7991a" />

## Requirements
### Jetson Nano 4GB
- Jetpack 4.6.1
- Cuda 10.2
- OpenCV 4.5

All the experiments reported were conducted using Ubuntu 16 on a single NVIDIA Jetson device.

## Running

1. Run the agent.
```
$ python agent.py
```
2. Simultaneously, run the client with either rendering or YOLO task with your specified workload characteristics.
```
$ python client.py --IP_ADDR 172.17.0.1 --app YOLO --exp_time 3000 --target_fps 3
```

## Citation
If you find our work helpful, please cite as follows:
```bibtex
@article{Hussain_LPO_DVFS,
  author = {Moayadeldin Hussain, Man Lin},
  title = {Learning in Real-World Partially Observable Environments: Revisiting Efficient DVFS Optimization for Resource-Constrained Embedded Devices.},
  year = {2026}
}
```

## Acknowledgment

This repository contains modified code from [zTT](https://github.com/ztt-21/zTT).

