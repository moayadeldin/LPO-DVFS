#!/usr/bin/env python3
import os
from threading import Thread
import time
import numpy as np
import random
import cv2
import socket
import matplotlib.pyplot as plt
import csv
import struct
from time import sleep
import argparse
import subprocess
from Util import CPU, GPU, SYSTEM, FPSDriver
import mmap
from inotify_simple import INotify, flags
from collections import deque

DEFAULT_PROTOCOL = 0
PORT = 8703


HISTORY_LEN = 4  #store 4 consecutive states
FEATURE_DIM = 7  #each single-step state is 7-dimensional (c_c, g_c, c_p, g_p, c_t, g_t, fps)

driver_path="Util/chromedriver"


target_temp=80
beta=6
fan_speed=0 # 0-5, 0 : turn off 

def setUserspace():
    for i in range(0,3):
        fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_governor' %(i)
        with open(fname,'w') as f:
            f.write('userspace')
            print('[cpu{}]Set userspace '.format(i),end="")
            f.close()
    fname='/sys/devices/gpu.0/devfreq/57000000.gpu/governor'
    with open(fname,'w') as f:
        f.write('userspace')
        print('[gpu]Set userspace')
    fname='/sys/devices/pwm-fan/temp_control'
    with open(fname,'w') as f:
        f.write('1')
        print('[FAN] control on')

def setdefault(mode):
    for i in range(0,3):
        fname='/sys/devices/system/cpu/cpu%s/cpufreq/scaling_governor' %(i)
        with open(fname,'w') as f:
            f.write(mode)
            print('[cpu{}]Set {}'.format(i,mode),end="")
            f.close()
    fname='/sys/devices/gpu.0/devfreq/57000000.gpu/governor'
    with open(fname,'w') as f:
        f.write('simple_ondemand')
        print('[gpu]simple_ondemand')
    

def getCurrentClock():
    for i in range(0,3):
        fname='/sys/devices/system/cpu/cpu%s/cpufreq/cpuinfo_cur_freq' %(i)
        with open(fname,'r') as f:
            line=f.readline()
            line=line.replace('\n','')
            print('[cpu{}]{}KHz '.format(i,line),end=""),
            f.close()
    fname='/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq'
    with open(fname,'r') as f:
        line=f.readline()
        line=line.replace('\n','')
        print('[gpu]{}Hz'.format(line))
        f.close()

        

if __name__=="__main__":
    
    
    
    """Get arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--IP_ADDR', type=str, default='', help='[server ip address (e.g. 192.168.1.1]')
    parser.add_argument('--app', type=str, default='rendering', help='Select an app to test: [rendering, YOLO, aquarium]')
    parser.add_argument('--file', type=str, default='video.mp4', help='Path of a video file for rendering')
    parser.add_argument('--exp_time', type=int, default='1500', help='Set time for experiment')
    parser.add_argument('--target_fps', type=int, default='30', help='Set target_fps [e.g., 30]')
    args = parser.parse_args()
    experiment_time=args.exp_time
    target_fps=args.target_fps
    IP_ADDR=args.IP_ADDR
    app=args.app
    print(args)
    print(args.app)
    print(args.file)

    """
    Initialize variables, modules and governor(userspace)
    """
    setUserspace()
#	setdefault('performance')
    getCurrentClock()


    c0=CPU.CPU(0)
    c1=CPU.CPU(1)
    g=GPU.GPU()
    fan=SYSTEM.FAN()
    sys=SYSTEM.SYSTEM()
    ddr=SYSTEM.DDR()
    getCurrentClock()

    c0.setCPUclock(11)
    c1.setCPUclock(11)
    g.setGPUclock(11)
    fan.setFANspeed(fan_speed)
    t=0
    fps=0
    cnt=0
    rd = 0
    ts=[]
    fps_data=[]
    viofps_data=[]
    avg_q_max_data=[]
    fan_data=[]
    prevTime=0
    curTime=0
    cnt=0
    u=0
    clk=11
    c_c=11
    g_c=11
    c_t=float(c0.getCPUtemp())
    g_t=float(g.getGPUtemp())
    
    """
    State definition for DRL
    state = (CPU clk, GPU clk, CPU power, GPU Power, CPU temperature, GPU temperature, fps)	
    """
    state=(c_c,g_c,int(c0.getCPUpower()/100),int(g.getGPUpower()/100), c_t, g_t,fps)
    
        
    state_deque = deque(maxlen=HISTORY_LEN)
    
    for _ in range(HISTORY_LEN):
        state_deque.append(state)
    



    if(app=="rendering"):
        file=args.file
        capture=cv2.VideoCapture(file)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 90)
    elif(app=="aquarium"):
        fps_driver = FPSDriver.FPSDriver(driver_path)
        aquarium_url = "https://webglsamples.org/aquarium/aquarium.html"
        fps_driver.open_page(aquarium_url)
    elif(app == "YOLO"):
            command_dir = os.path.abspath("yolov3")
            command_file = os.path.join(command_dir, "test")
            
            subprocess.Popen(['xterm', '-e', 'sh', command_file], cwd=command_dir)
            
            mmap_file_name = "yolov3/ipc_fps.txt"
            print(mmap_file_name)
            
            if not os.path.exists(mmap_file_name) or os.path.getsize(mmap_file_name) == 0:
                with open(mmap_file_name, "w") as f:
                    f.write("0" * 1024)
                print('File initialized')
            
            inotify = INotify()
            wd = inotify.add_watch(mmap_file_name, flags.MODIFY)
        

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((IP_ADDR,8703))

    while(1):
        
        if(app=="rendering"):
            if(capture.get(cv2.CAP_PROP_POS_FRAMES)==capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                capture.open(file)

            ret, frame=capture.read()


            str_msg="FPS: %0.1f" % fps
            cv2.putText(frame,str_msg,(0,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,355,0))
            cv2.imshow("VideoFrame",frame)
            cnt=cnt+1
            if cnt==30:
                curTime=time.time()
                cnt=0
                sec=curTime-prevTime
                prevTime=curTime
                fps=round(30/(sec),1)
                rd = 1
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break

        elif(app=="aquarium"):
            fps = float(fps_driver.get_fps())
            sleep(1)
            rd = 1

        elif(app == "YOLO"):
            try:
                events = inotify.read(timeout=1000) 
                
                if events:
                    for event in events:
                        if event.mask & flags.MODIFY:
                            try:
                                with open(mmap_file_name, "rb") as f:
                                    data = f.read(8) 
                                    if len(data) == 8:
                                        print(fps)
                                        fps = struct.unpack('d', data)[0]
                            except Exception as e:
                                print(f"Error reading FPS data: {e}")
                rd = 1
            except Exception as e:
                print(f"Error in YOLO processing: {e}")
                continue


        fps_data.append(fps)
        ts.append(t)
        
        
        if rd == 1:
            c0.collectdata()
            c1.collectdata()
            g.collectdata()
            sys.collectdata()
            ddr.collectdata()

            c_p = int(c0.getCPUpower()/100)
            g_p = int(g.getGPUpower()/100)
            c_t = float(c0.getCPUtemp())
            g_t = float(g.getGPUtemp())

            next_single_state = (c_c, g_c, c_p, g_p, c_t, g_t, fps)

            state_deque.append(next_single_state)

            stacked_state = np.array(state_deque)  # shape => (4,7)

            #flatten to a 1D array of length 28 for sending over socket
            flat_stacked = stacked_state.flatten()  # shape => (28,)

            send_msg = ",".join(str(val) for val in flat_stacked)
            client_socket.send(send_msg.encode())  # send to server

            print(f"[{t}] stacked_state shape: {stacked_state.shape}, sending {len(flat_stacked)} vals")

            recv_msg = client_socket.recv(8703).decode()
            clk = recv_msg.split(',')
            print('action taken ', clk)
            if len(clk) < 2:
                print("Error: not enough data from server")
                break

            c_c = int(clk[0])
            g_c = int(clk[1])

            c0.setCPUclock(c_c)
            c1.setCPUclock(c_c)
            g.setGPUclock(g_c)
            rd = 0

            t += 1
            fps_data.append(fps)
            ts.append(t)

            if t >= experiment_time:
                break

    client_socket.close()

    c0.setCPUclock(11)
    c1.setCPUclock(11)
    g.setGPUclock(12)
    fan.setFANspeed(5)

    if len(ts) > 0:
        print('Average Total power={} mW'.format(sum(sys.power_data)/len(sys.power_data)))
        print('Average CPU power={} mW'.format(sum(c0.power_data)/len(c0.power_data)))
        print('Average GPU power={} mW'.format(sum(g.power_data)/len(g.power_data)))
        print('Average DDR power={} mW'.format(sum(ddr.power_data)/len(ddr.power_data)))
        print('Average fps = {} fps'.format(sum(fps_data)/len(fps_data)))

