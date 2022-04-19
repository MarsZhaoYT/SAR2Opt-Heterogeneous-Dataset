import subprocess
from subprocess import run
import numpy as np

# spring: 974  summer: 610  fall: 1015  winter: 506  sar2opt: 587/ 627
num_test = 627
        

if __name__ == '__main__':
    cmd = "bash Image_translation_codes/pytorch-CycleGAN-and-pix2pix/run_time.sh"

    results = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    times = []
    for line in results.stdout.readlines():
        tmp = line.decode("utf-8").strip()
        
        if 'cost time' in tmp:
            times.append((float(tmp.split(": ")[-1])) / num_test)
    
    print('All cost times:', times)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print("Average running time: ", avg_time)
    print("Deviation of running time: ", std_time)

