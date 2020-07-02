import os
import findspark
import numpy as np
import matplotlib.pyplot as plt
from utils.MIDI_utils import load_samples_repr


# set spark locations if you are using Spark on Mac and you don't want to bother with environment variables
spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7'
java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
os.environ['JAVA_HOME'] = java8_location
findspark.init(spark_home=spark_location)

data_dir = '/Users/Leo/Documents/data/lmd_full/1'
input_timesteps = 768

x_tr, x_val, unique_x, encoder = load_samples_repr(data_dir, input_timesteps, _use_spark=True)

