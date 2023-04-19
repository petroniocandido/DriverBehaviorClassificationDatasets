import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset

class DriverBehaviorDataset(Dataset):
  def __init__(self, sensors, target, classes, data : pd.DataFrame, **kwargs):
    
    self.capital = kwargs.get('capital', False)
    if self.capital:
      self.variables = ["X", "Y", "Z"]
    else:
      self.variables = ["x", "y", "z"]  
    self.sensors = sensors
    self.ycolumn = target
    self.classes = classes
    self.num_classes = len([k for k in self.classes.keys()])
    self.xcolumns = []
    for s in self.sensors.keys():
      for v in self.variables:
        self.xcolumns.append(s+v)
    self.x = torch.from_numpy(data[self.xcolumns].values)
    self.y_o = torch.from_numpy(data[self.ycolumn].values)
    self.y_b = self.process_y_binary()

  def __getitem__(self, idx):  
    return self.x[:, idx], self.y_b[idx]

  def __len__(self):
    return len(self.x)

  def process_y_binary(self):
    z = torch.zeros((len(self.y_o), self.num_classes) )
    for ix in range(len(self.y_o)):
      z[ix,self.y_o[ix] - 1] = 1
    return z

def plot(ds, **kwargs):
  import matplotlib.lines as mlines

  classes_legend = []
  for ct3,cl in ds.classes.items():
    classes_legend.append(mlines.Line2D([], [], marker=markers[ct3], color="black", \
                                        linestyle='None', markersize=5, label=cl))

  sensors_legend = []
  for ct2, sen in enumerate(ds.sensors.keys()):
    sensors_legend.append(mlines.Line2D([], [], color=colors[ct2], label=ds.sensors[sen]))

  start = kwargs.get("start",0)
  end = kwargs.get("end",100)
  samples = kwargs.get("samples",100)
  inc = max((end - start) // samples, 1)

  ixs = [k for k in range(start, end, inc)]

  fig, ax = plt.subplots(3, 1, figsize=(15,6))

  for ct1, var in enumerate(ds.variables):
    ax[ct1].set_ylabel(var)
    for ct2, sen in enumerate(ds.sensors.keys()):
      ixv = ct2 * 3 + ct1
      xsliced = ds.x[ixs, ixv]
      ysliced = ds.y_o[ixs]
      for ct3,cl in ds.classes.items():
        ys = ysliced == ct3
        indexes = ys.nonzero()
        ax[ct1].scatter(indexes,xsliced[indexes].numpy(), c=colors[ct2], \
                      s=sizes[ct3-1], marker=markers[ct3-1], label=ds.sensors[sen])
        
  lgd1 = ax[0].legend(handles=sensors_legend, bbox_to_anchor=(1.1, 1.05), title="Sensors")
  lgd2 = ax[1].legend(handles=classes_legend, bbox_to_anchor=(1.1, 1.05), title="Classes")
  plt.tight_layout()
  
 
def github_junior_ferreira_passin():
  df1 = pd.read_csv("./Data/github_junior_ferreira_passin/16.csv", sep=";")
  df2 = pd.read_csv("./Data/github_junior_ferreira_passin/17.csv", sep=";")
  df3 = pd.read_csv("./Data/github_junior_ferreira_passin/20.csv", sep=";")
  df4 = pd.read_csv("./Data/github_junior_ferreira_passin/21.csv", sep=";")
  
  df = pd.concat([df1, df2, df3, df4])

  classes = df["Classe"].unique().tolist()

  classes.index('normal')

  df["Classe"] = [classes.index(k) for k in df["Classe"].values]

  classes = { classes.index(k) : k for k in classes}

  return DriverBehaviorDataset(
      sensors = {
        "acln": "Linear Acceleration",
        "giro": "Gyroscope",
        "magn": "Magnetometer",
        "acel": "Accelerometer"
      }, 
      target = "Classe",
      classes = classes, 
      data = df
    )


def kaggle_paul_stefan_popescu():
  df1 = pd.read_csv("./Data/kaggle_paul_stefan_popescu/train_motion_data.csv", sep=",")
  df2 = pd.read_csv("./Data/kaggle_paul_stefan_popescu/test_motion_data.csv", sep=",")
  
  df = pd.concat([df1, df2])
  
  classes = df["Class"].unique().tolist()

  df["Class"] = [classes.index(k) for k in df["Class"].values]

  classes = { classes.index(k) : k for k in classes}
  
  return DriverBehaviorDataset(
      sensors = {
        "Gyro": "Gyroscope",
        "Acc": "Accelerometer"
      }, 
      target = "Class",
      classes = classes, 
      data = df,
      capital = True
  )


def mendeley_yuksel_atmaca():
  df = pd.read_csv("./Data/mendeley_yuksel_atmaca/sensor_raw.csv", sep=",")
  classes = {
      1 : "Sudden Acceleration",
      2 : "Sudden Right Turn",
      3 : "Sudden Left Turn",
      4 : "Sudden Break"
  }

  return DriverBehaviorDataset(
      sensors = {
        "Gyro": "Gyroscope",
        "Acc": "Accelerometer"
      }, 
      target = "Target(Class)",
      classes = classes, 
      data = df,
      capital = True
  )
