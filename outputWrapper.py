import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d


X_IDXS = np.array([ 0. ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.])

class OutputWrapper:
    def __init__(self, output):
        self.lanes_start_idx = 4955
        self.lanes_end_idx = self.lanes_start_idx + 528
        self.road_start_idx = self.lanes_end_idx + 8
        self.road_end_idx = self.road_start_idx + 264

        self.output = np.array(output[:1])

    def get_lane_points(self):
        lanes_flat = self.output[:,:, self.lanes_start_idx:self.lanes_end_idx].flatten()
        df_lanes = pd.DataFrame(lanes_flat)

        ll_t2 = df_lanes[66:132]
        l_t = df_lanes[132:198]
        r_t = df_lanes[264:330]
        rr_t2 = df_lanes[462:528]

        points_ll_t2 = ll_t2.iloc[lambda x: x.index % 2 == 0]
        points_ll_t2 = pd.concat([points_ll_t2], ignore_index = True)
        
        points_l_t = l_t.iloc[lambda x: x.index % 2 == 0]
        points_l_t = pd.concat([points_l_t], ignore_index = True)

        points_r_t = r_t.iloc[lambda x: x.index % 2 == 0]
        points_r_t = pd.concat([points_r_t], ignore_index = True)

        points_rr_t2 = rr_t2.iloc[lambda x: x.index % 2 == 0]
        points_rr_t2 = pd.concat([points_rr_t2], ignore_index = True)

        return points_ll_t2, points_l_t, points_r_t, points_rr_t2

    def get_road_points(self):
        road_flat = self.output[:,:, self.road_start_idx:self.road_end_idx].flatten()
        df_road = pd.DataFrame(road_flat)

        roadr_t2 = df_road[66:132]
        roadl_t = df_road[132:198]

        points_roadr_t2 = roadr_t2.iloc[lambda x: x.index % 2 == 0]
        roadr_t2 = pd.concat([roadr_t2], ignore_index = True)

        points_roadl_t = roadl_t.iloc[lambda x: x.index % 2 == 0]
        points_roadl_t = pd.concat([points_roadl_t], ignore_index = True)

        return points_roadr_t2, points_roadl_t


class ModelOutputVisualizer:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))

        self.ax1.axis('off')

        self.ax2.set_title("Road lines")
        self.ax2.set_xlabel("red - road lines | green - predicted path | yellow - lane lines")
        self.ax2.set_ylabel("Range")

    def visualize(self, image, model_output):
        points_ll_t2, points_l_t, points_r_t, points_rr_t2 = model_output.get_lane_points()
        points_roadr_t2, points_roadl_t = model_output.get_road_points()
        middle = (points_ll_t2 + points_l_t) / 2

        self.ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        self.ax2.clear()
        self.ax2.plot(middle, X_IDXS, color="g", linewidth=2)
        self.ax2.plot(points_ll_t2, X_IDXS, color="y", linewidth=2)
        self.ax2.plot(points_l_t, X_IDXS, color="y", linewidth=2)
        self.ax2.plot(points_roadr_t2, X_IDXS, color="r", linewidth=2)
        self.ax2.plot(points_roadl_t, X_IDXS, color="r", linewidth=2)

        plt.draw()
        plt.pause(0.001)



class OutputWrapperWithStd:
    def __init__(self, model_output):
        self.model_output = model_output
        self.process_output()

    def process_output(self):
        res = np.array(self.model_output[:1])
        lanes_start_idx = 4955
        lanes_end_idx = lanes_start_idx + 528
        road_start_idx = lanes_end_idx + 8
        road_end_idx = road_start_idx + 264

        lanes = res[:, :, lanes_start_idx:lanes_end_idx]
        lane_road = res[:, :, road_start_idx:road_end_idx]

        lanes_flat = lanes.flatten()
        df_lanes = pd.DataFrame(lanes_flat)

        road_flat = lane_road.flatten()
        df_road = pd.DataFrame(road_flat)

        self.lane_data = {
            'left_lane_t': df_lanes.iloc[132:198],
            'left_lane_t2': df_lanes.iloc[198:264],
            'right_lane_t': df_lanes.iloc[264:330],
            'right_lane_t2': df_lanes.iloc[330:396],
            'road_left_t': df_road.iloc[0:66],
            'road_left_t2': df_road.iloc[66:132],
            'road_right_t': df_road.iloc[132:198],
            'road_right_t2': df_road.iloc[198:264],
        }

    def get_lane_data(self, lane_name):
        if lane_name in self.lane_data:
            points, std_values = self.separate_points_and_std_values(self.lane_data[lane_name])
            return points, std_values
        else:
            raise ValueError("Invalid lane_name. Available options are: left_lane_t, left_lane_t2, right_lane_t, right_lane_t2, road_left_t, road_left_t2, road_right_t, road_right_t2")

    @staticmethod
    def separate_points_and_std_values(df):
        points = df.iloc[lambda x: x.index % 2 == 0]
        std = df.iloc[lambda x: x.index % 2 != 0]
        points = pd.concat([points], ignore_index=True)
        std = pd.concat([std], ignore_index=True)

        return points, std
