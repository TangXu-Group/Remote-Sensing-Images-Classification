#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:37:12 2021

@author: lin
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

#labels表示你不同类别的代号，比如这里的demo中有13个类别
# labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
# labels=["agricultural", "airplane", "baseballdiamond", "beach", "buildings", "chaparral" , "denseresidential", 
#                    "forest", "freeway", "golfcourse","harbor", "intersection", "mediumresidential" ,"mobilehomepark",
#                    "overpass", "parkinglot", "river","runway" ,"sparseresidential","storagetanks", "tenniscourt"]

# labels=["Airport", "BareLand", "BaseballField", "Beach", "Bridge", "Center" , "Church", 
#                    "Commerical", "DenseResidential", "Desert","Farmland", "Forest", "Industrial" ,"Meadow",
#                    "MediumResidential", "Mountain", "Park","Parking" ,"Playground","Pond", "Port", "RailwayStation",
#                    "Resort", "River", "School", "SparseResidential", "Square", "Stadium", "StorageTanks", "Viaduct"]

# labels = ["airplane", "airport", "baseball_diamond", "basketball_court","beach","bridge",
#           "chaparral", "church", "circular_farmland", "cloud", "commercial_area", "dense_residential",
#           "desert", "forest", "freeway", "golf_course", "ground_track_field", "harbor", "industrial_area",
#           "intersection", "island", "lake", "meadow", "medium_residential", "mobile_home_park", 
#           "mountain", "overpass", "palace", "parking_plot", "railway", "railway_station", "rectangular_farmland",
#           "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential", "stadium", 
#           "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland"]


def labelandtick(dataset):
    if dataset == 'UC_Merced':
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 
                '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
    elif dataset == 'AID':
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', 
                '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',' 26', '27', '28', '29', '30']
    elif dataset == 'NWPU':
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45']
    tick_marks = np.array(range(len(labels))) + 0.5
    return labels, tick_marks


def plot_confusion_matrix(labels, cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    locations = np.array(range(len(labels)))
    plt.xticks(locations, labels)
    # plt.xticks(locations, labels, rotation=90)
    plt.yticks(locations, labels)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

def plotconfusion(true, prediction, save_path, dataset, num):
    cm = confusion_matrix(true, prediction)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm_normalized)
    if dataset == 'UC_Merced':
        plt.figure(figsize=(12, 8), dpi=120)
    elif dataset == 'AID':
        plt.figure(figsize=(18, 16), dpi=120) 
    elif dataset == 'NWPU':
        plt.figure(figsize=(22, 18), dpi=120)
    
    # y_ind_array = np.arange(len(labels))
    # x_ind_array = np.arange(len(labels))*1.5
    labels, tick_marks = labelandtick(dataset)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.4:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=8, va='center', ha='center')
        elif c <= 0.4 and c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=8, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    plot_confusion_matrix(labels, cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig(save_path+'/'+ dataset + num +'.png', format='png')
    # plt.show()
