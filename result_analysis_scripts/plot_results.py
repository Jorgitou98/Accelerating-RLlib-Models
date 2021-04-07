import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

def plot_bars(bars, values, name, save_name):
    plt.close('all')
    fig, ax = plt.subplots()
    pos = np.arange(len(bars))
    ax.bar(pos, values)
    plt.xticks(pos, bars)
    ax.set_title(name, loc='center', wrap=True)
    fig.autofmt_xdate()
    if os.path.exists(save_name):
        os.remove(save_name)
    plt.savefig(save_name)
    print('Graph saved at ' + save_name)

def plot_bars_double(bars, values1, values2, name, save_name, label1, label2):
    plt.close('all')
    fig, ax = plt.subplots()
    pos = np.arange(len(bars))
    ax.bar(pos + 0.0, values1, label = label1, width=0.25)
    ax.bar(pos + 0.25, values2, label= label2, width=0.25)
    plt.xticks(pos + 0.125, bars)
    ax.set_title(name, loc='center', wrap=True)
    plt.legend()
    fig.autofmt_xdate()
    if os.path.exists(save_name):
        os.remove(save_name)
    plt.savefig(save_name)
    print('Graph saved at ' + save_name)    

def plot_line_multiple(x_values, y_values, labels, title, save_name, num_plots):
    plt.close('all')
    fig,ax = plt.subplots()
    for i in range(0,num_plots):
        ax.plot(x_values,y_values[i], label = labels[i])
    plt.legend()
    ax.set_title(title, loc = 'center', wrap=True)
    if os.path.exists(save_name):
        os.remove(save_name)
    plt.savefig(save_name)
    print('Graph saved at ' + save_name)

def plot_bars_multiple(bars, values, name, save_name, labels):
    plt.close('all')
    fig, ax =plt.subplots()
    pos =np.arange(len(bars))
    width = 0.8/len(values)
    for i in range (0, len(values)):
        ax.bar(pos + width*i, values[i], label = labels[i], width = width)
    plt.xticks(pos + len(values)*width/2, bars)
    ax.set_title(name, loc='center', wrap=True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), ncol =2)
    fig.autofmt_xdate(rotation=15)  
    if os.path.exists(save_name):
        os.remove(save_name)
    plt.savefig(save_name)
    print('Graph saved at ' + save_name)      
