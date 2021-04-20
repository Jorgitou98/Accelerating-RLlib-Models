import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

def plot_bars(bars, values, name, save_name, y_label=None):
    plt.close('all')
    fig, ax = plt.subplots()
    pos = np.arange(len(bars))
    rects=ax.bar(pos, values)
    ax.bar_label(rects, padding=1)
    plt.xticks(pos, bars)
    ax.set_title(name, loc='center', wrap=True)
    fig.autofmt_xdate()
    if y_label is not None:
        ax.set_ylabel(y_label)
    fig.tight_layout()
    if os.path.exists(save_name):
        os.remove(save_name)
    plt.savefig(save_name, bbox_inches='tight')
    print('Graph saved at ' + save_name)

def plot_bars_double(bars, values1, values2, name, save_name, label1, label2, y_label=None):
    plt.close('all')
    fig, ax = plt.subplots()
    pos = np.arange(len(bars))
    rects1=ax.bar(pos + 0.0, values1, label = label1, width=0.25)
    rects2=ax.bar(pos + 0.25, values2, label= label2, width=0.25)
    #ax.bar_label(rects1, padding=1)
    #ax.bar_label(rects2, padding=1)
    ax.legend(fontsize=10)
    plt.xticks(pos + 0.125, bars)
    ax.set_title(name, loc='center', wrap=True)
    plt.legend()
    fig.autofmt_xdate()
    if y_label is not None:
        ax.set_ylabel(y_label)
    fig.tight_layout()
    if os.path.exists(save_name):
        os.remove(save_name)
    plt.savefig(save_name, bbox_inches='tight')
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
    fig.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    print('Graph saved at ' + save_name)

def plot_bars_multiple(bars, values, name, save_name, labels,y_label=None):
    plt.close('all')
    fig, ax =plt.subplots()
    pos =np.arange(len(bars))
    width = 0.8/len(values)
    for i in range (0, len(values)):
        rects=ax.bar(pos + width*i, values[i], label = labels[i], width = width)
        #ax.bar_label(rects, padding=1)
    plt.xticks(pos + len(values)*width/2-width/2, bars)
    ax.set_title(name, loc='center', wrap=True)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1), ncol =2)
    plt.legend(loc='best')
    fig.autofmt_xdate(rotation=30)
    if y_label is not None:
        ax.set_ylabel(y_label)
    fig.tight_layout()
    if os.path.exists(save_name):
        os.remove(save_name)
    plt.savefig(save_name, bbox_inches='tight')
    print('Graph saved at ' + save_name)      

def plot_bars_stacked(bars, values, name, save_name, labels, y_label=None):
    plt.close('all')
    fig, ax =plt.subplots()
    pos =np.arange(len(bars))
    ax.bar(pos, values[0], label = labels[0])
    for i in range(1,4):
        ax.bar(pos, values[i], label = labels[i], bottom=[sum([values[j][k] for j in range(i)]) for k in range(len(values[0]))])
    plt.xticks(pos , bars)
    ax.set_title(name, loc='center', wrap=True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.25), ncol =2)
    fig.autofmt_xdate(rotation=15)
    if y_label is not None:
        ax.set_ylabel(y_label)
    fig.tight_layout()
    if os.path.exists(save_name):
        os.remove(save_name)
    plt.savefig(save_name, bbox_inches='tight')
    print('Graph saved at ' + save_name)      
   
