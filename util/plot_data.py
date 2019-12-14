import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    folder_lists = os.listdir('../output')
    # x_label = 'Wall_Time'
    x_label = 'Iteration'
    y_label_lists = ['Train_Return', 'Test_Return', 'Wall_Time', 'Samples']
    # y_label_lists = ['Iteration', 'Wall_Time', 'Samples', 'Train_Return', 'Test_Return', 'State_Mean', 'State_Std', 'Goal_Mean', 'Goal_Std', 'Exp_Rate', 'Exp_Noise', 'Exp_Temp', 'Critic_Loss', 'Critic_Stepsize', 'Actor_Loss', 'Actor_Stepsize', 'Clip_Frac', 'Adv_Mean', 'Adv_Std']
    flag_show_label = True
    for y_label in y_label_lists:
        data_x = []
        data_y = []
        for folder_name in folder_lists:
            file_name = '../output/' + folder_name + '/agent0_log.txt'
            data = np.genfromtxt(file_name,dtype=None, names=True)
            if flag_show_label:
                print(data.dtype.names)
                flag_show_label = False
            data_x.append(data[x_label])
            data_y.append(data[y_label])

        plot_graph(data_x, data_y, x_label, y_label, x_lim = None, y_lim = None, legend = folder_lists, times_new_roman = False, rc_params = True)

    f_name = None
    if f_name != None:
        plt.savefig(f_name, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()

def plot_graph(x, y, x_label, y_label, x_lim = None, y_lim = None, legend = None, times_new_roman = False, rc_params = None):
    if rc_params:
        if times_new_roman:
            import matplotlib.font_manager as fon
            del fon.weight_dict['roman']
            fon._rebuild()
            plt.rcParams['font.family'] = 'Times New Roman' # font
        else:
            plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['mathtext.default'] = 'regular' # 数式フォント
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.width'] = 1.0 #x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0 #y軸主目盛り線の線幅
        plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['axes.xmargin'] = '0' #'.05'
        plt.rcParams['axes.ymargin'] = '.05'
        plt.rcParams['savefig.facecolor'] = 'None'
        plt.rcParams['savefig.edgecolor'] = 'None'
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = 14 #フォントの大きさ
        plt.rcParams["legend.fancybox"] = False # 丸角
        plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
        plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
        # plt.rcParams["legend.labelspacing"] = 5. # 垂直方向（縦）の距離の各凡例の距離
        # plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
        # plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale

    fig = plt.figure()
    fig_1 = fig.add_subplot(111)

    for num in range(len(x)):
        if legend != None:
            fig_1.plot(x[num], y[num], label = legend[num])
        else:
            fig_1.plot(x[num], y[num])
    fig_1.set_xlabel(x_label)
    fig_1.set_ylabel(y_label)
    if x_lim != None:
        fig_1.set_xlim(x_lim[0], x_lim[1])
    if y_lim != None:
        fig_1.set_ylim(y_lim[0], y_lim[1])
    if legend != None:
        fig_1.legend()

if __name__ == '__main__':
    main()
