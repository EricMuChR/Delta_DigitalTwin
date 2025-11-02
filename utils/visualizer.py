import matplotlib.pyplot as plt
import numpy as np

def plot_error_comparison(error_before, error_after, output_path):
    error_before_mm = error_before * 1000
    error_after_mm = error_after * 1000
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('机器人定位精度补偿效果验证', fontsize=16)

    ax1.plot(error_before_mm, 'o-', label=f'补偿前 (平均: {np.mean(error_before_mm):.3f} mm)', alpha=0.7)
    ax1.plot(error_after_mm, 'o-', label=f'补偿后 (平均: {np.mean(error_after_mm):.3f} mm)', alpha=0.7)
    ax1.set_title('各验证点绝对误差对比')
    ax1.set_xlabel('验证点序号')
    ax1.set_ylabel('绝对定位误差 (mm)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    labels = ['最大误差', '平均误差', '误差标准差']
    stats_before = [np.max(error_before_mm), np.mean(error_before_mm), np.std(error_before_mm)]
    stats_after = [np.max(error_after_mm), np.mean(error_after_mm), np.std(error_after_mm)]

    x = np.arange(len(labels))
    width = 0.35
    rects1 = ax2.bar(x - width/2, stats_before, width, label='补偿前', color='royalblue')
    rects2 = ax2.bar(x + width/2, stats_after, width, label='补偿后', color='limegreen')

    ax2.set_ylabel('误差值 (mm)')
    ax2.set_title('误差统计指标对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1, ax2)
    autolabel(rects2, ax2)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_path, dpi=300)
    print(f"验证结果图表已保存至: {output_path}")
    plt.close()