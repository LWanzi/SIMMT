import matplotlib.pyplot as plt

# 定义参数值和对应的NMI值
params = [0.2, 0.4, 0.6, 0.8]
nmi_values = [0.88, 0.92, 0.94, 0.93]

# 创建折线图
plt.figure(figsize=(6,4))
plt.plot(params, nmi_values, marker='o', color='red', linestyle='-', label='NMI')

# 添加标题和标签
plt.title('NMI values for different parameter values')
plt.xlabel('Parameter Value')
plt.ylabel('NMI Value')

# 仅显示横网格线
plt.grid(True, axis='y')
plt.legend()

# 显示图表
plt.show()
