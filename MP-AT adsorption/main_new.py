
from PyQt5 import QtWidgets
from ui_main_new import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer  # 导入均值填充方法

class myWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.resize(1330,830)
        #Qm
        self.pushButton.clicked.connect(self.pre_qm)
        #logKL
        self.pushButton_2.clicked.connect(self.pre_logKL)
        '''======================================'''
        # # logQmax
        # self.pushButton_3.clicked.connect(self.pre_logQmax)
        # # K
        # self.pushButton_4.clicked.connect(self.pre_KL)
        '''======================================'''


        # # Qmax
        self.pushButton_5.clicked.connect(self.pre_Qmax)
        #
        self.list = [10, 20, 40, 60, 80, 100, 120, 140]
        self.ce_list = []
        self.new_ce_list = []
        self.current_index = 0  # 用于追踪当前处理的列表索引

        self.pushButton_6.clicked.connect(self.start_plotting)
        self.Qmax = None
        self.KL = None
        self.fig = None
        self.ax = None
        self.canvas = None

        # 初始化绘图
        self.init_plot()

    def init_plot(self):
        # 初始化 Figure 和 Axes
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.horizontalLayout_10.addWidget(self.canvas)

    def start_plotting(self):
        # 开始处理所有点
        self.process_all_points()

    def process_all_points(self):
        # 处理列表中的所有点，每次处理一个
        while self.current_index < len(self.list):
            self.add_point_calc()



    def add_point_calc(self):
        try:
            point = float(self.list[self.current_index])
            self.ce_list.append(point)
            # 计算纵坐标列表
            i_new = self.Qmax * self.KL * point / (1 + self.KL * point)
            self.new_ce_list.append(i_new)
            # 使用matplotlib绘制曲线图
            self.draw_curve()
            # 更新索引以处理列表中的下一个点
            self.current_index += 1
        except Exception as e:
            print(e)

    def draw_curve(self):
        # 检查 self.new_ce_list 是否为空
        if not self.new_ce_list:
            print("No data to plot.")
            return
            # 如果 Figure 和 Axes 对象还没有被初始化，则创建它们

        # 使用 self.new_ce_list 的索引作为 x 轴数据
        x = np.array(self.ce_list)
        # 使用 self.new_ce_list 作为 y 轴数据
        y = np.array(self.new_ce_list)
        # 清除之前的图形（如果有的话）
        self.ax.clear()
        # Plot data
        self.ax.plot(x, y, marker='o')  # 添加标记以便更容易看到点
        self.ax.set_title('')
        self.ax.set_xlabel('Ce/(mg/L))')
        self.ax.set_ylabel('Qe/(mg/g)')
        # 在每个点旁边添加坐标文本
        for i in range(len(x)):
            # 确保 x[i] 和 y[i] 是数值而不是数组
            xi = x[i].item() if hasattr(x[i], 'item') else x[i]
            yi = y[i].item() if hasattr(y[i], 'item') else y[i]
            # 设定文本的位置，这里稍微偏移以避免覆盖点
            text_x = xi
            text_y = yi
            if yi >= 0:
                text_y += 0.01  # 在y轴正值时，稍微向上偏移
            else:
                text_y -= 0.01  # 在y轴负值时，稍微向下偏移
            # 添加文本（使用 .2f 格式化数值）
            # self.ax.text(text_x, text_y, f'({xi:.2f}, {yi:.2f})', fontsize=8)
            # 重新绘制图形
        self.canvas.draw()


    def pre_Qmax(self):
        self.particle_size = self.doubleSpinBox.value()
        self.age = self.doubleSpinBox_2.value()
        self.SSA = self.doubleSpinBox_3.value()
        self.Mw = self.doubleSpinBox_4.value()
        self.LogKw= self.doubleSpinBox_5.value()
        self.pka2= self.doubleSpinBox_6.value()
        self.solubility = self.doubleSpinBox_7.value()
        self.MP_con = self.doubleSpinBox_8.value()
        self.AT_con = self.doubleSpinBox_9.value()
        self.temeraputre = self.doubleSpinBox_10.value()
        self.PH = self.doubleSpinBox_11.value()
        X_test_list = [self.particle_size,self.SSA,self.age,self.Mw,self.LogKw,self.pka2,self.solubility,self.MP_con,self.temeraputre,self.PH]
        X_test_input = pd.DataFrame(X_test_list, columns=['Column1'])
        X_test_input = X_test_input.T
        result = self.train_pred(path='logQmax.csv',tezhenbianliang='logQmax',X_test_input=X_test_input)
        result = 10**result
        self.lineEdit_5.setText(f'{result[0]:.5f}')
        self.Qmax = result
        self.pre_logQmax()

    def pre_KL(self):
        self.particle_size = self.doubleSpinBox.value()
        self.age = self.doubleSpinBox_2.value()
        self.SSA = self.doubleSpinBox_3.value()
        self.Mw = self.doubleSpinBox_4.value()
        self.LogKw= self.doubleSpinBox_5.value()
        self.pka2= self.doubleSpinBox_6.value()
        self.solubility = self.doubleSpinBox_7.value()
        self.MP_con = self.doubleSpinBox_8.value()
        self.AT_con = self.doubleSpinBox_9.value()
        self.temeraputre = self.doubleSpinBox_10.value()
        self.PH = self.doubleSpinBox_11.value()
        X_test_list = [self.particle_size,self.SSA,self.age,self.Mw,self.LogKw,self.pka2,self.solubility,self.MP_con,self.temeraputre,self.PH]
        X_test_input = pd.DataFrame(X_test_list, columns=['Column1'])
        X_test_input = X_test_input.T
        result = self.train_pred(path='logKl.csv',tezhenbianliang='logKl',X_test_input=X_test_input)
        result = 10**result
        # self.lineEdit_4.setText(f'{result[0]:.5f}')
        self.KL = result

    def pre_logQmax(self):
        self.particle_size = self.doubleSpinBox.value()
        self.age = self.doubleSpinBox_2.value()
        self.SSA = self.doubleSpinBox_3.value()
        self.Mw = self.doubleSpinBox_4.value()
        self.LogKw= self.doubleSpinBox_5.value()
        self.pka2= self.doubleSpinBox_6.value()
        self.solubility = self.doubleSpinBox_7.value()
        self.MP_con = self.doubleSpinBox_8.value()
        self.AT_con = self.doubleSpinBox_9.value()
        self.temeraputre = self.doubleSpinBox_10.value()
        self.PH = self.doubleSpinBox_11.value()
        X_test_list = [self.particle_size,self.SSA,self.age,self.Mw,self.LogKw,self.pka2,self.solubility,self.MP_con,self.temeraputre,self.PH]
        X_test_input = pd.DataFrame(X_test_list, columns=['Column1'])
        X_test_input = X_test_input.T
        result = self.train_pred(path='logQmax.csv',tezhenbianliang='logQmax',X_test_input=X_test_input)
        # self.lineEdit_3.setText(f'{result[0]:.5f}')

    def pre_qm(self):
        self.particle_size = self.doubleSpinBox.value()
        self.age = self.doubleSpinBox_2.value()
        self.SSA = self.doubleSpinBox_3.value()
        self.Mw = self.doubleSpinBox_4.value()
        self.LogKw= self.doubleSpinBox_5.value()
        self.pka2= self.doubleSpinBox_6.value()
        self.solubility = self.doubleSpinBox_7.value()
        self.MP_con = self.doubleSpinBox_8.value()
        self.AT_con = self.doubleSpinBox_9.value()
        self.temeraputre = self.doubleSpinBox_10.value()
        self.PH = self.doubleSpinBox_11.value()
        X_test_list = [self.particle_size,self.age,self.SSA,self.Mw,self.LogKw,self.pka2,self.solubility,self.MP_con,self.AT_con,self.temeraputre,self.PH]
        X_test_input = pd.DataFrame(X_test_list, columns=['Column1'])
        X_test_input = X_test_input.T
        result = self.train_pred(path='Qm.csv',tezhenbianliang='Qm',X_test_input=X_test_input)
        self.lineEdit.setText(f'{result[0]:.5f}')


    def pre_logKL(self):
        self.particle_size = self.doubleSpinBox.value()
        self.age = self.doubleSpinBox_2.value()
        self.SSA = self.doubleSpinBox_3.value()
        self.Mw = self.doubleSpinBox_4.value()
        self.LogKw= self.doubleSpinBox_5.value()
        self.pka2= self.doubleSpinBox_6.value()
        self.solubility = self.doubleSpinBox_7.value()
        self.MP_con = self.doubleSpinBox_8.value()
        self.AT_con = self.doubleSpinBox_9.value()
        self.temeraputre = self.doubleSpinBox_10.value()
        self.PH = self.doubleSpinBox_11.value()
        X_test_list = [self.particle_size,self.SSA,self.age,self.Mw,self.LogKw,self.pka2,self.solubility,self.MP_con,self.temeraputre,self.PH]
        X_test_input = pd.DataFrame(X_test_list, columns=['Column1'])
        X_test_input = X_test_input.T
        result = self.train_pred(path='logKl.csv',tezhenbianliang='logKl',X_test_input=X_test_input)
        self.lineEdit_2.setText(f'{result[0]:.5f}')
        self.pre_KL()

    def train_pred(self,path, tezhenbianliang, X_test_input):
        # 读取数据
        # path = '"Qm.csv"'
        df = pd.read_csv(path)
        # 输出数据前几行查看
        print(df.head())
        # 检查缺失值数量
        print(df.isnull().sum())
        # 使用均值填充缺失值
        imputer = SimpleImputer(strategy='mean')
        imputed = imputer.fit_transform(df)
        data = pd.DataFrame(imputed, columns=df.columns, index=df.index)
        # 再次检查是否所有缺失值都已填充
        print(data.isnull().sum())
        # 划分特征和目标变量
        # X = data.drop(['Qm'], axis=1)
        # y = data['Qm']
        X = data.drop([tezhenbianliang], axis=1)
        y = data[tezhenbianliang]

        # 将数据集划分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 创建随机森林回归模型
        rf_regressor = RandomForestRegressor(n_estimators=500, random_state=42)
        # 在训练集上训练模型
        rf_regressor.fit(X_train, y_train)
        # 在测试集上进行预测
        y_pred = rf_regressor.predict(X_test_input)
        print(y_pred)
        return y_pred






    def decreasing_slope_function(self, x):
        # Example function where slope decreases with x
        return np.log1p(x)  # You can modify this function as needed

if __name__=="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    test=myWindow()
    test.show()
    sys.exit(app.exec_())