import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import date
from timesfm import TimesFm
from huggingface_hub import login
import matplotlib.pyplot as plt

# 给定需要处理的股票代码，上海票以.ss结尾，深圳票以.sz结尾
start = date(2020, 1, 1)  # 使用date类创建日期对象
end = date(2024, 1, 1)  # 指定结束日期为2024年1月1日
codelist = ["000001.ss"]

# 增加错误重试机制的下载数据部分
for retry in range(3):  # 尝试下载最多3次
    try:
        data2 = yf.download(codelist, start=start, end=end)
        # 数据预处理
        data2 = data2['Adj Close'].dropna()  # 使用调整结算价格删除缺损值
        if not data2.empty:
            break  # 成功下载并处理数据，跳出循环
    except Exception as e:
        print(f"下载失败，第{retry+1}次尝试。错误：{e}")
        if retry < 2:  # 在最后一次尝试前等待
            time.sleep(5)  # 等待5秒后重试

if data2.empty:
    raise ValueError("数据为空，请更改时间区间再试一次或检查网络连接。")

context_len = 512  # 设置上下文长度
horizon_len = 256  # 设置预测期间的长度

if len(data2) < context_len:
    raise ValueError(f"数据长度小于上下文长度（{context_len}）")

context_data = data2[-context_len:]  # 使用最近512天的数据作为上下文

# 初始化和导入TimesFM模型
tfm = TimesFm(
    context_len=context_len,
    horizon_len=horizon_len,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend='cpu',  # 修改这里，将'gpu'改为'cpu'
)

# 登录Hugging Face Hub，此处****需替换成自己的Hugging token
login("*****")

tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# 准备数据
forecast_input = [context_data.values]
frequency_input = [0]  #设置数据频率（0是高频率数据）

# 运行预测
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

# 设置图表尺寸为24*12英寸
plt.figure(figsize=(24, 12))

forecast_dates = pd.date_range(start=data2.index[-1] + pd.Timedelta(days=1), periods=horizon_len, freq='B')
forecast_series = pd.Series(point_forecast[0], index=forecast_dates)

# 添加部分：获取并绘制2024.1.1到当前时间的实际价格数据
current_date = datetime.datetime.now().date()
data_recent = yf.download(codelist, start=date(2024, 1, 1), end=current_date)

if not data_recent.empty:
    data_recent = data_recent['Adj Close'].dropna()
    plt.plot(data_recent.index, data_recent.values, label="Actual Prices (2024-Now)")

# 创建或更新图表（如果前面已有图表，这里是更新）
plt.plot(data2.index, data2.values, label="Actual Prices")
plt.plot(forecast_series.index, forecast_series.values, label="Forecasted Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"Price Comparison & Forecast for {codelist[0]}")
plt.legend()

# 保存图表到文件，确保尺寸更改已生效
plt.savefig(f'{codelist[0]}_comparison.png', bbox_inches='tight') 

# 显式关闭当前图表
plt.close(fig='all')
