import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from timesfm import TimesFm
from huggingface_hub import login
import matplotlib.pyplot as plt

# 日经平均指数的数据取得
# 获取yfinance数据需要翻墙，全局代理的那种
start = datetime.date(2022, 1, 1)
end = datetime.date.today()
codelist = ["^N225"]

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
horizon_len = 128  # 设置预测期间的长度

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

# 登录Hugging Face Hub
login("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")

tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# 准备数据
forecast_input = [context_data.values]
frequency_input = [0]  #设置数据频率（0是高频率数据）

# 运行预测
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

# 查看预测结果
forecast_dates = pd.date_range(start=data2.index[-1] + pd.Timedelta(days=1), periods=horizon_len, freq='B')
forecast_series = pd.Series(point_forecast[0], index=forecast_dates)

plt.figure(figsize=(14, 7))
plt.plot(data2.index, data2.values, label="Actual Prices")
plt.plot(forecast_series.index, forecast_series.values, label="Forecasted Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Price Forecast")  # 添加一个标题，可选
plt.legend()

# 保存图表到文件
plt.savefig('forecast_plot.png')  # 文件名可以根据需要自定义

# 注意：如果你在无图形界面的环境（如服务器）上运行此代码，确保之前设置了后端为'Agg'
plt.switch_backend('Agg')  # 这行代码在需要时取消注释

# 通常在保存图表后，为了避免在无GUI环境中打开图像窗口，可以显式关闭figure
plt.close()
