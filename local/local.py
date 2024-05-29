import datetime
import pandas as pd
import matplotlib.pyplot as plt
from timesfm import TimesFm
from huggingface_hub import login

# 从当前文件夹的CSV文件中读取数据
data = pd.read_csv('sh999999.csv', parse_dates=['Date'], index_col='Date')
data2 = data['Close'].dropna()  # 使用收盘价并删除缺损值

if data2.empty:
    raise ValueError("数据为空，请检查CSV文件。")

context_len = 512  # 设置上下文长度
horizon_len = 256  # 设置预测期间的长度

if len(data2) < context_len:
    raise ValueError(f"数据长度小于上下文长度（{context_len}）")

context_data = data2[-context_len:]  # 使用最近512天的数据作为上下文

# 登录Hugging Face Hub并添加到Git凭证助手
try:
    login("******", add_to_git_credential=True)
except Exception as e:
    print(f"登录Hugging Face Hub时发生错误: {e}")

# 尝试从Hugging Face Hub加载模型，增加错误处理
try:
    tfm = TimesFm(
        context_len=context_len,
        horizon_len=horizon_len,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend='cpu'
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
except Exception as e:
    print(f"加载模型时发生错误: {e}")

# 准备数据
forecast_input = [context_data.values]
frequency_input = [0]  # 设置数据频率（0是高频率数据）

# 运行预测
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

# 设置图表尺寸为24*12英寸
plt.figure(figsize=(24, 12))

forecast_dates = pd.date_range(start=date(2024, 1, 1), periods=horizon_len, freq='B')
forecast_series = pd.Series(point_forecast[0], index=forecast_dates)

# 获取并绘制2024.1.1到当前时间的实际价格数据
current_date = datetime.datetime.now().date()
data_recent = data['2024-01-01':current_date]['Close'].dropna()

if not data_recent.empty:
    plt.plot(data_recent.index, data_recent.values, label="Actual Prices (2024-Now)")

# 创建或更新图表（如果前面已有图表，这里是更新）
plt.plot(data2.index, data2.values, label="Historical Prices")
plt.plot(forecast_series.index, forecast_series.values, label="Forecasted Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Price Comparison & Forecast")
plt.legend()

# 保存图表到文件，确保尺寸更改已生效
plt.savefig('comparison.png', bbox_inches='tight') 

# 显式关闭当前图表
plt.close(fig='all')

# 使用本地数据进行预测和回溯对比
# 解决Hugging的警告提示