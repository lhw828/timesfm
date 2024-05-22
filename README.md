### 说明
- Google 的 timesfm 已经发布很久了，翻了官方的说明文档，搜了很多资料，但是都是高手在尝鲜，很多只给了结果，具体怎么部署，没有一个可以直接用的。
- 自己摸索了好几天，踩了很多坑，今天终于跑通了，就把过程记录一下。
- 仅供新手参考，有砖轻拍。
- 主要参考：https://qiita.com/syun88/items/9355dff0e68d806257b2
- 修改点：
	- 修改注释为中文
	- 将程序从GPU切换到CPU，没有可用的显卡
	- 增加下载错误重试机制
	- 修改程序，保存结果到文件
- 最开始为了方便，准备在Windows下跑，安装conda，各种折腾，最后和大部分人一样倒在 [此处](https://github.com/google-research/timesfm/issues/24#issuecomment-2118539951 "此处")，随即切换到Ubuntu。
- 再次声明，仅仅是做一个记录
### 程序说明
- timesfm.py为来自syun88的原始demo，将注释改为中文，切换至CPU模式，将结果保存到文件。
- timesfmhs.py在timesfm.py的基础上加入对比功能，默认是获取股票2020年1月1日至2024年1月1日的日线数据，向后预测256日，并将2024年1月1日到当前日期的实际数据和程序预测的数据叠加在同一个图上，进行回溯对比。
### 步骤
- 1、部署环境
	- 安装conda
`wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh`

`sh Anaconda3-2024.02-1-Linux-x86_64.sh`
- 2、 配置conda
编辑`~/.bashrc`在.bashrc文件末尾添加：`export PATH="/root/anaconda3/bin:$PATH"`
	- 修改后执行
`source ~/.bashrc`
- 2、下载timesfm

`wget https://github.com/google-research/timesfm/archive/refs/heads/master.zip`

`unzip timesfm-master.zip`

`mv timesfm-master timesfm`

`cd timesfm`
- 3、选择CPU分支
`conda env create --file=environment_cpu.yml`
- 4、切换到tfm_env
`conda activate tfm_env`
- 5、安装依赖
`pip install -e .`
- 6、安装yfinance
`conda install -c conda-forge yfinance`
- 7、获取huggingface Access Tokens
到 https://huggingface.co/settings/tokens 新建一个WRITE属性的Access Tokens
在代码中设置huggingface登录动作
`login("从网站复制Tokens到此处")`
设置好才能正常获取 `google/timesfm-1.0-200m`
- 8、开始预测
`python timesfm.py`
- 9、有可能用得上的
	- 退出conda环境：`conda deactivate`
	- 删除当前环境（tfm_env）：`conda env remove --name tfm_env`
	- 使用官方脚本重建环境：`conda env create --file=environment_cpu.yml`
	- 激活新建的环境：`conda activate tfm_env`
	- 为新环境安装依赖：`pip install -e .`
### 特别提示
使用yfinance数据需要魔法，不会魔法的用户请注意，一定要全局代理，否则获取数据会失败导致程序无法执行。
