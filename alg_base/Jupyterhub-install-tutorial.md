# Jupyterhub install tutorial

**一、下载安装依赖：**

Make sure that:
* Python 3.5 or greater
* nodejs/npm environment

执行如下命令操作：

> sudo apt-get install npm nodejs-legacy

> npm install -g configurable-http-proxy

> pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple notebook

> pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyter

**二、安装jupyterhub：**

> python3.5 -m pip install jupyterhub

**三、安装及配置：**

**1. 生成配置文件**

> mkdir /etc/jupyterhub

> cd /etc/jupyterhub

> jupyterhub --generate-config

**2. 配置ip和port**

> vim /etc/jupyterhub/jupyterhub_config.py

> c.JupyterHub.bind_url = 'http://ip_adress:port'


**3. 设置为系统服务，开机自启**

> touch /lib/systemd/system/jupyterhub.service


```
[Unit]
Description=Jupyterhub

[Service]
User=root
Environment="PATH=/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
ExecStart=/usr/local/bin/jupyterhub -f /etc/jupyterhub/jupyterhub_config.py

[Install]
WantedBy=multi-user.target
```
启动：

> service jupyterhub start

**四、 浏览器访问**

http://ip_adress:port/


**五、如何添加python虚拟环境到juperhub**
```
# 进入python虚拟环境
source ~/tensorflow-py3.5/bin/activate

# 安装ipykernel
pip3.5 install ipykernel

# 将tensorflow-py3,5加入到ipykernel
python3.5 -m ipykernel install --user --name=tensorflow-py3,5

# 重启服务使之生效
service jupyterhub restart

```
