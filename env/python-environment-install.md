# python environment install

## Python安装配置
* Windows一键安装
* Mac OSX/Linux
* 系统自带，可以通过homebrew（Mac OSX）和pyenv安装维护多个版本。
* pip安装第三方库
* pip install package（Mac在sudo后可能要加—user参数）
* 因为GFW影响，可以采用豆瓣源： pip install package -i --trusted-host http://pypi.douban.com/simple
* Windows专用已编译包下载

## Anaconda环境安装和使用
* 安装包下载： https://www.continuum.io/downloads
* 一键安装，别没事挑战自己！
优点:

省时省心： Anaconda通过管理工具包、开发环境、Python版本，大大简化了你的工作流程。不仅可以方便地安装、更新、卸载工具包，而且安装时能自动安装相应的依赖包，同时还能使用不同的虚拟环境隔离不同要求的项目。

分析利器： 在 Anaconda 官网中是这么宣传自己的：适用于企业级大数据分析的Python工具。其包含了720多个数据科学相关的开源包，在数据可视化、机器学习、深度学习等多方面都有涉及。不仅可以做数据分析，甚至可以用在大数据和人工智能领域。

## Anaconda常用命令
* 更新包：conda upgrade –all
* 列出已安装包：conda list
* 安装包：conda install package(=version)
* 删除包：conda remove package
* 搜索包名：conda search xxx

### 常用工具
* spyder集成环境
* Jupyter-notebook网页交互环境
