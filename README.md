# my_pythonfile
存储了我常用的python脚本

## python第三方库的路径指定

需要在环境的site-packages文件夹下添加一个文件<自定义名字.pth>，其中写入第三方库的路径
例如:
我的环境路径为<E:\Project\pycharm\.venv\env1>
我在<E:\Project\pycharm\.venv\env1\Lib\site-packages>下加入文件<my_path.pth>
其中写入<F:\my_pythonfile>

之后指定使用环境env1后，<F:\my_pythonfile>中的库文件也会被调用