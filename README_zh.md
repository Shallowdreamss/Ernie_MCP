# ERNIE-MCP 项目快速入门指南

## 📖 教程概览

本项目为助力文心开源所撰写，通过FastDeploy工具包实现**ERNIE-4.5-21B-A3B**模型本地部署方案，助力开发者快速掌握ERNIE 4.5系列模型的本地化部署及MCP服务接入，体验MCP查询实时天气。

## 🚀 快速开始

### 环境要求

- **Python**: 3.12
- **PaddlePaddle**: 最新版本 (GPU版本)
- **ERNIEKit**: 最新版本
- **GPU**: 推荐单张 80GB A/H 系列GPU，最低 24GB 显存
- **系统**: Linux/Windows/macOS

本项目在单张A800上开发

### 环境配置

打开一个终端，在根目录下创建一个虚拟环境（因为项目依赖高版本python，所以此处安装python==3.12）：

```
conda create -p /home/aistudio/envs/mcp python=3.12
```

一键启动虚拟环境：

```
source activate /home/aistudio/envs/mcp/
```

安装该项目相关依赖：

```
pip install mcp httpx openai python-dotenv
```

后续启动 MCP 服务时需要 uv 包管理工具，为了提升安装速度，可以配置镜像

方法：进入 Conda 环境后，修改激活脚本（如 `~/.bashrc` 或 `~/.zshrc`）

```
echo 'export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"' >> ~/.bashrc

source ~/.bashrc  # 重新加载配置
```

```
# 克隆项目
git clone https://github.com/your-username/Ernie_MCP.git
cd Ernie_MCP

# 安装 PaddlePaddle GPU 版本
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 克隆 ERNIEKit 仓库
git clone https://github.com/PaddlePaddle/ERNIE.git -b develop

# 安装 ERNIEKit 依赖
cd ERNIE
pip install -r requirements/gpu/requirements.txt

# 首先请先安装aistudio-sdk库
pip install --upgrade aistudio-sdk
# 使用aistudio cli下载模型，0.3B的模型对语意的理解貌似不是很强，所以我们选择21B模型
aistudio download --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle --local_dir baidu/ERNIE-4.5-21B-A3B-Paddle

# 安装FastDeploy
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 启动Fastdeploy

```
# 终端运行
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-21B-A3B-Paddle \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

**或**  运行下面的代码块

```
#  FastDeploy完整启动代码
import subprocess
import time
import requests
import threading

def start_fastdeploy():
    cmd = [
        "python", "-m", "fastdeploy.entrypoints.openai.api_server",
        "--model", "baidu/ERNIE-4.5-21B-A3B-Paddle",
        "--port", "8180",
        "--metrics-port", "8181", 
        "--engine-worker-queue-port", "8182",
        "--max-model-len", "32768",
        "--max-num-seqs", "32"
    ]
    
    print("🚀 启动FastDeploy服务...")
    print("-" * 50)
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print(f"📝 PID: {process.pid}")
    
    service_ready = False
    
    def monitor_logs():
        nonlocal service_ready
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(f"[日志] {line}")
                    
                    if "Loading Weights:" in line and "100%" in line:
                        print("✅ 权重加载完成")
                    elif "Loading Layers:" in line and "100%" in line:
                        print("✅ 层加载完成")
                    elif "Worker processes are launched" in line:
                        print("✅ 工作进程启动")
                    elif "Uvicorn running on" in line:
                        print("🎉 服务启动完成！")
                        service_ready = True
                        break
        except Exception as e:
            print(f"日志监控错误: {e}")
    
    log_thread = threading.Thread(target=monitor_logs, daemon=True)
    log_thread.start()
    
    start_time = time.time()
    while time.time() - start_time < 120:
        if service_ready:
            break
        if process.poll() is not None:
            print("❌ 进程退出")
            return None
        time.sleep(1)
    
    if not service_ready:
        print("❌ 启动超时")
        process.terminate()
        return None
    
    print("-" * 50)
    return process

def test_model():
    try:
        import openai
        
        print("🔌 测试模型连接...")
        
        client = openai.Client(base_url="http://localhost:8180/v1", api_key="null")
        
        response = client.chat.completions.create(
            model="null",
            messages=[
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": "你好"}
            ],
            max_tokens=50,
            stream=False
        )
        
        print("✅ 模型测试成功！")
        print(f"🤖 回复: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def check_service():
    try:
        response = requests.get("http://localhost:8180/v1/models", timeout=3)
        return response.status_code == 200
    except:
        return False

def setup_service():

    print("=== ERNIE-4.5-21B-A3B-Paddle 服务启动 ===")
    
    if check_service():
        print("✅ 发现运行中的服务")
        if test_model():
            print("🎉 服务已就绪！")
            return True
        print("⚠️ 服务异常，重新启动")
    
    process = start_fastdeploy()
    
    if process is None:
        print("❌ 启动失败")
        return False
    
    if test_model():
        print("🎊 启动成功！现在可以运行知识图谱代码")
        return True
    else:
        print("❌ 启动但连接失败")
        return False

if __name__ == "__main__" or True:
    setup_service()
```

### 测试模型

运行下面的代码块或者在终端运行

```
python model_test.py
```

```
# 代码块
import openai
host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "你是一个助手."},
        {"role": "user", "content": "介绍一下北京"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

## 🚀快速上手

### 代码运行

在终端中运行

```
python Weather.py Ernie_Server.py
```

效果展示
![](https://ai-studio-static-online.cdn.bcebos.com/99d4751a50a843da99b1275b8a3a2be167925ddae6cb4ea89696de40ffb5ef94)
```
python Weather_zh.py Ernie_Server_zh.py
```
![](https://ai-studio-static-online.cdn.bcebos.com/31a325fc4fbb4fd8b2fd3fe7fa149c79515ba5125df64d68b32ad86437baa637)
![](https://ai-studio-static-online.cdn.bcebos.com/9f8f60ea41fb433da0306126370c21afd32757d5ecac452984dc3171c45a26c3)

## 🤝 贡献指南

我们欢迎所有形式的贡献！无论是发现bug、提出改进建议，还是贡献新的教程内容。

### 贡献方式

- 🐛 **Bug 报告**： 发现问题请提交 [Issue](https://github.com/G-Fuji/ERNIE-Tutorial/issues)
- 💡 **功能建议**： 有好想法请在 [Discussions](https://github.com/G-Fuji/ERNIE-Tutorial/discussions) 中讨论
- 📝 **文档改进**： 帮助完善教程内容和代码注释
- 🔧 **代码贡献**： 提交 Pull Request 贡献新功能或修复
- 🎓 **教程扩展**： 贡献新的应用场景和实战案例

### 贡献流程

1. Fork 本项目到您的 GitHub 账户
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 进行您的修改并确保代码质量
4. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
5. 推送到您的分支 (`git push origin feature/AmazingFeature`)
6. 开启 Pull Request 并详细描述您的更改

### 代码规范

- 遵循 PEP 8 Python 代码规范
- 为新功能添加适当的注释和文档
- 确保所有示例代码都能正常运行
- 更新相关的 README 和配置文件

## 📄 许可证

本项目采用 [Apache 2.0 许可证](https://github.com/G-Fuji/ERNIE-Tutorial/blob/main/LICENSE)。您可以自由使用、修改和分发本项目的代码，但请保留原始许可证声明。

## 📞 联系我们

- **项目维护者**： Ernie_MCP Team
- **技术交流**： 欢迎在 Issues 中提出技术问题

## 🙏 致谢

感谢以下项目和团队的支持：

- [百度 PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - 深度学习框架
- [ERNIEKit](https://github.com/PaddlePaddle/ERNIE) - 官方模型开发套件
- [ERNIE 模型团队](https://aistudio.baidu.com/modelsoverview) - 提供强大的预训练模型
- 所有贡献者和使用者的反馈与支持

------

⭐ **如果这个项目对您有帮助，请给我们一个 Star！您的支持是我们持续改进的动力。**

🚀 **开始您的 ERNIE 大模型学习之旅吧！**

🔗 **相关链接**

- [人工智能工作室](https://aistudio.baidu.com/)
- [PaddlePaddle 官网](https://www.paddlepaddle.org.cn/)
- [ERNIEkit GitHub](https://github.com/PaddlePaddle/ERNIE)



