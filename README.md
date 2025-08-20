# ERNIE-MCP Project Quick Start Guide

## 📖 Tutorial overview

This project is written to help Wenxin Open Source, and implements the local deployment solution of **ERNIE-4.5-21B-A3B** model through the FastDeploy toolkit, helping developers quickly master the localized deployment of ERNIE 4.5 series models and MCP service access, and experience MCP querying real-time weather.

## 🚀 Get started quickly

### Environmental requirements

- **Python**: 3.12
- **PaddlePaddle**: Latest Version (GPU Version)
- **ERNIEKit**: The latest version
- **GPU**: A single 80GB A/H series GPU with a minimum of 24GB VRAM is recommended
- **System**: Linux/Windows/macOS

This project was developed on a single A800

### Environment configuration

Open a terminal and create a virtual environment under the root directory (install python==3.12 here because the project depends on a higher version of python):

```
conda create -p /home/aistudio/envs/mcp python=3.12
```

One-click virtual environment start:

```
source activate /home/aistudio/envs/mcp/
```

Installing the dependencies related to the project:

```
pip install mcp httpx openai python-dotenv
```

The UV package management tool is required to start the MCP service in the future, and the image can be configured to improve the installation speed

Method: Once in the Conda environment, modify the activation script (e.g. `~/.bashrc` or `~/.zshrc`)

```
echo 'export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"' >> ~/.bashrc

source ~/.bashrc  # Reload configuration
```

```
# Clone project
git clone https://github.com/your-username/Ernie_MCP.git
cd Ernie_MCP

# Install PaddlePaddle GPU version
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Clone the ERNIEKit repository
git clone https://github.com/PaddlePaddle/ERNIE.git -b develop

# Anso ERNIEKit dependence
cd ERNIE
pip install -r requirements/gpu/requirements.txt

# First, please install the aistudio-sdk library
pip install --upgrade aistudio-sdk
# Use aistudio cli to download the model. The 0.3B model does not seem to have a strong understanding of semantics, so we choose the 21B model.
aistudio download --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle --local_dir baidu/ERNIE-4.5-21B-A3B-Paddle

# Anso FastDeploy
python -m pip install fastdeploy-gpu -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-80_90/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### Start Fastdeploy

```
# Terminal operation
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-21B-A3B-Paddle \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

**或**  Run the code cell below

```
#  FastDeploy Complete startup code
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
    
    print("🚀 Start the FastDeploy service...")
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
                    print(f"[log] {line}")
                    
                    if "Loading Weights:" in line and "100%" in line:
                        print("✅ Weight loading completed")
                    elif "Loading Layers:" in line and "100%" in line:
                        print("✅ Layer loading completed")
                    elif "Worker processes are launched" in line:
                        print("✅ Worker process started")
                    elif "Uvicorn running on" in line:
                        print("🎉 Service startup completed！")
                        service_ready = True
                        break
        except Exception as e:
            print(f"Log monitoring errors: {e}")
    
    log_thread = threading.Thread(target=monitor_logs, daemon=True)
    log_thread.start()
    
    start_time = time.time()
    while time.time() - start_time < 120:
        if service_ready:
            break
        if process.poll() is not None:
            print("❌ Process Exit")
            return None
        time.sleep(1)
    
    if not service_ready:
        print("❌ Startup timeout")
        process.terminate()
        return None
    
    print("-" * 50)
    return process

def test_model():
    try:
        import openai
        
        print("🔌 Test model connection...")
        
        client = openai.Client(base_url="http://localhost:8180/v1", api_key="null")
        
        response = client.chat.completions.create(
            model="null",
            messages=[
                {"role": "system", "content": "You are a useful AI assistant。"},
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=50,
            stream=False
        )
        
        print("✅ Model testing successful！")
        print(f"🤖 回复: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Test failure: {e}")
        return False

def check_service():
    try:
        response = requests.get("http://localhost:8180/v1/models", timeout=3)
        return response.status_code == 200
    except:
        return False

def setup_service():

    print("=== ERNIE-4.5-21B-A3B-Paddle Service Startup ===")
    
    if check_service():
        print("✅ Discovering running services")
        if test_model():
            print("🎉 Service is ready！")
            return True
        print("⚠️ Service abnormality, restart")
    
    process = start_fastdeploy()
    
    if process is None:
        print("❌ Startup failure")
        return False
    
    if test_model():
        print("🎊 Startup successful! You can now run the knowledge graph code")
        return True
    else:
        print("❌ Started but failed to connect")
        return False

if __name__ == "__main__" or True:
    setup_service()
```

### Test the model

Run the code cell below or run it in the terminal

```
python model_test.py
```

```
# code cell
import openai
host = "0.0.0.0"
port = "8180"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Introducing Beijing"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

## 🚀Get started quickly

### Code run

Run in the terminal

```
python Weather.py Ernie_Server.py
```

Effect display
![](https://ai-studio-static-online.cdn.bcebos.com/99d4751a50a843da99b1275b8a3a2be167925ddae6cb4ea89696de40ffb5ef94)

![](https://ai-studio-static-online.cdn.bcebos.com/31a325fc4fbb4fd8b2fd3fe7fa149c79515ba5125df64d68b32ad86437baa637)
![](https://ai-studio-static-online.cdn.bcebos.com/9f8f60ea41fb433da0306126370c21afd32757d5ecac452984dc3171c45a26c3)

## 🤝 Contribution Guidelines

We welcome all forms of contributions! Whether it's finding bugs, suggesting improvements, or contributing new tutorial content.

### Contribution method

- 🐛 **Bug Report**: Please submit an [issue](https://github.com/G-Fuji/ERNIE-Tutorial/issues) if you find a problem
- 💡 **Feature suggestions**: If you have a good idea, please discuss it in [Discussions](https://github.com/G-Fuji/ERNIE-Tutorial/discussions)
- 📝 **Documentation Improvements**: Helps refine tutorial content and code comments
- 🔧 **Code Contributions**: Submit a Pull Request to contribute new features or fixes
- 🎓 **Tutorial Expansion**: Contribute new application scenarios and practical cases

### Contribution process

1. Fork this project to your GitHub account
2. Create a feature branch ( `git checkout -b feature/AmazingFeature` )
3. Make your changes and ensure code quality
4. Commit changes ( `git commit -m 'Add some AmazingFeature'` )
5. Push to your branch ( `git push origin feature/AmazingFeature` )
6. Open the Pull Request and describe your changes in detail

### Code specifications

- Follow the PEP 8 Python code specification
- appropriate notes and documentation for new features
- Make sure all sample code works correctly
- Update relevant README and configuration files

## 📄 Permit.

This project is licensed under [the Apache 2.0 license](https://github.com/G-Fuji/ERNIE-Tutorial/blob/main/LICENSE). You are free to use, modify, and distribute the code for this project, but please retain the original license notice.

## 📞 Contact us

- **Project Maintainer**： Ernie_MCP Team
- **Technical Communication**： Technical questions are welcome in Issues

## 🙏 Thanks.

Thanks to the following projects and teams for their support:

- [Baidu PaddlePaddle](https://github.com/PaddlePaddle/Paddle) - Deep Learning Framework
- [ERNIEKit](https://github.com/PaddlePaddle/ERNIE) - Official Model Development Kit
- [ERNIE Model Team](https://aistudio.baidu.com/modelsoverview) - Provides powerful pre-trained models
- Feedback and support from all contributors and users

------

⭐ **If this project helps you, please give us a Star! Your support is the driving force behind our continuous improvement.**

🚀 **Start your ERNIE model learning journey!**

🔗 **Related links**

- [AI STUDIO](https://aistudio.baidu.com/)
- [PaddlePaddle ](https://www.paddlepaddle.org.cn/)
- [ERNIEkit GitHub](https://github.com/PaddlePaddle/ERNIE)

