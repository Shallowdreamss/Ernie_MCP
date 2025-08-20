import asyncio
import os
import re
import json
import sys
from typing import Optional, Tuple, List, Dict
from contextlib import AsyncExitStack
from datetime import datetime

from openai import OpenAI
import openai
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from city import *

# 加载 .env 文件
load_dotenv()

# 本地模型配置
LOCAL_MODEL_HOST = "0.0.0.0"
LOCAL_MODEL_PORT = "8180"
LOCAL_MODEL_API_KEY = "null"
LOCAL_MODEL_NAME = "null"

class DialogueMemory:
    """对话上下文记忆类"""
    def __init__(self):
        self.history: List[Dict] = []
        self.max_history_length = 5  # 保留最近5轮对话

    def add_message(self, role: str, content: str):
        """添加对话消息"""
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        self.history.append(message)
        
        # 限制历史记录长度
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

    def get_recent_context(self) -> str:
        """获取最近的对话上下文"""
        if not self.history:
            return ""
        
        # 获取最近的用户查询和助手响应
        context_messages = []
        for i in range(len(self.history)-1, max(-1, len(self.history)-6), -2):
            if i >= 0 and self.history[i]['role'] == 'user':
                context_messages.append(f"用户: {self.history[i]['content']}")
            if i-1 >= 0 and self.history[i-1]['role'] == 'assistant':
                context_messages.append(f"助手: {self.history[i-1]['content']}")
        
        return "\n\n".join(reversed(context_messages)) if context_messages else ""

    def clear(self):
        """清空对话历史"""
        self.history = []

def translate_city(city: str) -> str:
    """将中文城市名称转换为英文名称"""
    if city in CITY_MAP:
        return CITY_MAP[city]
    
    for cn_city, en_city in CITY_MAP.items():
        if city.startswith(cn_city) or cn_city in city:
            return en_city
    
    try:
        import pypinyin
        pinyin = pypinyin.lazy_pinyin(city)
        return ''.join(pinyin).capitalize()
    except ImportError:
        return city

async def is_weather_query(query: str) -> bool:
    """判断查询是否为天气相关"""
    weather_keywords = ["天气", "温度", "气温", "湿度", "下雨", "下雪", "晴天", "雨天", "多云", "风力"]
    return any(keyword in query for keyword in weather_keywords)

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        
        # 本地模型客户端（用于决策）
        self.local_client = openai.Client(
            base_url=f"http://{LOCAL_MODEL_HOST}:{LOCAL_MODEL_PORT}/v1",
            api_key=LOCAL_MODEL_API_KEY
        )
        
        # 对话记忆
        self.dialogue_memory = DialogueMemory()
        
        self.session: Optional[ClientSession] = None
        self.tools_available = False

    async def connect_to_server(self, server_script_path: str):
        """连接到MCP服务器"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是.py或.js文件")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # 检查工具是否可用
        response = await self.session.list_tools()
        tools = {tool.name: tool for tool in response.tools}
        self.tools_available = "query_weather" in tools
        print("\n已连接到服务器，天气查询工具" + ("可用" if self.tools_available else "不可用"))
        
        return tools

    async def extract_city(self, query: str) -> Optional[str]:
        """从查询中提取城市名称"""
        # 移除前缀词
        query = query.replace("你", "").replace("请问", "").replace("知道", "").strip()
        
        # 尝试匹配标准格式
        city_match = re.search(r"(.+?)(?:现在|今天|的)天气", query)
        if city_match:
            return city_match.group(1).strip()
        
        # 尝试匹配更灵活的格式
        city_match = re.search(r"((?:北京|上海|广州|深圳|天津|重庆|成都|杭州|武汉|西安|苏州|郑州|南京|青岛|沈阳|大连|厦门|福州|长沙|哈尔滨|济南|长春|石家庄|合肥|太原|南昌|南宁|昆明|贵阳|海口|乌鲁木齐|呼和浩特|银川|西宁|兰州|拉萨|宁波|温州|无锡|常州|南通|徐州|扬州|盐城|泰州|镇江|淮安|连云港|宿迁|嘉兴|湖州|绍兴|金华|衢州|舟山|台州|丽水|芜湖|马鞍山|安庆|滁州|阜阳|宿州|六安|亳州|池州|宣城|漳州|南平|三明|龙岩|宁德|景德镇|萍乡|九江|新余|鹰潭|赣州|吉安|宜春|抚州|上饶|淄博|枣庄|东营|烟台|潍坊|济宁|泰安|威海|日照|临沂|德州|聊城|滨州|菏泽|开封|洛阳|平顶山|安阳|鹤壁|新乡|焦作|濮阳|许昌|漯河|三门峡|南阳|商丘|信阳|周口|驻马店|黄石|十堰|宜昌|襄阳|鄂州|荆门|孝感|荆州|黄冈|咸宁|随州|恩施|株洲|湘潭|衡阳|邵阳|岳阳|常德|张家界|益阳|郴州|永州|怀化|娄底|湘西|韶关|珠海|汕头|佛山|江门|湛江|茂名|肇庆|惠州|梅州|汕尾|河源|阳江|清远|东莞|中山|潮州|揭阳|云浮|柳州|桂林|梧州|北海|防城港|钦州|贵港|玉林|百色|贺州|河池|来宾|崇左|儋州|自贡|攀枝花|泸州|德阳|绵阳|广元|遂宁|内江|乐山|南充|眉山|宜宾|广安|达州|雅安|巴中|资阳|阿坝|甘孜|凉山|六盘水|遵义|安顺|毕节|铜仁|黔东南|黔南|黔西南|曲靖|玉溪|保山|昭通|丽江|普洱|临沧|楚雄|红河|文山|西双版纳|大理|德宏|怒江|迪庆|昌都|林芝|山南|那曲|阿里|铜川|宝鸡|咸阳|渭南|延安|汉中|榆林|安康|商洛|嘉峪关|金昌|白银|天水|武威|张掖|平凉|酒泉|庆阳|定西|陇南|临夏|甘南|海东|海北|黄南|海南|果洛|玉树|海西|石嘴山|吴忠|固原|中卫|克拉玛依|吐鲁番|哈密|昌吉|博尔塔拉|巴音郭楞|阿克苏|克孜勒苏|喀什|和田|伊犁|塔城|阿勒泰|石河子|阿拉尔|图木舒克|五家渠|北屯|铁门关|双河|可克达拉|昆玉|胡杨河|新星)[市省]?)", query)
        if city_match:
            return city_match.group(1).strip()
        
        # 尝试提取省级行政区
        province_match = re.search(r"((?:北京|天津|上海|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门)[省市自治区]?)", query)
        if province_match:
            return province_match.group(1).strip()
        
        return None

    async def should_call_tool(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        决策是否应该调用工具
        :return: (是否调用工具, 提取的城市名称)
        """
        # 先进行简单的关键词匹配
        if await is_weather_query(query):
            city = await self.extract_city(query)
            return (city is not None, city)
        
        # 如果简单匹配失败，使用本地模型进行更智能的判断
        system_prompt = """
        你是一个帮助判断用户意图的助手。请判断用户是否在询问天气信息。
        如果是，提取其中的城市名称；如果不是，返回None。
        
        判断原则：
        1. 如果查询明显是询问天气（包含"天气"、"温度"、"湿度"、"下雨"等词），且指定了城市，则应该调用天气查询工具
        2. 如果查询是闲聊、一般性问题或不需要外部数据的请求，则不需要调用工具
        3. 如果查询格式不符合标准天气查询格式（如"北京现在天气如何"），但意图明显是查询天气，仍应调用工具
        4. 优先使用工具获取实时数据，而不是依赖内置知识
        
        当前对话上下文：
        {context}
        
        请根据以上原则，判断以下查询是否需要调用工具（回答"是"或"否"），如果需要调用工具，请提取城市名称：
        """
        
        # 获取对话上下文
        context = self.dialogue_memory.get_recent_context()
        system_prompt_with_context = system_prompt.format(context=context if context else "无")
        
        messages = [
            {"role": "system", "content": system_prompt_with_context},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.local_client.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=messages,
                temperature=0,
                max_tokens=50,
                stream=False
            )
            
            decision = response.choices[0].message.content.strip()
            print(f"🤖 决策结果: {decision}")
            
            if "否" in decision or "不需要" in decision or "None" in decision:
                return (False, None)
            else:
                # 尝试从决策结果中提取城市名称
                city_match = re.search(r"城市[:：]\s*(\S+)", decision)
                if city_match:
                    return (True, city_match.group(1))
                
                # 尝试从原始查询中提取城市
                city = await self.extract_city(query)
                return (city is not None, city)
                
        except Exception as e:
            print(f"🚨 决策模型调用失败: {str(e)}")
            return (False, None)

    async def call_weather_tool(self, city: str) -> Optional[str]:
        """调用天气工具并处理响应"""
        if not self.tools_available:
            return None
            
        try:
            translated_city = translate_city(city)
            print(f"🌍 识别到的城市: {city} → {translated_city}")
            
            # 调用天气查询工具
            result = await self.session.call_tool("query_weather", {"city": translated_city})
            if not result or not result.content:
                print("🚨 警告: 天气工具返回空结果")
                return None
                
            try:
                # 尝试解析JSON响应
                weather_data = json.loads(result.content[0].text)
                if not weather_data or "error" in weather_data:
                    print(f"🚨 天气API返回错误: {weather_data.get('error', '未知错误')}")
                    return None
                    
                return weather_data
            except json.JSONDecodeError:
                # 如果不是JSON，可能是格式化好的文本
                print("📝 天气工具返回了预格式化的文本")
                return {"formatted_text": result.content[0].text}
                
        except Exception as e:
            print(f"🚨 调用天气工具失败: {str(e)}")
            return None

    async def get_weather_suitability(self, weather_data: dict) -> str:
        """根据天气数据判断是否适合外出"""
        try:
            # 如果是预格式化的文本，直接返回
            if "formatted_text" in weather_data:
                return weather_data["formatted_text"]
                
            city = weather_data.get("name", "该地区")
            temp = weather_data["main"]["temp"]
            weather_desc = weather_data["weather"][0]["description"]
            humidity = weather_data["main"]["humidity"]
            wind_speed = weather_data["wind"]["speed"]
            
            # 简单判断逻辑
            suitability = "适合"
            reasons = []
            
            # 温度判断
            if temp > 30:
                suitability = "不太适合"
                reasons.append("气温较高")
            elif temp > 35:
                suitability = "不适合"
                reasons.append("气温过高")
            elif temp < 5:
                suitability = "不太适合"
                reasons.append("气温较低")
            elif temp < -5:
                suitability = "不适合"
                reasons.append("气温过低")
                
            # 天气状况判断
            bad_weather_keywords = ["rain", "thunderstorm", "drizzle", "snow", "shower"]
            if any(keyword in weather_desc.lower() for keyword in bad_weather_keywords):
                suitability = "不太适合"
                reasons.append("有降水")
                
            # 风速判断
            if wind_speed > 10:
                suitability = "不太适合"
                reasons.append("风力较大")
            elif wind_speed > 15:
                suitability = "不适合"
                reasons.append("风力过大")
                
            # 空气质量判断（如果数据中有）
            if "air_quality" in weather_data:
                aqi = weather_data["air_quality"].get("aqi", 0)
                if aqi > 150:
                    suitability = "不太适合"
                    reasons.append("空气质量较差")
                elif aqi > 200:
                    suitability = "不适合"
                    reasons.append("空气质量差")
            
            # 构建响应
            response = f"根据{city}的当前天气情况：\n"
            response += f"- 温度: {temp}°C\n"
            response += f"- 天气: {weather_desc.capitalize()}\n"
            response += f"- 湿度: {humidity}%\n"
            response += f"- 风速: {wind_speed} m/s\n\n"
            
            if suitability == "适合":
                response += "✅ 当前天气状况良好，**适合外出活动**。建议根据天气情况适当着装。"
            else:
                response += f"⚠️ 当前天气{', '.join(reasons)}，**{suitability}外出**。建议根据实际情况调整计划。"
                
            return response
            
        except Exception as e:
            print(f"🚨 天气适合性分析失败: {str(e)}")
            return None

    async def get_local_model_response(self, query: str) -> str:
        """使用本地模型获取响应（非流式）"""
        try:
            # 获取对话上下文
            context = self.dialogue_memory.get_recent_context()
            
            # 构建系统提示
            system_prompt = """
            你是一个友好的AI助手，能够理解上下文并提供有帮助的回答。
            当前对话上下文：
            {context}
            
            用户问题：
            {query}
            
            请根据上下文和用户问题，提供自然、有帮助的回答。
            """
            
            prompt_with_context = system_prompt.format(
                context=context if context else "无",
                query=query
            )
            
            messages = [{"role": "system", "content": prompt_with_context}]
            response = self.local_client.chat.completions.create(
                model=LOCAL_MODEL_NAME,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"🚨 本地模型调用失败: {str(e)}")
            return "抱歉，我暂时无法处理这个请求，请稍后再试。"

    async def process_query(self, query: str) -> str:
        try:
            # 添加到对话记忆
            self.dialogue_memory.add_message("user", query)
            
            should_call, city = await self.should_call_tool(query)
            
            if should_call and city:
                # 调用天气工具
                weather_data = await self.call_weather_tool(city)
                
                if not weather_data:
                    # 如果天气调用失败，使用本地模型提供默认回答
                    fallback_response = f"抱歉，我暂时无法获取{city}的天气信息。你可以稍后再试，或者查看天气预报应用获取最新信息。"
                    self.dialogue_memory.add_message("assistant", fallback_response)
                    return fallback_response
                
                # 检查是否是适合外出的查询
                if "适合外出" in query.lower() or "外出" in query.lower():
                    suitability_response = await self.get_weather_suitability(weather_data)
                    if suitability_response:
                        self.dialogue_memory.add_message("assistant", suitability_response)
                        return suitability_response
                
                # 返回原始天气信息
                if "formatted_text" in weather_data:
                    response = weather_data["formatted_text"]
                else:
                    # 格式化天气数据为易读文本
                    temp = weather_data["main"]["temp"]
                    weather_desc = weather_data["weather"][0]["description"]
                    humidity = weather_data["main"]["humidity"]
                    wind_speed = weather_data["wind"]["speed"]
                    city_name = weather_data.get("name", city)
                    
                    response = (
                        f"🌍 {city_name}当前天气：\n"
                        f"🌡 温度: {temp}°C\n"
                        f"🌤 天气: {weather_desc}\n"
                        f"💧 湿度: {humidity}%\n"
                        f"🌬 风速: {wind_speed} m/s"
                    )
                
                self.dialogue_memory.add_message("assistant", response)
                return response
                
            else:
                # 非天气查询，使用本地模型处理
                response = await self.get_local_model_response(query)
                self.dialogue_memory.add_message("assistant", response)
                return response
                
        except Exception as e:
            print(f"🚨 处理查询时出错: {str(e)}")
            error_response = "抱歉，处理你的请求时出现了错误，请稍后再试。"
            self.dialogue_memory.add_message("assistant", error_response)
            return error_response

    async def chat_loop(self):
        print("\n🤖 智能助手已启动！输入'quit'退出")
        await self.connect_to_server(sys.argv[1])

        while True:
            try:
                query = input("\n你: ").strip()
                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print(f"\n🤖: {response}")

            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python kaiyuan.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())