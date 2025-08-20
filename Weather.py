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

# Load .env file
load_dotenv()

# Local model configuration
LOCAL_MODEL_HOST = "0.0.0.0"
LOCAL_MODEL_PORT = "8180"
LOCAL_MODEL_API_KEY = ""
LOCAL_MODEL_NAME = ""

class DialogueMemory:
    """Dialogue context memory class"""
    def __init__(self):
        self.history: List[Dict] = []
        self.max_history_length = 5  # Keep last 5 rounds of conversation

    def add_message(self, role: str, content: str):
        """Add dialogue message"""
        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        self.history.append(message)
        
        # Limit history length
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

    def get_recent_context(self) -> str:
        """Get recent dialogue context"""
        if not self.history:
            return ""
        
        # Get recent user queries and assistant responses
        context_messages = []
        for i in range(len(self.history)-1, max(-1, len(self.history)-6), -2):
            if i >= 0 and self.history[i]['role'] == 'user':
                context_messages.append(f"User: {self.history[i]['content']}")
            if i-1 >= 0 and self.history[i-1]['role'] == 'assistant':
                context_messages.append(f"Assistant: {self.history[i-1]['content']}")
        
        return "\n\n".join(reversed(context_messages)) if context_messages else ""

    def clear(self):
        """Clear dialogue history"""
        self.history = []

def translate_city(city: str) -> str:
    """Translate Chinese city name to English"""
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
    """Check if query is weather-related"""
    weather_keywords = ["weather", "temperature", "humidity", "rain", "snow", "sunny", "cloudy", "wind"]
    return any(keyword in query.lower() for keyword in weather_keywords)

class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        
        # Local model client (for decision making)
        self.local_client = openai.Client(
            base_url=f"http://{LOCAL_MODEL_HOST}:{LOCAL_MODEL_PORT}/v1",
            api_key=LOCAL_MODEL_API_KEY
        )
        
        # Dialogue memory
        self.dialogue_memory = DialogueMemory()
        
        self.session: Optional[ClientSession] = None
        self.tools_available = False

    async def connect_to_server(self, server_script_path: str):
        """Connect to MCP server"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be .py or .js file")

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

        # Check if tools are available
        response = await self.session.list_tools()
        tools = {tool.name: tool for tool in response.tools}
        self.tools_available = "query_weather" in tools
        print("\nConnected to server. Weather query tool is" + (" available" if self.tools_available else " not available"))
        
        return tools

    async def extract_city(self, query: str) -> Optional[str]:
        """Extract city name from query"""
        # Remove prefix words
        query = query.replace("you", "").replace("please", "").replace("know", "").strip()
        
        # Try to match standard format
        city_match = re.search(r"(.+?)(?: now| today|'s) weather", query)
        if city_match:
            return city_match.group(1).strip()
        
        # Try more flexible format
        city_match = re.search(r"((?:Beijing|Shanghai|Guangzhou|Shenzhen|Tianjin|Chongqing|Chengdu|Hangzhou|Wuhan|Xi'an|Suzhou|Zhengzhou|Nanjing|Qingdao|Shenyang|Dalian|Xiamen|Fuzhou|Changsha|Harbin|Jinan|Changchun|Shijiazhuang|Hefei|Taiyuan|Nanchang|Nanning|Kunming|Guiyang|Haikou|Urumqi|Hohhot|Yinchuan|Xining|Lanzhou|Lhasa|Ningbo|Wenzhou|Wuxi|Changzhou|Nantong|Xuzhou|Yangzhou|Yancheng|Taizhou|Zhenjiang|Huai'an|Lianyungang|Suqian|Jiaxing|Huzhou|Shaoxing|Jinhua|Quzhou|Zhoushan|Taizhou|Lishui|Wuhu|Ma'anshan|Anqing|Chuzhou|Fuyang|Suzhou|Lu'an|Bozhou|Chizhou|Xuancheng|Zhangzhou|Nanping|Sanming|Longyan|Ningde|Jingdezhen|Pingxiang|Jiujiang|Xinyu|Yingtan|Ganzhou|Ji'an|Yichun|Fuzhou|Shangrao|Zibo|Zaozhuang|Dongying|Yantai|Weifang|Jining|Tai'an|Weihai|Rizhao|Linyi|Dezhou|Liaocheng|Binzhou|Heze|Kaifeng|Luoyang|Pingdingshan|Anyang|Hebi|Xinxiang|Jiaozuo|Puyang|Xuchang|Luohe|Sanmenxia|Nanyang|Shangqiu|Xinyang|Zhoukou|Zhumadian|Huangshi|Shiyan|Yichang|Xiangyang|Ezhou|Jingmen|Xiaogan|Jingzhou|Huanggang|Xianning|Suizhou|Enshi|Zhuzhou|Xiangtan|Hengyang|Shaoyang|Yueyang|Changde|Zhangjiajie|Yiyang|Chenzhou|Yongzhou|Huaihua|Loudi|Xiangxi|Shaoguan|Zhuhai|Shantou|Foshan|Jiangmen|Zhanjiang|Maoming|Zhaoqing|Huizhou|Meizhou|Shanwei|Heyuan|Yangjiang|Qingyuan|Dongguan|Zhongshan|Chaozhou|Jieyang|Yunfu|Liuzhou|Guilin|Wuzhou|Beihai|Fangchenggang|Qinzhou|Guigang|Yulin|Baise|Hezhou|Hechi|Laibin|Chongzuo|Danzhou|Zigong|Panzhihua|Luzhou|Deyang|Mianyang|Guangyuan|Suining|Neijiang|Leshan|Nanchong|Meishan|Yibin|Guang'an|Dazhou|Ya'an|Bazhong|Ziyang|Aba|Ganzi|Liangshan|Liupanshui|Zunyi|Anshun|Bijie|Tongren|Qiandongnan|Qiannan|Qianxinan|Qujing|Yuxi|Baoshan|Zhaotong|Lijiang|Puer|Lincang|Chuxiong|Honghe|Wenshan|Xishuangbanna|Dali|Dehong|Nujiang|Diqing|Qamdo|Nyingchi|Shannan|Nagqu|Ali|Tongchuan|Baoji|Xianyang|Weinan|Yanan|Hanzhong|Yulin|Ankang|Shangluo|Jiayuguan|Jinchang|Baiyin|Tianshui|Wuwei|Zhangye|Pingliang|Jiuquan|Qingyang|Dingxi|Longnan|Linxia|Gannan|Haidong|Haibei|Huanan|Hainan|Guoluo|Yushu|Haixi|Shizuishan|Wuzhong|Guyuan|Zhongwei|Karamay|Turpan|Hami|Changji|Bortala|Bayingolin|Aksu|Kizilsu|Kashgar|Hotan|Ili|Tacheng|Altay|Shihezi|Aral|Tumxuk|Wujiaqu|Beitun|Tiemenguan|Shuanghe|Kekedala|Kunyu|Huyanghe|Xinxing)[city province]?)", query, re.IGNORECASE)
        if city_match:
            return city_match.group(1).strip()
        
        # Try to extract provincial administrative region
        province_match = re.search(r"((?:Beijing|Tianjin|Shanghai|Chongqing|Hebei|Shanxi|Liaoning|Jilin|Heilongjiang|Jiangsu|Zhejiang|Anhui|Fujian|Jiangxi|Shandong|Henan|Hubei|Hunan|Guangdong|Hainan|Sichuan|Guizhou|Yunnan|Shaanxi|Gansu|Qinghai|Taiwan|Inner Mongolia|Guangxi|Tibet|Ningxia|Xinjiang|Hong Kong|Macau)[ province autonomous region]?)", query, re.IGNORECASE)
        if province_match:
            return province_match.group(1).strip()
        
        return None

    async def should_call_tool(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Decide whether to call the tool
        :return: (should call tool, extracted city name)
        """
        # First do simple keyword matching
        if await is_weather_query(query):
            city = await self.extract_city(query)
            return (city is not None, city)
        
        # If simple matching fails, use local model for smarter judgment
        system_prompt = """
        You are an assistant helping to judge user intent. Please determine if the user is asking for weather information.
        If yes, extract the city name; if not, return None.
        
        Judgment principles:
        1. If the query is clearly asking about weather (containing words like "weather", "temperature", "humidity", "rain", etc.) and specifies a city, the weather query tool should be called
        2. If the query is casual chat, a general question, or a request that doesn't require external data, don't call the tool
        3. If the query format doesn't match standard weather query format (e.g., "What's the weather like in Beijing now"), but the intent is clearly to query weather, still call the tool
        4. Prefer to use the tool to get real-time data rather than relying on built-in knowledge
        
        Current dialogue context:
        {context}
        
        Please judge whether the following query needs to call the tool (answer "yes" or "no"). If needed, extract the city name:
        """
        
        # Get dialogue context
        context = self.dialogue_memory.get_recent_context()
        system_prompt_with_context = system_prompt.format(context=context if context else "None")
        
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
            print(f"🤖 Decision result: {decision}")
            
            if "no" in decision.lower() or "not needed" in decision.lower() or "none" in decision.lower():
                return (False, None)
            else:
                # Try to extract city name from decision
                city_match = re.search(r"city[:：]\s*(\S+)", decision)
                if city_match:
                    return (True, city_match.group(1))
                
                # Try to extract city from original query
                city = await self.extract_city(query)
                return (city is not None, city)
                
        except Exception as e:
            print(f"🚨 Decision model call failed: {str(e)}")
            return (False, None)

    async def call_weather_tool(self, city: str) -> Optional[str]:
        """Call weather tool and handle response"""
        if not self.tools_available:
            return None
            
        try:
            translated_city = translate_city(city)
            print(f"🌍 Recognized city: {city} → {translated_city}")
            
            # Call weather query tool
            result = await self.session.call_tool("query_weather", {"city": translated_city})
            if not result or not result.content:
                print("🚨 Warning: Weather tool returned empty result")
                return None
                
            try:
                # Try to parse JSON response
                weather_data = json.loads(result.content[0].text)
                if not weather_data or "error" in weather_data:
                    print(f"🚨 Weather API returned error: {weather_data.get('error', 'Unknown error')}")
                    return None
                    
                return weather_data
            except json.JSONDecodeError:
                # If not JSON, might be pre-formatted text
                print("📝 Weather tool returned pre-formatted text")
                return {"formatted_text": result.content[0].text}
                
        except Exception as e:
            print(f"🚨 Weather tool call failed: {str(e)}")
            return None

    async def get_weather_suitability(self, weather_data: dict) -> str:
        """Determine suitability for going out based on weather data"""
        try:
            # If pre-formatted text, return directly
            if "formatted_text" in weather_data:
                return weather_data["formatted_text"]
                
            city = weather_data.get("name", "the area")
            temp = weather_data["main"]["temp"]
            weather_desc = weather_data["weather"][0]["description"]
            humidity = weather_data["main"]["humidity"]
            wind_speed = weather_data["wind"]["speed"]
            
            # Simple judgment logic
            suitability = "suitable"
            reasons = []
            
            # Temperature judgment
            if temp > 30:
                suitability = "not very suitable"
                reasons.append("high temperature")
            elif temp > 35:
                suitability = "not suitable"
                reasons.append("extremely high temperature")
            elif temp < 5:
                suitability = "not very suitable"
                reasons.append("low temperature")
            elif temp < -5:
                suitability = "not suitable"
                reasons.append("extremely low temperature")
                
            # Weather condition judgment
            bad_weather_keywords = ["rain", "thunderstorm", "drizzle", "snow", "shower"]
            if any(keyword in weather_desc.lower() for keyword in bad_weather_keywords):
                suitability = "not very suitable"
                reasons.append("precipitation")
                
            # Wind speed judgment
            if wind_speed > 10:
                suitability = "not very suitable"
                reasons.append("strong wind")
            elif wind_speed > 15:
                suitability = "not suitable"
                reasons.append("very strong wind")
                
            # Air quality judgment (if in data)
            if "air_quality" in weather_data:
                aqi = weather_data["air_quality"].get("aqi", 0)
                if aqi > 150:
                    suitability = "not very suitable"
                    reasons.append("poor air quality")
                elif aqi > 200:
                    suitability = "not suitable"
                    reasons.append("bad air quality")
            
            # Build response
            response = f"Based on current weather in {city}:\n"
            response += f"- Temperature: {temp}°C\n"
            response += f"- Weather: {weather_desc.capitalize()}\n"
            response += f"- Humidity: {humidity}%\n"
            response += f"- Wind speed: {wind_speed} m/s\n\n"
            
            if suitability == "suitable":
                response += "✅ The current weather is good, **suitable for outdoor activities**. Suggest dressing appropriately."
            else:
                response += f"⚠️ The current weather has {', '.join(reasons)}, **{suitability} for going out**. Adjust plans based on actual conditions."
                
            return response
            
        except Exception as e:
            print(f"🚨 Weather suitability analysis failed: {str(e)}")
            return None

    async def get_local_model_response(self, query: str) -> str:
        """Get response from local model (non-streaming)"""
        try:
            # Get dialogue context
            context = self.dialogue_memory.get_recent_context()
            
            # Build system prompt
            system_prompt = """
            You are a friendly AI assistant capable of understanding context and providing helpful responses.
            Current dialogue context:
            {context}
            
            User question:
            {query}
            
            Please provide a natural and helpful response based on the context and user question.
            """
            
            prompt_with_context = system_prompt.format(
                context=context if context else "None",
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
            print(f"🚨 Local model call failed: {str(e)}")
            return "Sorry, I can't process this request right now. Please try again later."

    async def process_query(self, query: str) -> str:
        try:
            # Add to dialogue memory
            self.dialogue_memory.add_message("user", query)
            
            should_call, city = await self.should_call_tool(query)
            
            if should_call and city:
                # Call weather tool
                weather_data = await self.call_weather_tool(city)
                
                if not weather_data:
                    # If weather call fails, use local model for default response
                    fallback_response = f"Sorry, I can't get weather information for {city} right now. You can check weather apps later for updated information."
                    self.dialogue_memory.add_message("assistant", fallback_response)
                    return fallback_response
                
                # Check if it's a suitability query
                if "go out" in query.lower() or "outdoor" in query.lower():
                    suitability_response = await self.get_weather_suitability(weather_data)
                    if suitability_response:
                        self.dialogue_memory.add_message("assistant", suitability_response)
                        return suitability_response
                
                # Return raw weather info
                if "formatted_text" in weather_data:
                    response = weather_data["formatted_text"]
                else:
                    # Format weather data into readable text
                    temp = weather_data["main"]["temp"]
                    weather_desc = weather_data["weather"][0]["description"]
                    humidity = weather_data["main"]["humidity"]
                    wind_speed = weather_data["wind"]["speed"]
                    city_name = weather_data.get("name", city)
                    
                    response = (
                        f"🌍 Current weather in {city_name}:\n"
                        f"🌡 Temperature: {temp}°C\n"
                        f"🌤 Weather: {weather_desc}\n"
                        f"💧 Humidity: {humidity}%\n"
                        f"🌬 Wind speed: {wind_speed} m/s"
                    )
                
                self.dialogue_memory.add_message("assistant", response)
                return response
                
            else:
                # Non-weather query, use local model
                response = await self.get_local_model_response(query)
                self.dialogue_memory.add_message("assistant", response)
                return response
                
        except Exception as e:
            print(f"🚨 Error processing query: {str(e)}")
            error_response = "Sorry, an error occurred while processing your request. Please try again later."
            self.dialogue_memory.add_message("assistant", error_response)
            return error_response

    async def chat_loop(self):
        print("\n🤖 AI Assistant started! Type 'quit' to exit")
        await self.connect_to_server(sys.argv[1])

        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print(f"\n🤖: {response}")

            except Exception as e:
                print(f"\n⚠️ Error occurred: {str(e)}")

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
