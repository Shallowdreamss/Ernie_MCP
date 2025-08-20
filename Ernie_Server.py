import sys
sys.path.append('/home/aistudio/external-libraries/lib/python3.12/site-packages')
import httpx
from mcp.server.fastmcp import FastMCP
import json
from typing import Any


# Initialize MCP server
mcp = FastMCP("WeatherServer")


# OpenWeather API configuration
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "930c516f5f4c4c3dcd072f731f5c1c37"  # Please replace with your own OpenWeather API Key
USER_AGENT = "weather-app/1.0"


async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    Fetch weather information from OpenWeather API.
    :param city: City name (must be in English, e.g., Beijing)
    :return: Weather data dictionary; returns dictionary with error info if error occurs
    """
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "en"  # Changed to English response
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(OPENWEATHER_API_BASE, params=params, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()  # Returns dictionary type
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

def format_weather(data: dict[str, Any] | str) -> str:
    """
    Format weather data into readable text.
    :param data: Weather data (can be dictionary or JSON string)
    :return: Formatted weather information string
    """
    # Convert string to dictionary if needed
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return f"Failed to parse weather data: {e}"

    # Return error message if data contains error info
    if "error" in data:
        return f"⚠️ {data['error']}"

    # Error handling when extracting data
    city = data.get("name", "Unknown")
    country = data.get("sys", {}).get("country", "Unknown")
    temp = data.get("main", {}).get("temp", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    # weather might be empty list, so provide default dict before using [0]
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "Unknown")

    return (
        f"🌍 {city}, {country}\n"
        f"🌡 Temperature: {temp}°C\n"
        f"💧 Humidity: {humidity}%\n"
        f"🌬 Wind Speed: {wind_speed} m/s\n"
        f"🌤 Conditions: {description}\n"
    )

@mcp.tool()
async def query_weather(city: str) -> str:
    """
    Input a city name in English and return today's weather information.
    :param city: City name (must be in English)
    :return: Formatted weather information
    """
    data = await fetch_weather(city)
    return format_weather(data)

if __name__ == "__main__":
    # Run MCP server in standard I/O mode
    mcp.run(transport='stdio')
