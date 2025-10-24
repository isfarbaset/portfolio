"""
MCP Server for Air Quality Intelligence
Provides real-time air quality data through the Model Context Protocol
"""

import asyncio
import aiohttp
from dataclasses import dataclass, asdict
from typing import Optional, List
import os
from datetime import datetime


@dataclass
class AQIData:
    """Air Quality Index data structure"""
    location: str
    latitude: float
    longitude: float
    overall_aqi: int
    co_concentration: float
    no2_concentration: float
    o3_concentration: float
    so2_concentration: float
    pm2_5_concentration: float
    pm10_concentration: float
    timestamp: str


class AQIMCPServer:
    """MCP Server for Air Quality Intelligence"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.api-ninjas.com/v1"
        
    async def get_coordinates(self, city: str) -> Optional[tuple]:
        """Get GPS coordinates for a city using geocoding"""
        async with aiohttp.ClientSession() as session:
            headers = {"X-Api-Key": self.api_key}
            url = f"{self.base_url}/geocoding?city={city}"
            
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and len(data) > 0:
                            return (data[0]['latitude'], data[0]['longitude'])
            except Exception as e:
                print(f"Error getting coordinates for {city}: {e}")
                return None
    
    async def get_air_quality(self, city: str) -> Optional[AQIData]:
        """Get air quality data for a city"""
        # First get coordinates
        coords = await self.get_coordinates(city)
        if not coords:
            print(f"Could not find coordinates for {city}")
            return None
        
        lat, lon = coords
        
        # Get air quality data
        async with aiohttp.ClientSession() as session:
            headers = {"X-Api-Key": self.api_key}
            url = f"{self.base_url}/airquality?lat={lat}&lon={lon}"
            
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return AQIData(
                            location=city,
                            latitude=lat,
                            longitude=lon,
                            overall_aqi=data.get('overall_aqi', 0),
                            co_concentration=data.get('CO', {}).get('concentration', 0),
                            no2_concentration=data.get('NO2', {}).get('concentration', 0),
                            o3_concentration=data.get('O3', {}).get('concentration', 0),
                            so2_concentration=data.get('SO2', {}).get('concentration', 0),
                            pm2_5_concentration=data.get('PM2.5', {}).get('concentration', 0),
                            pm10_concentration=data.get('PM10', {}).get('concentration', 0),
                            timestamp=datetime.now().isoformat()
                        )
            except Exception as e:
                print(f"Error getting air quality for {city}: {e}")
                return None
    
    def get_health_category(self, aqi: int) -> str:
        """Convert AQI to health category"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def get_health_advice(self, aqi: int, activity: str = "general") -> str:
        """Get health advice based on AQI and activity"""
        category = self.get_health_category(aqi)
        
        advice = {
            "Good": {
                "general": "Air quality is excellent. Enjoy outdoor activities!",
                "jogging": "Perfect conditions for jogging and running.",
                "cycling": "Ideal day for cycling.",
                "sensitive": "No precautions needed."
            },
            "Moderate": {
                "general": "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
                "jogging": "Good for jogging, but sensitive individuals should monitor their response.",
                "cycling": "Safe for cycling. Take breaks if needed.",
                "sensitive": "Consider reducing prolonged outdoor activities."
            },
            "Unhealthy for Sensitive Groups": {
                "general": "Sensitive groups should limit prolonged outdoor exertion.",
                "jogging": "Sensitive individuals should reduce jogging duration or exercise indoors.",
                "cycling": "Consider indoor cycling if you have respiratory conditions.",
                "sensitive": "Avoid prolonged outdoor activities. Use N95 masks if necessary."
            },
            "Unhealthy": {
                "general": "Everyone should avoid prolonged outdoor activities.",
                "jogging": "Avoid outdoor jogging. Exercise indoors instead.",
                "cycling": "Not recommended. Use indoor cycling equipment.",
                "sensitive": "Stay indoors. Use air purifiers and keep windows closed."
            },
            "Very Unhealthy": {
                "general": "Health alert! Everyone should avoid all outdoor activities.",
                "jogging": "Do not jog outdoors under any circumstances.",
                "cycling": "Do not cycle outdoors. Stay indoors.",
                "sensitive": "Emergency precautions: Stay indoors with air purifiers. Seek medical attention if experiencing symptoms."
            },
            "Hazardous": {
                "general": "Health emergency! Everyone should remain indoors.",
                "jogging": "Absolutely no outdoor exercise.",
                "cycling": "Absolutely no outdoor cycling.",
                "sensitive": "Medical emergency level. Stay indoors, use air purifiers, seal windows. Seek immediate medical help if experiencing symptoms."
            }
        }
        
        return advice.get(category, {}).get(activity, advice[category]["general"])


async def demo():
    """Demo the AQI MCP Server"""
    # Get API key from environment or use demo key
    api_key = os.getenv("API_NINJAS_KEY", "YOUR_API_KEY_HERE")
    
    if api_key == "YOUR_API_KEY_HERE":
        print("⚠️  Please set your API_NINJAS_KEY environment variable")
        print("Get your free API key at: https://api.api-ninjas.com")
        return
    
    server = AQIMCPServer(api_key)
    
    # Test cities
    cities = ["Dhaka", "New York", "Los Angeles", "Seattle"]
    
    print("🌍 Air Quality Intelligence System")
    print("=" * 60)
    
    for city in cities:
        print(f"\n📍 Checking {city}...")
        aqi_data = await server.get_air_quality(city)
        
        if aqi_data:
            category = server.get_health_category(aqi_data.overall_aqi)
            advice = server.get_health_advice(aqi_data.overall_aqi)
            
            print(f"   AQI: {aqi_data.overall_aqi} ({category})")
            print(f"   PM2.5: {aqi_data.pm2_5_concentration:.1f} µg/m³")
            print(f"   Location: ({aqi_data.latitude:.4f}, {aqi_data.longitude:.4f})")
            print(f"   💡 {advice}")
        else:
            print(f"   ❌ Could not retrieve data for {city}")
    
    # Compare cities
    print("\n" + "=" * 60)
    print("🔍 Multi-City Comparison")
    print("=" * 60)
    
    results = []
    for city in cities[:3]:
        aqi_data = await server.get_air_quality(city)
        if aqi_data:
            results.append((city, aqi_data.overall_aqi, server.get_health_category(aqi_data.overall_aqi)))
    
    results.sort(key=lambda x: x[1])
    
    print("\nRanking (Best to Worst Air Quality):")
    for i, (city, aqi, category) in enumerate(results, 1):
        print(f"{i}. {city}: AQI {aqi} ({category})")


if __name__ == "__main__":
    asyncio.run(demo())
