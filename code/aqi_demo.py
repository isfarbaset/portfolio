"""
Simple Air Quality Intelligence Demo
Demonstrates the AQI MCP Server without requiring Letta
"""

import asyncio
import sys
import os

# Add the code directory to path
sys.path.insert(0, os.path.dirname(__file__))

from aqi_mcp_server import AQIMCPServer


async def interactive_demo():
    """Interactive demo of the AQI system"""
    
    # Get API key
    api_key = os.getenv("API_NINJAS_KEY")
    
    if not api_key:
        print("\n⚠️  API KEY REQUIRED")
        print("=" * 60)
        print("Please set your API_NINJAS_KEY environment variable:")
        print("  export API_NINJAS_KEY='your_key_here'")
        print("\nGet your free API key at: https://api.api-ninjas.com")
        print("=" * 60)
        return
    
    server = AQIMCPServer(api_key)
    
    print("\n🌍 AIR QUALITY INTELLIGENCE SYSTEM")
    print("=" * 60)
    print("Ask me about air quality in any city!")
    print("Commands:")
    print("  - Type a city name (e.g., 'Dhaka', 'New York')")
    print("  - 'compare [city1] [city2] [city3]' to compare cities")
    print("  - 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using Air Quality Intelligence!")
                break
            
            if query.lower().startswith('compare'):
                # Parse cities from compare command
                parts = query.split()[1:]
                if len(parts) < 2:
                    print("❌ Please provide at least 2 cities to compare")
                    continue
                
                print(f"\n📊 Comparing air quality across {len(parts)} cities...")
                results = []
                
                for city in parts:
                    aqi_data = await server.get_air_quality(city)
                    if aqi_data:
                        results.append((
                            city,
                            aqi_data.overall_aqi,
                            server.get_health_category(aqi_data.overall_aqi),
                            aqi_data.pm2_5_concentration
                        ))
                
                if results:
                    results.sort(key=lambda x: x[1])
                    print("\n🏆 Rankings (Best to Worst):")
                    print("-" * 60)
                    for i, (city, aqi, category, pm25) in enumerate(results, 1):
                        emoji = "🟢" if aqi <= 50 else "🟡" if aqi <= 100 else "🟠" if aqi <= 150 else "🔴"
                        print(f"{emoji} {i}. {city}")
                        print(f"   AQI: {aqi} ({category})")
                        print(f"   PM2.5: {pm25:.1f} µg/m³")
                        print()
                    
                    best = results[0]
                    worst = results[-1]
                    print(f"✅ Best: {best[0]} (AQI {best[1]})")
                    print(f"⚠️  Worst: {worst[0]} (AQI {worst[1]})")
                else:
                    print("❌ Could not retrieve data for comparison")
            
            elif query:
                # Single city query
                print(f"\n📍 Checking air quality in {query}...")
                aqi_data = await server.get_air_quality(query)
                
                if aqi_data:
                    category = server.get_health_category(aqi_data.overall_aqi)
                    
                    # Choose emoji based on AQI
                    if aqi_data.overall_aqi <= 50:
                        emoji = "🟢"
                    elif aqi_data.overall_aqi <= 100:
                        emoji = "🟡"
                    elif aqi_data.overall_aqi <= 150:
                        emoji = "🟠"
                    else:
                        emoji = "🔴"
                    
                    print("\n" + "=" * 60)
                    print(f"{emoji} {query.upper()} - {category}")
                    print("=" * 60)
                    print(f"Overall AQI: {aqi_data.overall_aqi}")
                    print(f"Location: ({aqi_data.latitude:.4f}, {aqi_data.longitude:.4f})")
                    print("\nPollutant Concentrations:")
                    print(f"  PM2.5: {aqi_data.pm2_5_concentration:.1f} µg/m³")
                    print(f"  PM10:  {aqi_data.pm10_concentration:.1f} µg/m³")
                    print(f"  O3:    {aqi_data.o3_concentration:.1f} µg/m³")
                    print(f"  NO2:   {aqi_data.no2_concentration:.1f} µg/m³")
                    print(f"  SO2:   {aqi_data.so2_concentration:.1f} µg/m³")
                    print(f"  CO:    {aqi_data.co_concentration:.1f} µg/m³")
                    
                    print("\n💡 HEALTH RECOMMENDATIONS")
                    print("-" * 60)
                    print("General:", server.get_health_advice(aqi_data.overall_aqi, "general"))
                    print("\nFor Jogging:", server.get_health_advice(aqi_data.overall_aqi, "jogging"))
                    print("\nFor Cycling:", server.get_health_advice(aqi_data.overall_aqi, "cycling"))
                    print("\nSensitive Groups:", server.get_health_advice(aqi_data.overall_aqi, "sensitive"))
                else:
                    print(f"❌ Could not retrieve air quality data for {query}")
                    print("   Please check the city name and try again.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Air Quality Intelligence!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again.")


async def quick_demo():
    """Quick demo with preset cities"""
    api_key = os.getenv("API_NINJAS_KEY")
    
    if not api_key:
        print("\n⚠️  Please set API_NINJAS_KEY environment variable")
        return
    
    server = AQIMCPServer(api_key)
    
    cities = ["Dhaka", "New York", "Los Angeles", "Seattle", "London"]
    
    print("\n🌍 AIR QUALITY QUICK REPORT")
    print("=" * 60)
    
    for city in cities:
        aqi_data = await server.get_air_quality(city)
        if aqi_data:
            category = server.get_health_category(aqi_data.overall_aqi)
            emoji = "🟢" if aqi_data.overall_aqi <= 50 else "🟡" if aqi_data.overall_aqi <= 100 else "🟠" if aqi_data.overall_aqi <= 150 else "🔴"
            print(f"{emoji} {city:15} | AQI: {aqi_data.overall_aqi:3d} | {category}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(quick_demo())
    else:
        asyncio.run(interactive_demo())
