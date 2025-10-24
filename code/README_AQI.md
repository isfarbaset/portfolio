# Air Quality Intelligence System

An intelligent system that provides real-time air quality monitoring and personalized health recommendations using the Model Context Protocol (MCP).

## Quick Start

### 1. Install Dependencies

```bash
pip install aiohttp
```

### 2. Get API Key

1. Sign up at [API Ninjas](https://api.api-ninjas.com)
2. Get your free API key
3. Set it as an environment variable:

```bash
export API_NINJAS_KEY='your_key_here'
```

### 3. Run the Demo

**Interactive Mode:**
```bash
python aqi_demo.py
```

**Quick Report:**
```bash
python aqi_demo.py quick
```

## Features

- ✅ Real-time air quality data for any city worldwide
- ✅ Detailed pollutant breakdown (PM2.5, PM10, O3, NO2, SO2, CO)
- ✅ Health category classification
- ✅ Activity-specific recommendations (jogging, cycling, etc.)
- ✅ Multi-city comparison
- ✅ Sensitive group guidance

## Example Usage

### Single City Query
```
🔍 Your question: Dhaka

📍 Checking air quality in Dhaka...

🔴 DHAKA - Unhealthy
AQI: 189
PM2.5: 148.2 µg/m³

💡 HEALTH RECOMMENDATIONS
General: Everyone should avoid prolonged outdoor activities.
For Jogging: Avoid outdoor jogging. Exercise indoors instead.
```

### Multi-City Comparison
```
🔍 Your question: compare Dhaka Seattle Tokyo

📊 Comparing air quality across 3 cities...

🏆 Rankings (Best to Worst):
1. Seattle - AQI 42 (Good)
2. Tokyo - AQI 78 (Moderate)
3. Dhaka - AQI 189 (Unhealthy)
```

## Health Categories

- **Good (0-50)**: Air quality is excellent
- **Moderate (51-100)**: Acceptable for most people
- **Unhealthy for Sensitive Groups (101-150)**: Sensitive individuals should limit prolonged outdoor exertion
- **Unhealthy (151-200)**: Everyone should avoid prolonged outdoor activities
- **Very Unhealthy (201-300)**: Health alert for everyone
- **Hazardous (301+)**: Health emergency

## Project Structure

```
code/
├── aqi_mcp_server.py  # Core MCP server implementation
├── aqi_demo.py        # Interactive demo
└── README_AQI.md      # This file
```

## API Data Source

This project uses the [API Ninjas Air Quality API](https://api.api-ninjas.com/api/airquality) which provides:
- Global coverage
- Real-time updates
- Comprehensive pollutant data
- GPS coordinate precision

## Future Enhancements

- [ ] 24-48 hour air quality forecasting
- [ ] Historical trend analysis
- [ ] Weather integration
- [ ] Personal exposure tracking
- [ ] Mobile app
- [ ] Alert notifications
