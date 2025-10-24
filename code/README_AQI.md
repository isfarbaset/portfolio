# Air Quality Intelligence System 🌍🤖

An intelligent system that provides real-time air quality monitoring and **AI-powered personalized health recommendations** using the Model Context Protocol (MCP) and OpenAI GPT.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install aiohttp openai
```

### 2. Get API Keys

**Air Quality Data (Required):**
1. Sign up at [API Ninjas](https://api.api-ninjas.com)
2. Get your free API key

**Gen AI Features (Optional):**
1. Get OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Enables AI-powered conversational assistant

Set environment variables:
```bash
export API_NINJAS_KEY='your_air_quality_key_here'
export OPENAI_API_KEY='your_openai_key_here'  # Optional
```

### 3. Run the Demos

**🤖 Gen AI Conversational Agent (Recommended):**
```bash
python aqi_genai_agent.py
```

**Interactive Air Quality Check:**
```bash
python aqi_demo.py
```

**Quick Report:**
```bash
python aqi_demo.py quick
```

## ✨ Features

### Core Air Quality Monitoring
- ✅ Real-time air quality data for any city worldwide
- ✅ Detailed pollutant breakdown (PM2.5, PM10, O3, NO2, SO2, CO)
- ✅ Health category classification
- ✅ Activity-specific recommendations (jogging, cycling, etc.)
- ✅ Multi-city comparison
- ✅ Sensitive group guidance

### 🤖 Gen AI Capabilities (New!)
- ✅ **Natural language conversations** about air quality
- ✅ **Personalized health advice** based on user conditions (asthma, allergies, etc.)
- ✅ **Activity planning guidance** (outdoor sports, travel, etc.)
- ✅ **Contextual recommendations** powered by GPT-4
- ✅ **Multi-city intelligent comparisons** with insights
- ✅ **Fallback mode** works without OpenAI (basic responses)

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

## 📁 Project Structure

```
code/
├── aqi_mcp_server.py              # Core MCP server implementation
├── aqi_demo.py                    # Interactive demo
├── aqi_genai_agent.py             # 🤖 Gen AI conversational agent (NEW!)
├── genai_example_conversations.txt # Example AI conversations
├── aqi_example_output.txt         # Sample outputs
├── .env.aqi                       # Environment configuration
└── README_AQI.md                  # This file
```

## 🤖 Gen AI Examples

The Gen AI agent combines real-time data with AI intelligence for natural conversations:

**Example 1: Simple Query**
```
💬 You: What's the air quality like in Boston today?

🤖 AI: Great news! The air quality in Boston is currently excellent 
with an AQI of 47 (Good). All outdoor activities are perfectly safe. 
PM2.5 is very low at 0.7 µg/m³. Perfect day for jogging, cycling, 
or spending time outdoors! 🌤️
```

**Example 2: Health-Specific Advice**
```
💬 You: I have asthma. Can I go jogging in Dhaka today?

🤖 AI: I'd advise caution. Dhaka's AQI is 116 (Unhealthy for Sensitive 
Groups). With asthma, I recommend:
1. Exercise indoors today (gym, treadmill)
2. If you must jog outside, keep it short (15-20 min), go early morning
3. Carry your inhaler and monitor for symptoms
Your health comes first! 💙
```

**Example 3: Travel Planning**
```
💬 You: Compare air quality in Seattle and Los Angeles

🤖 AI: Seattle wins today! 🏆
- Seattle: AQI 96 (Moderate) - Great for outdoor activities
- Los Angeles: AQI 107 (Unhealthy for Sensitive Groups) - Consider 
  indoor activities if you're sensitive to pollution

Seattle has 11 points better AQI. Both have low PM2.5, but LA has 
higher ozone levels due to typical Southern California weather patterns.
```

See `genai_example_conversations.txt` for more detailed examples!

## API Data Source

This project uses the [API Ninjas Air Quality API](https://api.api-ninjas.com/api/airquality) which provides:
- Global coverage
- Real-time updates
- Comprehensive pollutant data
- GPS coordinate precision

## 🎯 Technical Highlights

- **MCP Architecture**: Modern server-client pattern for extensibility
- **Async/Await**: Non-blocking I/O for fast multi-city queries
- **OpenAI Integration**: GPT-4 for natural language understanding
- **Smart Fallback**: Works without OpenAI with rule-based responses
- **Type Safety**: Pydantic models for data validation
- **Real-time Data**: Live API integration with sub-second response

## 🌟 Future Enhancements

- [ ] 24-48 hour air quality forecasting
- [ ] Historical trend analysis
- [ ] Weather integration
- [ ] Personal exposure tracking
- [ ] Mobile app
- [ ] Alert notifications
