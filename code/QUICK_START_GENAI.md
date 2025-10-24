# Quick Start: Gen AI Air Quality Agent 🤖

## Installation

```bash
# Install dependencies
pip install aiohttp openai

# Get API keys
# 1. Air Quality Data: https://api.api-ninjas.com
# 2. OpenAI GPT: https://platform.openai.com/api-keys

# Set environment variables
export API_NINJAS_KEY='your_air_quality_api_key'
export OPENAI_API_KEY='your_openai_api_key'
```

## Running the Agent

### Interactive Mode (Recommended)
```bash
cd /Users/isfarbaset/Documents/portfolio/code
python3 aqi_genai_agent.py
```

Ask questions like:
- "What's the air quality in Boston?"
- "I have asthma. Can I jog in Seattle today?"
- "Compare air quality in New York and Los Angeles"
- "Is it safe to take my kids to the park in London?"

### Demo Mode (Preset Conversations)
```bash
python3 aqi_genai_agent.py demo
```

### Fallback Mode (Without OpenAI)
```bash
# Only set air quality key
export API_NINJAS_KEY='your_air_quality_api_key'

python3 aqi_genai_agent.py
# Still works with basic rule-based responses!
```

## Example Session

```
🤖 GEN AI AIR QUALITY INTELLIGENCE AGENT
============================================================
Ask me anything about air quality! I use real-time data
and AI to provide personalized recommendations.

Example questions:
  - What's the air quality like in Boston today?
  - I have asthma. Can I jog in Seattle?
  - Compare air quality in Dhaka and New York
  - Should I take my kids to the park in London?

Type 'quit' to exit
============================================================

💬 You: I want to go cycling in Boston. Is it safe?

🤔 Thinking...

🤖 AI Assistant:
Great choice! Boston's air quality is currently excellent with 
an AQI of 47 (Good category). This is perfect for cycling!

Here's what you need to know:
• Air quality: Safe for all outdoor activities
• PM2.5: Very low (0.7 µg/m³)
• No special precautions needed

Cycling Recommendations:
✓ Go ahead with your ride - conditions are ideal
✓ You can ride at any intensity level
✓ No need to wear a mask
✓ Perfect day for a long ride or training session

Stay hydrated and enjoy your ride! 🚴‍♂️

💬 You: quit

👋 Thanks for chatting! Stay safe and breathe easy!
```

## Files

- `aqi_genai_agent.py` - Main Gen AI agent
- `aqi_mcp_server.py` - Core MCP server (used by agent)
- `genai_example_conversations.txt` - Detailed examples
- `README_AQI.md` - Full documentation
- `.env.aqi` - Configuration template

## Features

✨ Natural language understanding  
✨ Personalized health recommendations  
✨ Real-time air quality data  
✨ Multi-city comparisons  
✨ Conversation memory  
✨ Empathetic, actionable advice  
✨ Works without OpenAI (fallback)  

## Get Help

See `README_AQI.md` for:
- Detailed documentation
- Technical architecture
- Use cases
- Troubleshooting

See `genai_example_conversations.txt` for:
- 5 detailed example conversations
- Different scenarios
- Expected responses
