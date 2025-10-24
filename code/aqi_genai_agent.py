"""
Gen AI Air Quality Intelligence Agent
Combines MCP server with OpenAI GPT for natural language conversations
"""

import asyncio
import os
from aqi_mcp_server import AQIMCPServer
import json


class GenAIAQIAgent:
    """Conversational AI agent for air quality intelligence"""
    
    def __init__(self, api_ninjas_key: str, openai_api_key: str = None):
        self.mcp_server = AQIMCPServer(api_ninjas_key)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.conversation_history = []
        
        # Check if OpenAI is available
        try:
            import openai
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            self.has_openai = True
        except (ImportError, Exception) as e:
            print(f"⚠️  OpenAI not available: {e}")
            print("Falling back to rule-based responses")
            self.has_openai = False
    
    async def get_aqi_context(self, query: str) -> str:
        """Extract city names from query and get AQI data"""
        # Simple city extraction (can be enhanced with NLP)
        import re
        
        cities = []
        
        # Look for compare commands
        if 'compare' in query.lower():
            words = query.split()
            cities = [w.strip(',').strip('.').strip('?') for w in words if w[0].isupper() and w.lower() not in ['compare', 'and', 'or']]
        else:
            # Extract capitalized words as potential cities
            capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            cities.extend(capitalized)
            
            # Also look for common city patterns (e.g., "in reston", "reston,")
            # Extract words after "in", "at", "for" that might be cities
            patterns = [
                r'(?:in|at|for)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)',
                r'([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s*,\s*[A-Z]{2}',  # City, ST pattern
            ]
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                cities.extend([m.title() for m in matches if m.lower() not in ['the', 'a', 'an', 'is', 'are', 'was', 'were']])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_cities = []
        for city in cities:
            city_clean = city.strip()
            if city_clean and city_clean not in seen:
                seen.add(city_clean)
                unique_cities.append(city_clean)
        
        context = ""
        
        if unique_cities:
            context += f"Air Quality Data for {len(unique_cities)} location(s):\n\n"
            
            for city in unique_cities[:5]:  # Limit to 5 cities
                aqi_data = await self.mcp_server.get_air_quality(city)
                if aqi_data:
                    category = self.mcp_server.get_health_category(aqi_data.overall_aqi)
                    context += f"**{city}**:\n"
                    context += f"- AQI: {aqi_data.overall_aqi} ({category})\n"
                    context += f"- PM2.5: {aqi_data.pm2_5_concentration:.1f} µg/m³\n"
                    context += f"- PM10: {aqi_data.pm10_concentration:.1f} µg/m³\n"
                    context += f"- O3: {aqi_data.o3_concentration:.1f} µg/m³\n"
                    context += f"- Location: ({aqi_data.latitude:.4f}, {aqi_data.longitude:.4f})\n\n"
        
        return context
    
    async def chat(self, user_message: str) -> str:
        """Process user message and generate AI response"""
        
        # Get AQI context if cities are mentioned
        aqi_context = await self.get_aqi_context(user_message)
        
        if not self.has_openai:
            # Fallback to rule-based response
            return await self._rule_based_response(user_message, aqi_context)
        
        # Build system message with context
        system_message = {
            "role": "system",
            "content": """You are an expert Air Quality Intelligence assistant. Your role is to:
1. Provide accurate, helpful information about air quality
2. Offer personalized health recommendations based on AQI levels
3. Give activity-specific advice (jogging, cycling, outdoor activities)
4. Be conversational, friendly, and empathetic
5. Use the provided air quality data to give specific, actionable advice

Health Categories:
- Good (0-50): Safe for all activities
- Moderate (51-100): Generally acceptable, sensitive individuals should monitor
- Unhealthy for Sensitive Groups (101-150): Sensitive groups limit outdoor exertion
- Unhealthy (151-200): Everyone should avoid prolonged outdoor activities
- Very Unhealthy (201-300): Health alert, everyone affected
- Hazardous (301+): Emergency conditions

Always prioritize user health and safety in your recommendations."""
        }
        
        # Add AQI context if available
        if aqi_context:
            context_message = {
                "role": "system",
                "content": f"Current Air Quality Data:\n\n{aqi_context}"
            }
            messages = [system_message, context_message] + self.conversation_history[-6:] + [
                {"role": "user", "content": user_message}
            ]
        else:
            messages = [system_message] + self.conversation_history[-6:] + [
                {"role": "user", "content": user_message}
            ]
        
        try:
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            assistant_message = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # Keep only last 10 exchanges
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return assistant_message
            
        except Exception as e:
            return f"⚠️  Error generating AI response: {e}\n\nFalling back to basic response:\n{aqi_context}"
    
    async def _rule_based_response(self, user_message: str, aqi_context: str) -> str:
        """Fallback rule-based response when OpenAI is unavailable"""
        if aqi_context:
            return f"Here's the air quality information I found:\n\n{aqi_context}\n\nFor personalized AI recommendations, please set up OpenAI API key."
        else:
            return "I couldn't find air quality data in your message. Please mention a city name (e.g., 'What's the air quality in Boston?')"


async def interactive_genai_demo():
    """Interactive Gen AI demo"""
    
    # Get API keys
    api_ninjas_key = os.getenv("API_NINJAS_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_ninjas_key:
        print("\n⚠️  API_NINJAS_KEY not set")
        print("Please set: export API_NINJAS_KEY='your_key'")
        return
    
    if not openai_api_key:
        print("\n⚠️  OPENAI_API_KEY not set - using fallback mode")
        print("For full Gen AI experience, set: export OPENAI_API_KEY='your_key'")
        print("Get your key at: https://platform.openai.com/api-keys")
        print("\n" + "="*60)
    
    agent = GenAIAQIAgent(api_ninjas_key, openai_api_key)
    
    print("\n🤖 GEN AI AIR QUALITY INTELLIGENCE AGENT")
    print("="*60)
    print("Ask me anything about air quality! I use real-time data")
    print("and AI to provide personalized recommendations.")
    print("\nExample questions:")
    print("  - What's the air quality like in Boston today?")
    print("  - I have asthma. Can I jog in Seattle?")
    print("  - Compare air quality in Dhaka and New York")
    print("  - Should I take my kids to the park in London?")
    print("\nType 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for chatting! Stay safe and breathe easy!")
                break
            
            if not user_input:
                continue
            
            print("\n🤔 Thinking...")
            response = await agent.chat(user_input)
            print(f"\n🤖 AI Assistant:\n{response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for chatting! Stay safe and breathe easy!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def demo_conversations():
    """Demo with preset conversations"""
    
    api_ninjas_key = os.getenv("API_NINJAS_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_ninjas_key:
        print("⚠️  Please set API_NINJAS_KEY")
        return
    
    agent = GenAIAQIAgent(api_ninjas_key, openai_api_key)
    
    print("\n🤖 GEN AI AIR QUALITY DEMO CONVERSATIONS")
    print("="*60)
    
    demo_queries = [
        "What's the air quality in Boston right now?",
        "I want to go jogging in Dhaka. Is it safe?",
        "Compare the air quality between Seattle and Los Angeles",
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"Demo {i}/3")
        print(f"{'='*60}")
        print(f"💬 User: {query}")
        print("\n🤔 AI thinking...")
        
        response = await agent.chat(query)
        print(f"\n🤖 AI Assistant:\n{response}")
        
        if i < len(demo_queries):
            await asyncio.sleep(2)  # Pause between demos


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_conversations())
    else:
        asyncio.run(interactive_genai_demo())
