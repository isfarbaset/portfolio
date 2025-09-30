// Advanced Conversational AI System for Isfar's Portfolio
class IsfarChatbotAI {
  constructor() {
    this.conversationHistory = [];
    this.context = {
      currentTopic: null,
      subtopics: [],
      lastCategory: null,
      conversationDepth: 0,
      userInterests: [],
      askedFollowups: [],
      pendingPromise: null
    };
    this.isProcessing = false;
    this.messageCount = 0;
    
    console.log('🚀 IsfarChatbotAI constructor called');
    
    // Initialize advanced NLP components
    this.initializeAdvancedNLP();
    
    // Comprehensive knowledge base
    this.knowledgeBase = {
      skills: {
        keywords: ['skills', 'technologies', 'programming', 'languages', 'tools', 'python', 'sql', 'r', 'tableau', 'plotly', 'machine', 'learning', 'technical'],
        response: "Isfar's technical skills include Python, SQL, R, Plotly, Tableau, machine learning, and database architecture. She specializes in translating technical concepts into business insights.",
        followups: [
          { text: "Would you like to know how she applies these technical skills to solve real business problems?", promise: "technical_applications" },
          { text: "Are you curious about her learning journey and staying current with new technologies?", promise: "professional_growth" }
        ]
      },
      
      projects: {
        keywords: ['projects', 'project', 'portfolio', 'built', 'created', 'developed', 'academic', 'work', 'github', 'code', 'implementation', 'application', 'system', 'model', 'analysis', 'interesting', 'impressive', 'worked'],
        response: "Isfar has built impressive projects spanning AI, data analysis, and visualization. Key projects include: FMBench Assistant (production AI chatbot with RAG architecture), Reddit Sentiment Analysis (processed 1+ billion posts), Spotify Music Classification (machine learning with 50K tracks), Air Quality Intelligence Agent (MCP + Letta AI), and Climate Impact Visualization (40+ years of data). Each demonstrates end-to-end technical capabilities.",
        followups: [
          { text: "Would you like technical details about the FMBench AI Assistant with RAG architecture?", promise: "fmbench_project" },
          { text: "Are you curious about the billion-post Reddit sentiment analysis project?", promise: "reddit_project" },
          { text: "Want to hear about the Spotify music classification using machine learning?", promise: "spotify_project" },
          { text: "Interested in the Air Quality Intelligence Agent with advanced AI architecture?", promise: "aqi_project" }
        ]
      }
    };
  }

  initializeAdvancedNLP() {
    console.log('🔧 Initializing NLP components');
    // Simplified for testing
  }

  // ULTRA SIMPLE TEST VERSION
  generateResponse(userInput) {
    console.log('🔍 TESTING - generateResponse called with:', userInput);
    
    const lowerInput = userInput.toLowerCase().trim();
    
    // Direct pattern matching for immediate testing
    if (lowerInput.includes('skills') || lowerInput.includes('what are her skills')) {
      console.log('✅ SKILLS MATCH DETECTED');
      return "✅ WORKING! Isfar's technical skills include Python, SQL, R, Plotly, Tableau, machine learning, and database architecture. She specializes in translating technical concepts into business insights.";
    }
    
    if (lowerInput.includes('projects') || lowerInput.includes('interesting projects')) {
      console.log('✅ PROJECTS MATCH DETECTED');
      return "✅ WORKING! She's worked on some really exciting projects! Key projects include: FMBench Assistant (production AI chatbot), Reddit Sentiment Analysis (1+ billion posts), Spotify Music Classification, Air Quality Intelligence Agent, and Climate Impact Visualization.";
    }
    
    if (lowerInput.includes('experience')) {
      return "✅ WORKING! Isfar is a Data Analyst II at Shift Digital with nearly 5 years of tech experience, previously at Array of Engineers.";
    }
    
    if (lowerInput.includes('education')) {
      return "✅ WORKING! She has a BS in Computer Science (Grand Valley, 2015-2020) and MS in Data Science (Georgetown, 2023-2025, 4.0 GPA).";
    }
    
    // Default test response
    return `✅ EXTERNAL JS WORKING! You asked: "${userInput}" - The chatbot is now functioning with external JavaScript!`;
  }
}

// Initialize the chatbot
console.log('🚀 Initializing chatbot...');
const chatbotAI = new IsfarChatbotAI();

// Message handling functions
function addMessage(content, isBot = false, withTyping = false) {
  console.log('📝 Adding message:', content, 'isBot:', isBot);
  const messagesContainer = document.querySelector('.chat-messages');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${isBot ? 'bot-message' : 'user-message'}`;
  
  const messageContent = document.createElement('div');
  messageContent.className = 'message-content';
  
  if (isBot && withTyping) {
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Typing animation
    let i = 0;
    const typeWriter = () => {
      if (i < content.length) {
        messageContent.textContent += content.charAt(i);
        i++;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        setTimeout(typeWriter, 20 + Math.random() * 30);
      }
    };
    
    setTimeout(typeWriter, 300);
  } else {
    messageContent.textContent = content;
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
  
  return messageDiv;
}

function addTypingIndicator() {
  const messagesContainer = document.querySelector('.chat-messages');
  const typingDiv = document.createElement('div');
  typingDiv.className = 'message bot-message typing-indicator';
  typingDiv.innerHTML = '<div class="message-content"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
  messagesContainer.appendChild(typingDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
  return typingDiv;
}

function removeTypingIndicator(typingDiv) {
  if (typingDiv && typingDiv.parentNode) {
    typingDiv.parentNode.removeChild(typingDiv);
  }
}

// Main message handling
function handleUserMessage(message) {
  console.log('💬 Handling user message:', message);
  if (!message.trim()) return;
  
  if (chatbotAI.isProcessing) return;
  
  chatbotAI.isProcessing = true;
  
  // Add user message
  addMessage(message, false);
  
  // Show thinking indicator
  const typingIndicator = addTypingIndicator();
  
  // Calculate realistic thinking time
  const thinkingTime = 800 + (message.length * 30) + (Math.random() * 600);
  
  setTimeout(() => {
    try {
      removeTypingIndicator(typingIndicator);
      const response = chatbotAI.generateResponse(message);
      addMessage(response, true, true);
    } catch (error) {
      console.error('❌ Response error:', error);
      removeTypingIndicator(typingIndicator);
      addMessage("I apologize, but I encountered an issue. Please try again!", true);
    } finally {
      chatbotAI.isProcessing = false;
    }
  }, thinkingTime);
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
  console.log('🎯 DOM Content Loaded - Setting up event listeners');
  
  const chatInput = document.getElementById('chat-input');
  const sendButton = document.getElementById('send-button');
  const suggestionChips = document.querySelectorAll('.suggestion-chip');
  
  if (!chatInput || !sendButton) {
    console.error('❌ Critical elements not found!');
    return;
  }
  
  function sendMessage() {
    console.log('📤 Send message triggered');
    const message = chatInput.value.trim();
    if (message && !chatbotAI.isProcessing) {
      handleUserMessage(message);
      chatInput.value = '';
      chatInput.style.height = '56px'; // Reset height
    }
  }
  
  sendButton.addEventListener('click', sendMessage);
  
  chatInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  
  chatInput.addEventListener('input', function() {
    const value = this.value.trim();
    sendButton.disabled = !value || chatbotAI.isProcessing;
    
    // Auto-resize textarea
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    
    // Ensure minimum height
    if (this.scrollHeight < 56) {
      this.style.height = '56px';
    }
  });
  
  suggestionChips.forEach(chip => {
    chip.addEventListener('click', function() {
      const question = this.textContent;
      if (!chatbotAI.isProcessing) {
        chatInput.value = question;
        setTimeout(() => sendMessage(), 100);
      }
    });
  });
  
  console.log('✅ Advanced Isfar Chatbot initialized successfully with external JavaScript!');
});
