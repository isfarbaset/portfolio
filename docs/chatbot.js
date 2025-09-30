// SIMPLE WORKING CHATBOT - Direct approach
console.log('🚀 External JavaScript file loaded!');

class IsfarChatbotAI {
  constructor() {
    this.isProcessing = false;
    console.log('✅ IsfarChatbotAI initialized with external JS');
  }

  generateResponse(userInput) {
    console.log('� generateResponse called with:', userInput);
    
    const input = userInput.toLowerCase().trim();
    
    // Direct matching - no complex logic
    if (input.includes('skills') || input.includes('what are her skills')) {
      return "Isfar's technical skills include Python, SQL, R, Plotly, Tableau, machine learning, and database architecture. She specializes in translating technical concepts into business insights.";
    }
    
    if (input.includes('projects') || input.includes('work on')) {
      return "She's worked on some really exciting projects! Key projects include: FMBench Assistant (production AI chatbot with RAG architecture), Reddit Sentiment Analysis (processed 1+ billion posts), Spotify Music Classification (machine learning with 50K tracks), Air Quality Intelligence Agent (MCP + Letta AI), and Climate Impact Visualization (40+ years of data).";
    }
    
    if (input.includes('experience')) {
      return "Isfar is a Data Analyst II at Shift Digital with nearly 5 years of tech experience. She previously worked as a Computer Engineer at Array of Engineers (2020-2021).";
    }
    
    if (input.includes('education')) {
      return "She has a Bachelor's in Computer Science from Grand Valley State University (2015-2020) and a Master's in Data Science from Georgetown University (2023-2025) with a perfect 4.0 GPA.";
    }
    
    if (input.includes('background')) {
      return "Isfar is from Dhaka, Bangladesh and now lives in Reston, VA. She's built a successful career in data science with a unique multicultural perspective.";
    }
    
    // Default response
    return "I can tell you about Isfar's skills, projects, experience, education, or background. What would you like to know?";
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
