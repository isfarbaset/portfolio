// ADVANCED AI CHATBOT - NLP-Enhanced with Comprehensive Knowledge Base and Guardrails
console.log('🚀 Advanced External JavaScript file loaded!');

class IsfarChatbotAI {
  constructor() {
    this.isProcessing = false;
    this.conversationContext = {
      lastTopic: null,
      askedTopics: [],
      entities: [],
      sentiment: 'neutral',
      questionType: null
    };
    
    // Advanced • NLP: Common word variations and synonyms
    this.synonymMap = {
      'proficient': ['skilled', 'expert', 'good at', 'knows', 'familiar with', 'experienced in'],
      'work': ['job', 'position', 'role', 'career', 'employment', 'occupation'],
      'education': ['academic', 'degree', 'study', 'school', 'university', 'qualification'],
      'create': ['build', 'develop', 'make', 'construct', 'design', 'implement'],
      'skill': ['ability', 'capability', 'expertise', 'competency', 'proficiency'],
      'like': ['enjoy', 'love', 'prefer', 'interested in', 'passionate about'],
      'where': ['location', 'place', 'based', 'live', 'from'],
      'why': ['reason', 'because', 'motivation', 'purpose'],
      'how': ['method', 'way', 'process', 'approach'],
      'what': ['which', 'describe', 'tell me about']
    };
    
    // • NLP: Question word patterns
    this.questionPatterns = {
      who: /\b(who|whom)\b/i,
      what: /\b(what|which)\b/i,
      where: /\b(where|location)\b/i,
      when: /\b(when|time|date)\b/i,
      why: /\b(why|reason|because)\b/i,
      how: /\b(how|method|way)\b/i,
      can: /\b(can|could|able)\b/i,
      does: /\b(does|do|did)\b/i,
      is: /\b(is|are|was|were)\b/i
    };
    
    // • NLP: Stopwords for better keyword extraction
    this.stopwords = new Set([
      'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
      'could', 'can', 'may', 'might', 'must', 'shall', 'about', 'me', 'you',
      'he', 'she', 'it', 'we', 'they', 'them', 'their', 'this', 'that',
      'these', 'those', 'i', 'my', 'your', 'his', 'her', 'its', 'our',
      'and', 'or', 'but', 'if', 'then', 'so', 'than', 'such', 'no', 'not',
      'only', 'own', 'same', 'too', 'very', 'just', 'tell', 'know', 'want'
    ]);
    
    // GUARDRAILS: Inappropriate content detection
    this.inappropriatePatterns = {
      romantic: {
        patterns: ['date', 'dating', 'marry', 'marriage', 'relationship', 'single', 'boyfriend', 'girlfriend', 'romance', 'romantic', 'attractive', 'hot', 'sexy', 'beautiful', 'pretty', 'cute', 'gorgeous'],
        response: "I'm here to share information about Isfar's professional background and achievements. For professional networking, please connect via LinkedIn at linkedin.com/in/isfarbaset."
      },
      personal: {
        patterns: ['address', 'home address', 'street', 'apartment', 'phone number', 'cell phone', 'mobile number', 'telephone', 'social security', 'ssn', 'bank', 'credit card', 'salary', 'how much earn', 'how much make', 'income', 'age', 'how old', 'weight', 'height', 'religion', 'religious', 'political', 'politics', 'vote', 'democrat', 'republican'],
        response: "I respect privacy and don't share personal information. I can tell you about Isfar's professional experience, education, skills, and public projects. What would you like to know about her professional work?"
      },
      offensive: {
        patterns: ['hate', 'stupid', 'dumb', 'idiot', 'moron', 'loser', 'suck', 'terrible person', 'awful person', 'worst', 'garbage', 'trash', 'pathetic', 'useless'],
        response: "I'm designed to have respectful, professional conversations. I'd be happy to share information about Isfar's technical expertise, projects, or career journey. What would you like to learn about?"
      },
      tooPersonal: {
        patterns: ['married', 'husband', 'wife', 'spouse', 'partner', 'children', 'kids', 'baby', 'pregnant', 'pregnancy', 'family', 'parents', 'mother', 'father', 'sexual', 'orientation', 'health', 'medical', 'disease', 'illness', 'mental health', 'therapy', 'medication', 'disability'],
        response: "That's outside the scope of what I can discuss. I focus on Isfar's professional accomplishments, technical skills, and career development. Is there something specific about her work experience or projects you'd like to know?"
      },
      financial: {
        patterns: ['net worth', 'how much money', 'rich', 'wealth', 'wealthy', 'assets', 'finances', 'debt', 'loan', 'mortgage', 'investment'],
        response: "I don't have access to personal financial information. I can share details about Isfar's professional experience, technical projects, and career achievements instead. What interests you?"
      },
      illegal: {
        patterns: ['hack', 'crack', 'steal', 'illegal', 'cheat', 'fraud', 'scam', 'password', 'breach', 'exploit', 'bypass'],
        response: "I can't help with that. I'm here to provide information about Isfar's professional background and technical expertise in data science and AI. What would you like to know about her work?"
      },
      spam: {
        patterns: ['buy', 'purchase', 'sell', 'sale', 'discount', 'cheap', 'offer', 'deal', 'subscribe', 'sign up', 'free money', 'click here', 'download now'],
        response: "This chatbot is designed to share information about Isfar's professional background. I'm not equipped to handle commercial requests. Would you like to learn about her technical skills or projects instead?"
      }
    };
    
    // Comprehensive knowledge base
    this.knowledgeBase = {
      hobbies: {
        patterns: ['hobby', 'hobbies', 'free time', 'leisure', 'fun', 'do for fun', 'outside work', 'personal', 'interest', 'interests', 'passion', 'passions', 'enjoy', 'like to do', 'downtime', 'spare time'],
        response: "Isfar loves spending time in charming cafes and exploring new cuisines. During her downtime, she enjoys reading, painting, and watching movies/TV shows.",
        category: 'personal'
      },
      
      skills: {
        patterns: ['skill', 'skills', 'skils', 'technical', 'technology', 'technologies', 'tool', 'tools', 'programming', 'language', 'languages', 'stack', 'proficient', 'good at', 'know', 'expertise', 'capabilities'],
        response: "Isfar's technical skills include:\n\n• Programming: Python, SQL, R, Java, C++\n\n• Machine Learning: Scikit-learn, TensorFlow, PyTorch, XGBoost\n\n• Data Visualization: Tableau, Plotly, Seaborn\n\n• Big Data & Cloud: Apache Spark, Databricks, AWS SageMaker\n\n• NLP: spaCy, Hugging Face, BERT\n\n• Databases: MySQL, PostgreSQL, MongoDB, Snowflake\n\n• Tools & Methods: Docker, Flask, Git, Quarto, A/B Testing, Bayesian Analysis\n\nShe specializes in leveraging data-driven solutions and communicating complex insights to non-technical stakeholders.",
        category: 'technical'
      },
      
      projects: {
        patterns: ['project', 'projects', 'work on', 'built', 'created', 'developed', 'portfolio', 'showcase', 'work'],
        response: "Isfar has worked on several exciting projects:\n\n🤖 FMBench Assistant - A conversational AI assistant for Foundation Model Benchmarking built with Amazon Bedrock, AWS Lambda, and LangGraph\n\n🎵 Wicked Spotify Analysis - Analyzed 20 years of streaming data from all 28 Wicked tracks. Key finding: song length has zero impact on popularity!\n\n🌍 US Insights - Analyzing sentiment patterns across U.S. states based on Reddit conversations\n\n🌡️ Temp Talk - Exploring climate trends in Southeastern Utah National Parks and their impacts on ecosystems\n\n🎼 Beats and Bytes - Exploring the intersection of music and machine learning through data analysis and predictive models\n\n🚗 EV Insights - Examining EVs' environmental impact using Naïve Bayes, clustering, decision trees and ARM\n\n🌫️ Air Quality Intelligence - A Gen AI conversational agent for real-time air quality monitoring with personalized health recommendations using GPT-4 and MCP",
        category: 'technical'
      },
      
      experience: {
        patterns: ['experience', 'work', 'job', 'career', 'position', 'role', 'working', 'worked', 'employment', 'professional', 'company', 'employer'],
        response: "Isfar is currently a Data Analyst II at Shift Digital in the Data Operations and Analytics team, where she leverages data-driven solutions for enterprise clients.\n\nKey Achievements:\n• Leading advanced data operations and analytics initiatives\n• Improved data accuracy by 25% across 10+ enterprise projects\n• Saved 15 hours weekly by automating billing data processing\n• Streamlined operations for 12 high-value clients\n• Achieved near 100% customer approval rate on technical proposals\n• Enhanced data visibility with interactive dashboards\n\nPrevious Experience:\n• Data Analyst I at Shift Digital (Nov 2021 - May 2023)\n• Computer Engineer at Array of Engineers (Jul 2020 - Nov 2021) - Verified Airbus A350 avionics software and optimized web apps",
        category: 'professional'
      },
      
      education: {
        patterns: ['education', 'degree', 'school', 'university', 'college', 'study', 'studied', 'georgetown', 'grand valley', 'gvsu', 'gpa', 'academic', 'masters', 'bachelor', 'coursework', 'courses'],
        response: "Isfar has an exceptional academic background:\n\n🎓 Master of Science in Data Science and Analytics\nGeorgetown University, Washington D.C. (2023-2025)\n• Perfect 4.0 GPA\n• Completed while working full-time\n• Key Coursework: Probabilistic Modeling, Database Systems, Advanced Data Visualization, Data Ethics, Computational Linguistics, Machine Learning Deployment, Digital Storytelling, Big Data & Cloud Computing, Applied Generative AI\n• Leadership: Social Committee Member & Lead Mentor\n\n🎓 Bachelor of Science in Computer Science\nGrand Valley State University, Grand Rapids MI (2015-2020)\n• Strong foundation in computer science and programming",
        category: 'education'
      },
      
      background: {
        patterns: ['background', 'from', 'where', 'origin', 'hometown', 'born', 'grew up', 'heritage', 'culture', 'bangladesh', 'dhaka', 'story', 'about'],
        response: "Isfar is originally from Dhaka, the vibrant capital city of Bangladesh. She came to the US in 2015 to pursue her bachelor's degree in Computer Science and lived in West Michigan before moving to Northern Virginia in 2022.\n\nHer passion lies in leveraging tailored data and algorithms to solve intricate business problems and communicating complex insights to non-technical stakeholders.",
        category: 'personal'
      },
      
      location: {
        patterns: ['location', 'where live', 'based', 'reside', 'living', 'virginia', 'reston', 'va', 'dmv', 'live now', 'currently'],
        response: "Isfar currently lives in Northern Virginia (moved here in 2022). She previously lived in West Michigan during her undergraduate studies at Grand Valley State University.",
        category: 'personal'
      },
      
      contact: {
        patterns: ['contact', 'reach', 'email', 'linkedin', 'connect', 'get in touch', 'message', 'talk to', 'hire', 'hiring'],
        response: "I'd be happy to help you connect with Isfar!\n\n• LinkedIn: linkedin.com/in/isfarbaset (best for professional inquiries)\n• Email: Available on her resume/portfolio\n• Portfolio: isfarbaset.github.io/portfolio\n\nFeel free to reach out for collaboration opportunities, technical discussions, or professional networking!",
        category: 'contact'
      },
      
      ai_ml: {
        patterns: ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning', 'neural', 'model', 'models', 'nlp', 'rag', 'langchain', 'tensorflow', 'pytorch', 'bedrock', 'generative ai'],
        response: "Isfar has hands-on experience in AI/ML:\n\n• Built production RAG systems with Amazon Bedrock and LangGraph (FMBench Assistant)\n• Developed conversational AI agents for real-time monitoring (Air Quality Intelligence)\n• Machine Learning tools: Scikit-learn, TensorFlow, PyTorch, XGBoost\n• NLP expertise: spaCy, Hugging Face, BERT\n• Coursework in Applied Generative AI, Computational Linguistics, and Machine Learning Deployment\n\nHer projects showcase practical applications of cutting-edge AI technologies.",
        category: 'technical'
      },
      
      data_analysis: {
        patterns: ['data analysis', 'analytics', 'analyze', 'data science', 'insights', 'visualization', 'dashboard', 'tableau', 'plotly', 'databricks', 'spark', 'etl', 'pipeline'],
        response: "Data analysis is at the core of what Isfar does:\n\n• Expert in exploratory data analysis and statistical modeling\n• Creates compelling visualizations using Tableau, Plotly, Seaborn\n• Big Data tools: Apache Spark, Databricks\n• Improved data accuracy by 25% through optimized ETL processes\n• Built interactive dashboards for stakeholder communication\n• Managed data transformation within Databricks, creating efficient pipelines\n• Strong foundation in A/B Testing and Bayesian Analysis\n\nShe specializes in translating data into actionable business insights for non-technical stakeholders.",
        category: 'technical'
      },
      
      aws_cloud: {
        patterns: ['aws', 'amazon', 'cloud', 'bedrock', 'lambda', 's3', 'serverless', 'sagemaker', 'cloud computing'],
        response: "Isfar has hands-on AWS experience:\n\n• Amazon Bedrock for LLM integration (FMBench Assistant project)\n• AWS Lambda for serverless computing\n• AWS SageMaker for machine learning deployment\n• Coursework in Big Data & Cloud Computing\n• Built production-ready conversational AI with AWS services\n\nHer FMBench Assistant demonstrates sophisticated cloud architecture using multiple AWS services.",
        category: 'technical'
      },
      
      personality: {
        patterns: ['personality', 'like', 'person', 'who is', 'describe', 'about herself', 'type of person', 'character', 'what is she like'],
        response: "Isfar is passionate about leveraging tailored data and algorithms to solve intricate business problems. She combines technical excellence with strong communication skills, making complex insights accessible to non-technical stakeholders.\n\nShe's detail-oriented, collaborative, and thrives in environments that value innovation and data-driven decision making.",
        category: 'personal'
      },
      
      achievements: {
        patterns: ['achievement', 'achievements', 'accomplish', 'proud', 'success', 'award', 'recognition', 'accomplishment'],
        response: "Some of Isfar's notable achievements:\n\n🏆 Perfect 4.0 GPA in her Master's program at Georgetown University\n💼 Improved data accuracy by 25% across 10+ enterprise projects\n⏱️ Saved 15 hours weekly by automating billing data processing\n📊 Achieved near 100% customer approval rate on technical proposals\n🎓 Completed Master's degree while working full-time\n💻 Increased client retention by 25% through web app optimization\n📈 Achieved 98% on-time delivery rate for software testing workflows\n👥 Served as Lead Mentor in Georgetown's Data Science program",
        category: 'achievements'
      },
      
      goals: {
        patterns: ['goal', 'goals', 'future', 'plan', 'aspire', 'want to', 'next', 'ambition', 'career path', 'looking for', 'seeking'],
        response: "Isfar is focused on continuing to leverage data-driven solutions to solve complex business problems:\n\n• Building production-ready AI/ML systems with real-world impact\n• Deepening expertise in advanced machine learning and generative AI\n• Leading data science initiatives and mentoring team members\n• Contributing to innovative projects that bridge research and practical applications\n• Continuing to grow in roles that combine technical challenge with meaningful business impact",
        category: 'professional'
      }
    };;

    console.log('✅ IsfarChatbotAI initialized with NLP, guardrails, and 14+ knowledge domains');
  }

  // GUARDRAILS: Check for inappropriate content FIRST
  checkInappropriateContent(userInput) {
    const input = userInput.toLowerCase().trim();
    
    for (const [category, data] of Object.entries(this.inappropriatePatterns)) {
      for (const pattern of data.patterns) {
        const regex = new RegExp('\\b' + pattern + '\\b', 'i');
        if (regex.test(input)) {
          console.log('🚫 Guardrail triggered:', category, pattern);
          return { isInappropriate: true, category, response: data.response };
        }
      }
    }
    return { isInappropriate: false };
  }

  // • NLP: Tokenize and clean
  tokenize(text) {
    return text.toLowerCase().replace(/[^\w\s'-]/g, ' ').split(/\s+/).filter(word => word.length > 0);
  }

  // • NLP: Extract keywords (remove stopwords)
  extractKeywords(text) {
    return this.tokenize(text).filter(word => !this.stopwords.has(word));
  }

  // • NLP: Detect question type
  detectQuestionType(text) {
    for (const [type, pattern] of Object.entries(this.questionPatterns)) {
      if (pattern.test(text)) return type;
    }
    return 'statement';
  }

  // • NLP: Basic sentiment analysis
  analyzeSentiment(text) {
    const positive = ['love', 'great', 'awesome', 'excellent', 'amazing', 'wonderful', 'fantastic', 'good', 'like', 'enjoy', 'impressed', 'interested'];
    const negative = ['hate', 'bad', 'terrible', 'awful', 'worst', 'dislike', 'poor', 'disappointed'];
    const words = this.tokenize(text);
    let score = 0;
    words.forEach(word => {
      if (positive.includes(word)) score += 1;
      if (negative.includes(word)) score -= 1;
    });
    if (score > 0) return 'positive';
    if (score < 0) return 'negative';
    return 'neutral';
  }

  // Advanced pattern matching with scoring
  findBestMatch(userInput) {
    const input = userInput.toLowerCase().trim();
    this.conversationContext.questionType = this.detectQuestionType(input);
    this.conversationContext.sentiment = this.analyzeSentiment(input);
    
    let bestMatch = null;
    let highestScore = 0;
    
    for (const [key, data] of Object.entries(this.knowledgeBase)) {
      let score = 0;
      
      // Pattern matching
      for (const pattern of data.patterns) {
        if (input.includes(pattern)) {
          score += pattern.length * 3;
          const regex = new RegExp('\\b' + pattern + '\\b', 'i');
          if (regex.test(input)) score += 15;
        }
      }
      
      // Context bonus
      if (this.conversationContext.lastTopic === key) score += 5;
      
      // Question type alignment
      if (this.conversationContext.questionType === 'what' && ['skills', 'projects', 'experience', 'education'].includes(key)) score += 3;
      if (this.conversationContext.questionType === 'where' && ['location', 'background'].includes(key)) score += 5;
      
      if (score > highestScore) {
        highestScore = score;
        bestMatch = { key, data, score };
      }
    }
    
    console.log('🎯 Match:', bestMatch?.key, 'Score:', bestMatch?.score, 'Q-Type:', this.conversationContext.questionType, 'Sentiment:', this.conversationContext.sentiment);
    return highestScore > 5 ? bestMatch : null;
  }

  // Generate response with guardrails and NLP
  generateResponse(userInput) {
    console.log('🔍 Processing:', userInput);
    
    // STEP 1: Check guardrails FIRST
    const guardrailCheck = this.checkInappropriateContent(userInput);
    if (guardrailCheck.isInappropriate) {
      return guardrailCheck.response;
    }
    
    // STEP 2: Find best knowledge match
    const match = this.findBestMatch(userInput);
    
    if (match) {
      this.conversationContext.lastTopic = match.key;
      this.conversationContext.askedTopics.push(match.key);
      
      let response = match.data.response;
      
      // Add positive sentiment prefix
      if (this.conversationContext.sentiment === 'positive') {
        const prefix = ["I'm glad you're interested! ", "Great question! ", "Happy to share! "][Math.floor(Math.random() * 3)];
        response = prefix + response;
      }
      
      return response;
    }
    
    // STEP 3: Helpful fallback
    const suggestedTopics = this.conversationContext.askedTopics.length > 0
      ? "You could also ask about her hobbies, goals, achievements, or specific technologies!"
      : "You can ask me about her skills, projects, experience, education, background, hobbies, goals, or achievements!";
    
    return "I'd love to help you learn about Isfar! " + suggestedTopics + "\n\nWhat would you like to know?";
  }
}

// Initialize chatbot
console.log('🚀 Initializing advanced chatbot...');
const chatbotAI = new IsfarChatbotAI();

// Message handling
function addMessage(content, isBot = false, withTyping = false) {
  console.log('📝 Adding message:', content.substring(0, 50), '...');
  const messagesContainer = document.querySelector('.chat-messages');
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message ' + (isBot ? 'bot-message' : 'user-message');
  
  const messageContent = document.createElement('div');
  messageContent.className = 'message-content';
  
  if (isBot && withTyping) {
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
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
  if (typingDiv && typingDiv.parentNode) typingDiv.parentNode.removeChild(typingDiv);
}

function handleUserMessage(message) {
  console.log('💬 User message:', message);
  if (!message.trim() || chatbotAI.isProcessing) return;
  
  chatbotAI.isProcessing = true;
  addMessage(message, false);
  
  const typingIndicator = addTypingIndicator();
  const thinkingTime = 800 + (message.length * 30) + (Math.random() * 600);
  
  setTimeout(() => {
    try {
      removeTypingIndicator(typingIndicator);
      const response = chatbotAI.generateResponse(message);
      addMessage(response, true, true);
    } catch (error) {
      console.error('❌ Error:', error);
      removeTypingIndicator(typingIndicator);
      addMessage("I apologize, but I encountered an issue. Please try again!", true);
    } finally {
      chatbotAI.isProcessing = false;
    }
  }, thinkingTime);
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
  console.log('�� DOM loaded - setting up chatbot...');
  
  const chatInput = document.getElementById('chat-input');
  const sendButton = document.getElementById('send-button');
  const suggestionChips = document.querySelectorAll('.suggestion-chip');
  
  if (!chatInput || !sendButton) {
    console.error('❌ Required elements not found!');
    return;
  }
  
  function sendMessage() {
    const message = chatInput.value.trim();
    if (message && !chatbotAI.isProcessing) {
      handleUserMessage(message);
      chatInput.value = '';
      chatInput.style.height = '56px';
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
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    if (this.scrollHeight < 56) this.style.height = '56px';
  });
  
  suggestionChips.forEach(chip => {
    chip.addEventListener('click', function() {
      const question = this.textContent.trim();
      if (!chatbotAI.isProcessing) {
        chatInput.value = question;
        setTimeout(() => sendMessage(), 100);
      }
    });
  });
  
  console.log('✅ Advanced chatbot ready! NLP + Guardrails + 14 domains active');
});
