// ADVANCED AI CHATBOT - NLP-Enhanced with Comprehensive Knowledge Base
console.log('🚀 External JavaScript file loaded!');

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
    
    // Advanced NLP: Common word variations and synonyms
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
    
    // NLP: Question word patterns
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
    
    // NLP: Stopwords for better keyword extraction
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
      // Romantic/Dating
      romantic: {
        patterns: ['date', 'dating', 'marry', 'marriage', 'relationship', 'single', 'boyfriend', 'girlfriend', 'romance', 'romantic', 'attractive', 'hot', 'sexy', 'beautiful', 'pretty', 'cute', 'gorgeous'],
        response: "I'm here to share information about Isfar's professional background and achievements. For professional networking, please connect via LinkedIn at linkedin.com/in/isfarbaset."
      },
      
      // Personal/Private Information
      personal: {
        patterns: ['address', 'home address', 'street', 'apartment', 'phone number', 'cell phone', 'mobile number', 'telephone', 'social security', 'ssn', 'bank', 'credit card', 'salary', 'how much earn', 'how much make', 'income', 'age', 'how old', 'weight', 'height', 'religion', 'religious', 'political', 'politics', 'vote', 'democrat', 'republican'],
        response: "I respect privacy and don't share personal information. I can tell you about Isfar's professional experience, education, skills, and public projects. What would you like to know about her professional work?"
      },
      
      // Offensive/Inappropriate
      offensive: {
        patterns: ['hate', 'stupid', 'dumb', 'idiot', 'moron', 'loser', 'suck', 'terrible person', 'awful person', 'worst', 'garbage', 'trash', 'pathetic', 'useless'],
        response: "I'm designed to have respectful, professional conversations. I'd be happy to share information about Isfar's technical expertise, projects, or career journey. What would you like to learn about?"
      },
      
      // Overly Personal Questions
      tooPersonal: {
        patterns: ['married', 'husband', 'wife', 'spouse', 'partner', 'children', 'kids', 'baby', 'pregnant', 'pregnancy', 'family', 'parents', 'mother', 'father', 'sexual', 'orientation', 'health', 'medical', 'disease', 'illness', 'mental health', 'therapy', 'medication', 'disability'],
        response: "That's outside the scope of what I can discuss. I focus on Isfar's professional accomplishments, technical skills, and career development. Is there something specific about her work experience or projects you'd like to know?"
      },
      
      // Financial Information
      financial: {
        patterns: ['net worth', 'how much money', 'rich', 'wealth', 'wealthy', 'assets', 'finances', 'debt', 'loan', 'mortgage', 'investment'],
        response: "I don't have access to personal financial information. I can share details about Isfar's professional experience, technical projects, and career achievements instead. What interests you?"
      },
      
      // Illegal/Unethical Requests
      illegal: {
        patterns: ['hack', 'crack', 'steal', 'illegal', 'cheat', 'fraud', 'scam', 'password', 'breach', 'exploit', 'bypass'],
        response: "I can't help with that. I'm here to provide information about Isfar's professional background and technical expertise in data science and AI. What would you like to know about her work?"
      },
      
      // Spam/Inappropriate Requests
      spam: {
        patterns: ['buy', 'purchase', 'sell', 'sale', 'discount', 'cheap', 'offer', 'deal', 'subscribe', 'sign up', 'free money', 'click here', 'download now'],
        response: "This chatbot is designed to share information about Isfar's professional background. I'm not equipped to handle commercial requests. Would you like to learn about her technical skills or projects instead?"
      }
    };
    
    // Comprehensive knowledge base with multiple matching strategies
    this.knowledgeBase = {
      hobbies: {
        patterns: ['hobby', 'hobbies', 'free time', 'leisure', 'fun', 'do for fun', 'outside work', 'personal', 'interest', 'interests', 'passion', 'passions', 'enjoy', 'like to do', 'downtime', 'spare time'],
        response: "Isfar loves spending time in charming cafes and exploring new cuisines. During her downtime, she enjoys reading, painting, and watching movies/TV shows.",
        category: 'personal'
      },
      
      skills: {
        patterns: ['skill', 'skills', 'skils', 'technical', 'technology', 'technologies', 'tool', 'tools', 'programming', 'language', 'languages', 'stack', 'proficient', 'good at', 'know', 'expertise', 'capabilities'],
        response: "Isfar's technical skills include:\n\n**Programming:** Python, SQL, R, Java, C++\n\n**Machine Learning:** Scikit-learn, TensorFlow, PyTorch, XGBoost\n\n**Data Visualization:** Tableau, Plotly, Seaborn\n\n**Big Data & Cloud:** Apache Spark, Databricks, AWS SageMaker\n\n**NLP:** spaCy, Hugging Face, BERT\n\n**Databases:** MySQL, PostgreSQL, MongoDB, Snowflake\n\n**Tools & Methods:** Docker, Flask, Git, Quarto, A/B Testing, Bayesian Analysis\n\nShe specializes in leveraging data-driven solutions and communicating complex insights to non-technical stakeholders.",
        category: 'technical'
      },
      
      projects: {
        patterns: ['project', 'projects', 'work on', 'built', 'created', 'developed', 'portfolio', 'showcase', 'work'],
        response: "Isfar has worked on several exciting projects:\n\n🤖 **FMBench Assistant** - A conversational AI assistant for Foundation Model Benchmarking built with Amazon Bedrock, AWS Lambda, and LangGraph\n\n🎵 **Wicked Spotify Analysis** - Analyzed 20 years of streaming data from all 28 Wicked tracks. Key finding: song length has zero impact on popularity!\n\n🌍 **US Insights** - Analyzing sentiment patterns across U.S. states based on Reddit conversations\n\n🌡️ **Temp Talk** - Exploring climate trends in Southeastern Utah National Parks and their impacts on ecosystems\n\n🎼 **Beats and Bytes** - Exploring the intersection of music and machine learning through data analysis and predictive models\n\n🚗 **EV Insights** - Examining EVs' environmental impact using Naïve Bayes, clustering, decision trees and ARM\n\n🌬️ **Air Quality Intelligence** - A Gen AI conversational agent for real-time air quality monitoring with personalized health recommendations using GPT-4 and MCP",
        category: 'technical'
      },
      
      experience: {
        patterns: ['experience', 'work', 'job', 'career', 'position', 'role', 'working', 'worked', 'employment', 'professional', 'company', 'employer'],
        response: "Isfar is currently a **Data Analyst II** at Shift Digital in the Data Operations and Analytics team, where she leverages data-driven solutions for enterprise clients.\n\n**Key Achievements:**\n• Leading advanced data operations and analytics initiatives\n• Improved data accuracy by 25% across 10+ enterprise projects\n• Saved 15 hours weekly by automating billing data processing\n• Streamlined operations for 12 high-value clients\n• Achieved near 100% customer approval rate on technical proposals\n• Enhanced data visibility with interactive dashboards\n\n**Previous Experience:**\n• Data Analyst I at Shift Digital (Nov 2021 - May 2023)\n• Computer Engineer at Array of Engineers (Jul 2020 - Nov 2021) - Verified Airbus A350 avionics software and optimized web apps",
        category: 'professional'
      },
      
      education: {
        patterns: ['education', 'degree', 'school', 'university', 'college', 'study', 'studied', 'georgetown', 'grand valley', 'gvsu', 'gpa', 'academic', 'masters', 'bachelor', 'coursework', 'courses'],
        response: "Isfar has an exceptional academic background:\n\n🎓 **Master of Science in Data Science and Analytics**\nGeorgetown University, Washington D.C. (2023-2025)\n• Perfect 4.0 GPA\n• Completed while working full-time\n• Key Coursework: Probabilistic Modeling, Database Systems, Advanced Data Visualization, Data Ethics, Computational Linguistics, Machine Learning Deployment, Digital Storytelling, Big Data & Cloud Computing, Applied Generative AI\n• Leadership: Social Committee Member & Lead Mentor\n\n🎓 **Bachelor of Science in Computer Science**\nGrand Valley State University, Grand Rapids MI (2015-2020)\n• Strong foundation in computer science and programming",
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
        response: "Some of Isfar's notable achievements:\n\n🏆 Perfect 4.0 GPA in her Master's program at Georgetown University\n� Improved data accuracy by 25% across 10+ enterprise projects\n⏱️ Saved 15 hours weekly by automating billing data processing\n📊 Achieved near 100% customer approval rate on technical proposals\n🎓 Completed Master's degree while working full-time\n� Increased client retention by 25% through web app optimization\n📈 Achieved 98% on-time delivery rate for software testing workflows\n👥 Served as Lead Mentor in Georgetown's Data Science program",
        category: 'achievements'
      },
      
      goals: {
        patterns: ['goal', 'goals', 'future', 'plan', 'aspire', 'want to', 'next', 'ambition', 'career path', 'looking for', 'seeking'],
        response: "Isfar is focused on continuing to leverage data-driven solutions to solve complex business problems:\n\n• Building production-ready AI/ML systems with real-world impact\n• Deepening expertise in advanced machine learning and generative AI\n• Leading data science initiatives and mentoring team members\n• Contributing to innovative projects that bridge research and practical applications\n• Continuing to grow in roles that combine technical challenge with meaningful business impact",
        category: 'professional'
      }
    };
    
    console.log('✅ IsfarChatbotAI initialized with comprehensive knowledge base');
  }

  // NLP: Tokenize and clean input
  tokenize(text) {
    return text.toLowerCase()
      .replace(/[^\w\s'-]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 0);
  }

  // NLP: Remove stopwords and extract meaningful keywords
  extractKeywords(text) {
    const tokens = this.tokenize(text);
    return tokens.filter(word => !this.stopwords.has(word));
  }

  // NLP: Expand query with synonyms
  expandWithSynonyms(keywords) {
    const expanded = new Set(keywords);
    keywords.forEach(keyword => {
      for (const [base, synonyms] of Object.entries(this.synonymMap)) {
        if (synonyms.includes(keyword) || keyword === base) {
          expanded.add(base);
          synonyms.forEach(syn => expanded.add(syn));
        }
      }
    });
    return Array.from(expanded);
  }

  // NLP: Detect question type for context-aware responses
  detectQuestionType(text) {
    for (const [type, pattern] of Object.entries(this.questionPatterns)) {
      if (pattern.test(text)) {
        return type;
      }
    }
    return 'statement';
  }

  // NLP: Basic sentiment analysis
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

  // NLP: Extract named entities (simple version)
  extractEntities(text) {
    const entities = [];
    const words = text.split(/\s+/);
    
    // Look for capitalized words (potential names/places)
    words.forEach((word, idx) => {
      if (/^[A-Z][a-z]+/.test(word) && !this.stopwords.has(word.toLowerCase())) {
        entities.push(word);
      }
    });
    
    // Look for technologies mentioned
    const techKeywords = ['python', 'sql', 'aws', 'bedrock', 'lambda', 'tableau', 'plotly', 'langchain', 'docker', 'git', 'postgresql', 'mongodb'];
    const lowerText = text.toLowerCase();
    techKeywords.forEach(tech => {
      if (lowerText.includes(tech)) {
        entities.push(tech);
      }
    });
    
    return entities;
  }

  // NLP: Calculate semantic similarity using Jaccard similarity
  calculateSimilarity(set1, set2) {
    const intersection = set1.filter(x => set2.includes(x)).length;
    const union = new Set([...set1, ...set2]).size;
    return union === 0 ? 0 : intersection / union;
  }

  // NLP: Fuzzy matching for typo tolerance
  levenshteinDistance(str1, str2) {
    const matrix = [];
    
    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    return matrix[str2.length][str1.length];
  }

  // NLP: Check if words are similar (typo tolerance)
  isSimilar(word1, word2, threshold = 2) {
    if (word1 === word2) return true;
    if (word1.includes(word2) || word2.includes(word1)) return true;
    return this.levenshteinDistance(word1, word2) <= threshold;
  }

  // GUARDRAILS: Check for inappropriate content
  checkInappropriateContent(userInput) {
    const input = userInput.toLowerCase().trim();
    
    for (const [category, data] of Object.entries(this.inappropriatePatterns)) {
      for (const pattern of data.patterns) {
        // Check for exact word matches to avoid false positives
        const regex = new RegExp(`\\b${pattern}\\b`, 'i');
        if (regex.test(input)) {
          console.log('🚫 Inappropriate content detected:', category, '- pattern:', pattern);
          return {
            isInappropriate: true,
            category: category,
            response: data.response
          };
        }
      }
    }
    
    return { isInappropriate: false };
  }

  // Advanced pattern matching with NLP techniques
  findBestMatch(userInput) {
    const input = userInput.toLowerCase().trim();
    
    // Update conversation context with NLP analysis
    this.conversationContext.questionType = this.detectQuestionType(input);
    this.conversationContext.sentiment = this.analyzeSentiment(input);
    this.conversationContext.entities = this.extractEntities(userInput);
    
    // Extract and expand keywords
    const keywords = this.extractKeywords(input);
    const expandedKeywords = this.expandWithSynonyms(keywords);
    
    let bestMatch = null;
    let highestScore = 0;
    
    for (const [key, data] of Object.entries(this.knowledgeBase)) {
      let score = 0;
      
      // Method 1: Direct pattern matching (high weight)
      for (const pattern of data.patterns) {
        if (input.includes(pattern)) {
          score += pattern.length * 3;
          
          // Exact word boundary matches get bonus
          const regex = new RegExp(`\\b${pattern}\\b`, 'i');
          if (regex.test(input)) {
            score += 15;
          }
        }
        
        // Fuzzy matching for typo tolerance
        const inputWords = this.tokenize(input);
        const patternWords = this.tokenize(pattern);
        
        patternWords.forEach(pWord => {
          inputWords.forEach(iWord => {
            if (this.isSimilar(pWord, iWord)) {
              score += 5;
            }
          });
        });
      }
      
      // Method 2: Keyword similarity (medium weight)
      const patternKeywords = data.patterns.flatMap(p => this.extractKeywords(p));
      const similarity = this.calculateSimilarity(expandedKeywords, patternKeywords);
      score += similarity * 20;
      
      // Method 3: Context bonus - if continuing conversation on same topic
      if (this.conversationContext.lastTopic === key) {
        score += 5;
      }
      
      // Method 4: Question type alignment
      if (this.conversationContext.questionType === 'what' && 
          ['skills', 'projects', 'experience', 'education'].includes(key)) {
        score += 3;
      }
      if (this.conversationContext.questionType === 'where' && 
          ['location', 'background'].includes(key)) {
        score += 5;
      }
      if (this.conversationContext.questionType === 'how' && 
          ['contact', 'projects', 'ai_ml'].includes(key)) {
        score += 3;
      }
      
      if (score > highestScore) {
        highestScore = score;
        bestMatch = { key, data, score };
      }
    }
    
    console.log('🎯 Best match:', bestMatch?.key, 'with score:', bestMatch?.score);
    console.log('📊 Question type:', this.conversationContext.questionType);
    console.log('💭 Sentiment:', this.conversationContext.sentiment);
    console.log('🏷️ Entities:', this.conversationContext.entities);
    
    return highestScore > 5 ? bestMatch : null;
  }

  // Generate context-aware response with sentiment adaptation and guardrails
  generateResponse(userInput) {
    console.log('🔍 generateResponse called with:', userInput);
    
    // STEP 1: Check for inappropriate content FIRST
    const guardrailCheck = this.checkInappropriateContent(userInput);
    if (guardrailCheck.isInappropriate) {
      console.log('⚠️ Guardrail triggered:', guardrailCheck.category);
      return guardrailCheck.response;
    }
    
    // STEP 2: Proceed with normal matching
    const match = this.findBestMatch(userInput);
    
    if (match) {
      this.conversationContext.lastTopic = match.key;
      this.conversationContext.askedTopics.push(match.key);
      
      // Add sentiment-aware response prefix
      let response = match.data.response;
      
      if (this.conversationContext.sentiment === 'positive') {
        const positivePrefix = ["I'm glad you're interested! ", "Great question! ", "Happy to share! "][Math.floor(Math.random() * 3)];
        response = positivePrefix + response;
      }
      
      return response;
    }
    
    // STEP 3: Smart fallback with entity recognition
    if (this.conversationContext.entities.length > 0) {
      return `I noticed you mentioned ${this.conversationContext.entities.join(', ')}. While I don't have specific information about that exact query, I can tell you about Isfar's skills, projects, experience, education, background, hobbies, goals, or achievements. What would you like to know?`;
    }
    
    // Provide helpful default with suggestions
    const suggestedTopics = this.conversationContext.askedTopics.length > 0
      ? "You could also ask about her hobbies, goals, achievements, or specific technologies she works with!"
      : "You can ask me about her skills, projects, experience, education, background, hobbies, or anything else about her professional journey!";
    
    return `I'd love to help you learn about Isfar! ${suggestedTopics}\n\nWhat would you like to know?`;
  }
}

// Initialize the chatbot
console.log('🚀 Initializing chatbot...');
const chatbotAI = new IsfarChatbotAI();

// Message handling functions
// Convert markdown-style text to clean HTML
function formatBotResponse(text) {
  var safe = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  safe = safe.replace(/^[•●]\s*(.+)$/gm, '<li>$1</li>');
  safe = safe.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
  safe = safe.replace(/\n{2,}/g, '</p><p>');
  safe = safe.replace(/\n/g, '<br>');
  safe = '<p>' + safe + '</p>';
  safe = safe.replace(/<p>\s*<\/p>/g, '');
  return safe;
}

function addMessage(content, isBot = false, withTyping = false) {
  console.log('📝 Adding message:', content, 'isBot:', isBot);
  const messagesContainer = document.querySelector('.chat-messages');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${isBot ? 'bot-message' : 'user-message'}`;
  
  const messageContent = document.createElement('div');
  messageContent.className = 'message-content';
  
  if (isBot) {
    // Render bot messages as formatted HTML
    messageContent.innerHTML = formatBotResponse(content);
    messageDiv.appendChild(messageContent);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
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
  
  console.log('✅ Advanced Isfar Chatbot initialized successfully with comprehensive knowledge base!');
});
