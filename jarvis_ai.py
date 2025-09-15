import sys
import os
from dotenv import load_dotenv
load_dotenv()
import json
import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import threading
import time
from pathlib import Path
import logging
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain_groq import ChatGroq

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                           QLabel, QListWidget, QListWidgetItem, QFrame,
                           QScrollArea, QSplitter, QGraphicsDropShadowEffect)
from PyQt5.QtCore import (Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, 
                         QEasingCurve, QRect, QPoint)
from PyQt5.QtGui import (QFont, QPalette, QColor, QPixmap, QPainter, 
                        QLinearGradient, QRadialGradient, QBrush, QPen,
                        QFontDatabase, QIcon)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

word_to_number = {
    "zero": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10
}


class JarvisAI:
    def __init__(self, gemini_api_key: Optional[str] = None, groq_api_key: Optional[str] = None, data_dir: str = "jarvis_data"):
        logger.info("Initializing Jarvis")
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.gemini_api_key or not self.groq_api_key:
            logger.error("GEMINI_API_KEY and GROQ_API_KEY must be set in environment or passed to JarvisAI.")
            sys.exit(1)
        logger.info("API keys loaded successfully")
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        logger.info("Initializing AI models...")
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        self.groq_model = ChatGroq(model_name="llama3-70b-convolution", groq_api_key=self.groq_api_key, temperature=0)
        logger.info("AI models initialized")
        logger.info("Initializing speech systems...")
        self.tts_engine = pyttsx3.init()
        self.setup_voice()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        logger.info("Speech systems ready")
        self.wake_word = "jarvis"
        self.is_active = False
        self.listening_for_wake_word = True        
        logger.info("Loading sentence transformer model (this may take a moment)...")
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        self.conversations_file = self.data_dir / "conversations.xlsx"
        self.tasks_file = self.data_dir / "tasks.xlsx"
        self.vector_db_file = self.data_dir / "vector_db.pkl"
        self.faiss_index_file = self.data_dir / "faiss_index.idx"
        self.conversations = []
        self.tasks = []
        self.vector_db = []
        self.faiss_index = None
        self.load_data()
        self.initialize_vector_db()
        self.user_context = {
            "preferences": {},
            "frequent_topics": {},
            "conversation_history": []
        }
        self.setup_agents()
        self.conversation_chain = []
        
        logger.info("Jarvis AI Assistant initialized successfully!")
    
    def setup_voice(self):
        voices = self.tts_engine.getProperty('voices')
        
        if voices:
            preferred_voices = ['daniel', 'alex', 'david', 'mark', 'male']
            voice_set = False
            for preferred in preferred_voices:
                for voice in voices:
                    voice_name = voice.name.lower()
                    if preferred in voice_name:
                        self.tts_engine.setProperty('voice', voice.id)
                        voice_set = True
                        break
                if voice_set:
                    breakpoint
        self.tts_engine.setProperty('rate', 210)
        self.tts_engine.setProperty('volume', 1.0)
        
    def setup_agents(self):
        self.tools = [
            Tool(
                name="add_task",
                func=self.agent_add_task,
                description="Add a new task. Input should be task description, optionally with due date and priority separated by commas."
            ),
            Tool(
                name="get_tasks",
                func=self.agent_get_tasks,
                description="Get all tasks or tasks with specific status. Input can be 'all', 'pending', 'completed', or 'in_progress'."
            ),
            Tool(
                name="list_tasks",
                func=self.agent_list_tasks,
                description="List all tasks with their details. Input should be empty."
            ),
            Tool(
                name="update_task",
                func=self.agent_update_task,
                description="Update task status. Input should be 'task_id,new_status' where status is 'pending', 'in_progress', or 'completed'."
            ),
            Tool(
                name="search_conversations",
                func=self.agent_search_conversations,
                description="Search previous conversations. Input should be the search query."
            ),
            Tool(
                name="get_user_context",
                func=self.agent_get_user_context,
                description="Get user context including preferences and frequent topics."
            )
        ]
        self.prompt_react = hub.pull("hwchase17/react")
        self.react_agent = create_react_agent(self.groq_model, tools=self.tools, prompt=self.prompt_react)
        self.agent_executor = AgentExecutor(
            agent=self.react_agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=3
        )
    def agent_add_task(self, input_string: str) -> str:
        parts = input_string.split(',')
        task_desc = parts[0].strip()
        due_date = parts[1].strip() if len(parts) > 1 else None
        priority = parts[2].strip() if len(parts) > 2 else "medium"
        return self.add_task(task_desc, due_date, priority)
    
    def agent_get_tasks(self, status: str = "all") -> str:
        if status == "all":
            tasks = self.get_tasks()
        else:
            tasks = self.get_tasks(status.strip())
        
        if not tasks:
            return f"No tasks found with status: {status}"
        task_list = []
        for task in tasks:
            task_list.append(f"ID: {task['id']} - {task['description']} (Status: {task['status']}, Priority: {task['priority']})")
        return "\n".join(task_list)
    
    def agent_list_tasks(self, input_string: str = "") -> str:
        tasks = self.get_tasks()
        if not tasks:
            return "No tasks found."
        task_list = []
        for task in tasks:
            task_list.append(
                f"ID: {task['id']} - {task['description']} (Status: {task['status']}, Due: {task.get('due_date','')}, Priority: {task.get('priority','')})"
            )
        return "\n".join(task_list)
    
    def agent_update_task(self, input_string: str) -> str:
        parts = input_string.split(',')
        if len(parts) != 2:
            return "Invalid input. Please provide task_id,new_status"
        try:
            task_id = int(parts[0].strip())
            new_status = parts[1].strip()
            return self.update_task_status(task_id, new_status)
        except ValueError:
            return "Invalid task ID. Please provide a valid number."
    
    def agent_search_conversations(self, query: str) -> str:
        results = self.search_vector_db(query, k=3)
        if not results:
            return "No relevant conversations found."
        context = "Previous conversations:\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. {result['text'][:200]}...\n"
        
        return context
    
    def agent_get_user_context(self, input_string: str = "") -> str:
        frequent_topics = sorted(self.user_context['frequent_topics'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        context = f"User preferences: {self.user_context['preferences']}\n"
        context += f"Top topics: {[topic[0] for topic in frequent_topics]}\n"
        context += f"Recent conversations: {len(self.user_context['conversation_history'])}"
        
        return context

    def load_data(self):
        try:
            if self.conversations_file.exists():
                df = pd.read_excel(self.conversations_file)
                self.conversations = df.to_dict('records')
            if self.tasks_file.exists():
                df = pd.read_excel(self.tasks_file)
                self.tasks = df.to_dict('records')
            if self.vector_db_file.exists():
                with open(self.vector_db_file, 'rb') as f:
                    self.vector_db = pickle.load(f)
            logger.info("Data loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def save_data(self):
        try:
            if self.conversations:
                df = pd.DataFrame(self.conversations)
                df.to_excel(self.conversations_file, index=False)
            if self.tasks:
                df = pd.DataFrame(self.tasks)
                df.to_excel(self.tasks_file, index=False)
            with open(self.vector_db_file, 'wb') as f:
                pickle.dump(self.vector_db, f)
            
            logger.info("Data saved successfully!")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def initialize_vector_db(self):
        try:
            if self.vector_db and len(self.vector_db) > 0:
                embeddings = np.array([item['embedding'] for item in self.vector_db])
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(dimension)
                self.faiss_index.add(embeddings.astype('float32'))
                faiss.write_index(self.faiss_index, str(self.faiss_index_file))
            else:
                dimension = 384  
                self.faiss_index = faiss.IndexFlatL2(dimension)
            
            logger.info("Vector database initialized!")
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            dimension = 384
            self.faiss_index = faiss.IndexFlatL2(dimension)
    
    def speak(self, text: str):
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
    
    def listen(self, timeout: int = 5) -> Optional[str]:
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            text = self.recognizer.recognize_google(audio)
            return text.lower().strip()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            logger.error(f"Error in speech-to-text: {e}")
            return None
    
    def add_to_vector_db(self, text: str, metadata: Dict[str, Any]):
        try:
            if self.sentence_model is None:
                logger.warning("Sentence model not available, skipping vector database update")
                return
                
            embedding = self.sentence_model.encode(text)
            vector_item = {
                'text': text,
                'embedding': embedding,
                'metadata': metadata,
                'timestamp': datetime.datetime.now().isoformat()
            }
            self.vector_db.append(vector_item)
            if self.faiss_index is not None:
                self.faiss_index.add(np.array([embedding]).astype('float32'))
            logger.info("Added to vector database")
        except Exception as e:
            logger.error(f"Error adding to vector database: {e}")
    
    def search_vector_db(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.vector_db or self.faiss_index is None or self.sentence_model is None:
                return []
                
            query_embedding = self.sentence_model.encode(query)
            distances, indices = self.faiss_index.search(
                np.array([query_embedding]).astype('float32'), k
            )
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.vector_db):
                    result = self.vector_db[idx].copy()
                    result['similarity_score'] = 1 / (1 + distances[0][i])
                    results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def should_use_agent(self, user_input: str) -> bool:
        agent_keywords = [
            'task', 'add', 'complete', 'finish', 'what did', 'previous', 
            'before', 'earlier', 'context', 'search', 'find', 'look for', 'remember'
        ]
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in agent_keywords)

    def process_with_agent(self, user_input: str) -> str:
        try:
            result = self.agent_executor.invoke({"input": user_input})
            return result.get("output", "I encountered an error processing your request.")
        except Exception as e:
            logger.error(f"Error in agent processing: {e}")
            return "I encountered an error while processing your request. Let me try a different approach."
    def generate_response_with_context(self, user_input: str) -> str:
        try:
            chain_context = ""
            if self.conversation_chain:
                chain_context = "Recent conversation:\n"
                for turn in self.conversation_chain[-5:]:
                    chain_context += f"User: {turn['user']}\nAI: {turn['ai']}\n"
            similar_conversations = self.search_vector_db(user_input, k=3)    
            context = ""
            if similar_conversations:
                context = "Previous relevant conversations:\n"
                for conv in similar_conversations:
                    context += f"- {conv['text'][:150]}...\n"
            if self.user_context['preferences']:
                context += f"\nUser preferences: {self.user_context['preferences']}\n"
            prompt = f"""
            You are Jarvis, a helpful AI assistant. Use the following context to provide personalized responses.

            {chain_context}
            {context}

            Current user input: {user_input}

            Provide a helpful, conversational response. Keep it concise but informative.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request."
    
    def process_user_input(self, user_input: str) -> str:
        if self.should_use_agent(user_input):
            response = self.process_with_agent(user_input)
        else:
            response = self.generate_response_with_context(user_input)
        self._update_conversation_chain(user_input, response)
        self.save_conversation(user_input, response)
        return response
    
    def _update_conversation_chain(self, user_input: str, ai_response: str):
        """Maintain a short conversation chain for follow-up context"""
        self.conversation_chain.append({'user': user_input, 'ai': ai_response})
        if len(self.conversation_chain) > 5:
            self.conversation_chain = self.conversation_chain[-8:]

    def save_conversation(self, user_input: str, ai_response: str):
        """Save conversation to Excel and vector database"""
        conversation = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'ai_response': ai_response,
            'conversation_id': len(self.conversations) + 1
        }
        
        self.conversations.append(conversation)
        
        full_text = f"User: {user_input} | AI: {ai_response}"
        metadata = {
            'type': 'conversation',
            'timestamp': conversation['timestamp'],
            'conversation_id': conversation['conversation_id']
        }
        self.add_to_vector_db(full_text, metadata)
        self.update_user_context(user_input)
        self.save_data()
    
    def update_user_context(self, user_input: str):
        """Update user context based on conversation"""
        # Add to conversation history
        self.user_context['conversation_history'].append({
            'input': user_input,
            'timestamp': datetime.datetime.now().isoformat()
        })
        if len(self.user_context['conversation_history']) > 70:
            self.user_context['conversation_history'] = self.user_context['conversation_history'][-70:]
        
        words = user_input.lower().split()
        for word in words:
            if len(word) > 3:
                self.user_context['frequent_topics'][word] = self.user_context['frequent_topics'].get(word, 0) + 1
    
    def add_task(self, task_description: str, due_date: str = None, priority: str = "medium"):
        """Add a new task with id, status, etc."""
        next_id = max([t['id'] for t in self.tasks], default=0) + 1
        task = {
            'id': next_id,
            'description': task_description,
            'due_date': due_date if due_date else "",
            'priority': priority,
            'status': 'pending',
            'created_at': datetime.datetime.now().isoformat()
        }
        self.tasks.append(task)
        self.save_data()
        return f"Task added: {task_description}{', ' + due_date if due_date else ''}"

    def get_tasks(self, status: str = None):
        if not isinstance(self.tasks, list):
            self.tasks = []
        if status:
            return [task for task in self.tasks if task.get('status') == status]
        return list(self.tasks)

    def update_task_status(self, task_id: int, new_status: str):
        for task in self.tasks:
            if task['id'] == task_id:
                task['status'] = new_status
                task['updated_at'] = datetime.datetime.now().isoformat()
                self.save_data()
                return f"Task {task_id} status updated to {new_status}"
        return f"Task {task_id} not found"


class VoiceThread(QThread):
    voice_result = pyqtSignal(str)
    voice_status = pyqtSignal(str)
    
    def __init__(self, jarvis_ai):
        super().__init__()
        self.jarvis_ai = jarvis_ai
        self.is_listening = False
        
    def run(self):
        self.voice_status.emit("Listening...")
        try:
            result = self.jarvis_ai.listen(timeout=10)
            if result:
                self.voice_result.emit(result)
            else:
                self.voice_status.emit("No speech detected")
        except Exception as e:
            self.voice_status.emit(f"Voice error: {str(e)}")
        
    def start_listening(self):
        if not self.isRunning():
            self.start()


class ResponseThread(QThread):
    response_ready = pyqtSignal(str)
    
    def __init__(self, jarvis_ai, user_input):
        super().__init__()
        self.jarvis_ai = jarvis_ai
        self.user_input = user_input
        
    def run(self):
        try:
            response = self.jarvis_ai.process_user_input(self.user_input)
            self.response_ready.emit(response)
        except Exception as e:
            self.response_ready.emit(f"Error: {str(e)}")


class SpeechThread(QThread):
    speech_finished = pyqtSignal()
    
    def __init__(self, jarvis_ai, text):
        super().__init__()
        self.jarvis_ai = jarvis_ai
        self.text = text
        
    def run(self):
        try:
            self.jarvis_ai.speak(self.text)
            self.speech_finished.emit()
        except Exception as e:
            logger.error(f"Speech error: {e}")
            self.speech_finished.emit()


class TransparentWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QRadialGradient(self.width()//2, self.height()//2, min(self.width(), self.height())//2)
        gradient.setColorAt(0, QColor(0, 245, 255, 20))
        gradient.setColorAt(1, QColor(0, 128, 255, 10))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(0, 245, 255, 80), 2))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 20, 20)


class AnimatedButton(QPushButton):    
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 245, 255, 180),
                    stop:1 rgba(0, 128, 255, 180));
                border: 2px solid rgba(0, 245, 255, 100);
                border-radius: 15px;
                padding: 10px 20px;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 245, 255, 220),
                    stop:1 rgba(0, 128, 255, 220));
                border: 2px solid rgba(0, 245, 255, 150);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 128, 255, 200),
                    stop:1 rgba(0, 80, 200, 200));
            }
        """)


class JarvisGUI(QMainWindow):
    def __init__(self, jarvis_instance=None):
        super().__init__()
        if jarvis_instance:
            self.jarvis = jarvis_instance
            logger.info("Using existing JARVIS instance")
        else:
            self.init_jarvis()
        self.init_ui()
        self.voice_thread = None
        self.response_thread = None
        self.speech_thread = None   
        self.is_listening = False
        self.is_speaking = False
        
    def init_jarvis(self):
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        
        if not GEMINI_API_KEY or not GROQ_API_KEY:
            logger.error("GEMINI_API_KEY and GROQ_API_KEY environment variables must be set.")
            sys.exit(1)
        try:
            self.jarvis = JarvisAI(GEMINI_API_KEY, GROQ_API_KEY)
            logger.info("JARVIS AI initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize JARVIS: {e}")
            sys.exit(1)
            
    def init_ui(self):
        self.setWindowTitle("JARVIS Neural Interface")
        self.setGeometry(100, 100, 1400, 900)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        central_widget = TransparentWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.create_chat_panel(main_layout)
        self.create_arc_reactor_panel(main_layout)
        self.create_tasks_panel(main_layout)
        self.apply_dark_theme()
        self.create_status_bar()
        self.setup_timers()
    def create_chat_panel(self, parent_layout):
        chat_frame = QFrame()
        chat_frame.setFixedWidth(420)
        chat_frame.setStyleSheet("""
            QFrame {
                background: rgba(15, 15, 35, 150);
                border: 2px solid rgba(0, 245, 255, 80);
                border-radius: 20px;
            }
        """)
        
        chat_layout = QVBoxLayout(chat_frame)
        chat_layout.setSpacing(15)
        chat_layout.setContentsMargins(20, 20, 20, 20)
        header_label = QLabel("NEURAL INTERFACE")
        header_label.setStyleSheet("""
            QLabel {
                color: rgb(0, 245, 255);
                font-family: 'Orbitron', monospace;
                font-weight: bold;
                font-size: 16px;
                text-align: center;
                padding: 10px;
                border-bottom: 1px solid rgba(0, 245, 255, 80);
                margin-bottom: 10px;
            }
        """)
        header_label.setAlignment(Qt.AlignCenter)
        chat_layout.addWidget(header_label)
        self.chat_messages = QTextEdit()
        self.chat_messages.setReadOnly(True)
        self.chat_messages.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 0, 0, 100);
                border: 1px solid rgba(0, 245, 255, 50);
                border-radius: 15px;
                padding: 15px;
                color: white;
                font-size: 14px;
                line-height: 1.5;
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 100);
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 245, 255, 150);
                border-radius: 5px;
            }
        """)
        chat_layout.addWidget(self.chat_messages)
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Enter your command...")
        self.message_input.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 245, 255, 30);
                border: 2px solid rgba(0, 245, 255, 80);
                border-radius: 15px;
                padding: 15px 20px;
                color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(0, 245, 255, 150);
                background: rgba(0, 245, 255, 50);
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        self.send_btn = AnimatedButton("SEND")
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)
        self.voice_btn = AnimatedButton("🎤")
        self.voice_btn.clicked.connect(self.toggle_voice)
        self.voice_btn.setFixedWidth(60)
        input_layout.addWidget(self.voice_btn)
        
        chat_layout.addLayout(input_layout)
        parent_layout.addWidget(chat_frame)
        self.add_message("Welcome to JARVIS Neural Interface. I'm your advanced AI assistant, ready to help with tasks and intelligent conversation. How may I assist you today?", "ai")
    
    def create_arc_reactor_panel(self, parent_layout):
        """Create the central arc reactor animation"""
        reactor_frame = QFrame()
        reactor_frame.setFixedSize(300, 300)
        reactor_frame.setStyleSheet("""
            QFrame {
                background: transparent;
            }
        """)
        self.arc_reactor = ArcReactorWidget()
        reactor_layout = QVBoxLayout(reactor_frame)
        reactor_layout.addWidget(self.arc_reactor)
        
        parent_layout.addWidget(reactor_frame)
    
    def create_tasks_panel(self, parent_layout):
        """Create the tasks panel"""
        tasks_frame = QFrame()
        tasks_frame.setFixedWidth(420)
        tasks_frame.setStyleSheet("""
            QFrame {
                background: rgba(15, 15, 35, 150);
                border: 2px solid rgba(0, 245, 255, 80);
                border-radius: 20px;
            }
        """)
        
        tasks_layout = QVBoxLayout(tasks_frame)
        tasks_layout.setSpacing(15)
        tasks_layout.setContentsMargins(20, 20, 20, 20)
        tasks_header = QLabel("MISSION CONTROL")
        tasks_header.setStyleSheet("""
            QLabel {
                color: rgb(0, 245, 255);
                font-family: 'Orbitron', monospace;
                font-weight: bold;
                font-size: 16px;
                text-align: center;
                padding: 10px;
                border-bottom: 1px solid rgba(0, 245, 255, 80);
                margin-bottom: 10px;
            }
        """)
        tasks_header.setAlignment(Qt.AlignCenter)
        tasks_layout.addWidget(tasks_header)
        self.tasks_list = QListWidget()
        self.tasks_list.setStyleSheet("""
            QListWidget {
                background: rgba(0, 0, 0, 100);
                border: 1px solid rgba(0, 245, 255, 50);
                border-radius: 15px;
                padding: 10px;
                color: white;
                font-size: 14px;
            }
            QListWidget::item {
                background: rgba(0, 128, 255, 30);
                border: 1px solid rgba(0, 245, 255, 60);
                border-radius: 10px;
                padding: 12px;
                margin: 5px 0;
            }
            QListWidget::item:hover {
                background: rgba(0, 128, 255, 60);
                border: 1px solid rgba(0, 245, 255, 100);
            }
            QListWidget::item:selected {
                background: rgba(0, 245, 255, 80);
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 100);
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 245, 255, 150);
                border-radius: 5px;
            }
        """)
        self.tasks_list.itemDoubleClicked.connect(self.toggle_task)
        tasks_layout.addWidget(self.tasks_list)
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("Add new mission...")
        self.task_input.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 245, 255, 30);
                border: 2px solid rgba(0, 245, 255, 80);
                border-radius: 15px;
                padding: 15px 20px;
                color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid rgba(0, 245, 255, 150);
                background: rgba(0, 245, 255, 50);
            }
        """)
        self.task_input.returnPressed.connect(self.add_task)
        tasks_layout.addWidget(self.task_input)
        
        # Add task button
        add_task_btn = AnimatedButton("ADD MISSION")
        add_task_btn.clicked.connect(self.add_task)
        tasks_layout.addWidget(add_task_btn)
        
        parent_layout.addWidget(tasks_frame)
        self.load_tasks()
    
    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background: transparent;
                color: white;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QWidget {
                font-family: 'Inter', sans-serif;
            }
        """)
    
    def create_status_bar(self):
        self.status_label = QLabel("JARVIS ONLINE")
        self.status_label.setStyleSheet("""
            QLabel {
                color: rgb(0, 245, 255);
                font-family: 'Orbitron', monospace;
                font-weight: bold;
                font-size: 12px;
                padding: 5px 15px;
                background: rgba(0, 0, 0, 100);
                border: 1px solid rgba(0, 245, 255, 80);
                border-radius: 10px;
            }
        """)
        self.status_label.setParent(self)
        self.status_label.move(self.width()//2 - 75, self.height() - 50)
    
    def setup_timers(self):
        self.reactor_timer = QTimer()
        self.reactor_timer.timeout.connect(self.update_arc_reactor)
        self.reactor_timer.start(50)
        
    def update_arc_reactor(self):
        if hasattr(self, 'arc_reactor'):
            self.arc_reactor.update()
    
    def add_message(self, message, sender):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if sender == "user":
            formatted_message = f"<div style='color: #00f5ff; margin: 10px 0;'><b>[{timestamp}] USER:</b></div>"
            formatted_message += f"<div style='background: rgba(0, 245, 255, 20); padding: 10px; border-radius: 10px; margin: 5px 0 15px 20px; border-left: 3px solid #00f5ff;'>{message}</div>"
        else:
            formatted_message = f"<div style='color: #0080ff; margin: 10px 0;'><b>[{timestamp}] JARVIS:</b></div>"
            formatted_message += f"<div style='background: rgba(0, 128, 255, 20); padding: 10px; border-radius: 10px; margin: 5px 0 15px 20px; border-left: 3px solid #0080ff;'>{message}</div>"
        
        self.chat_messages.append(formatted_message)
        scrollbar = self.chat_messages.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def send_message(self):
        message = self.message_input.text().strip()
        if not message:
            return
        self.add_message(message, "user")
        self.message_input.clear()
        self.status_label.setText("JARVIS PROCESSING...")
        
        # Process message in background thread
        self.response_thread = ResponseThread(self.jarvis, message)
        self.response_thread.response_ready.connect(self.handle_response)
        self.response_thread.start()
    
    def handle_response(self, response):
        self.add_message(response, "ai")
        self.status_label.setText("JARVIS READY")
        if response:
            self.speak_text(response)
        if any(word in response.lower() for word in ['task', 'mission', 'added', 'completed']):
            self.load_tasks()
    
    def toggle_voice(self):
        if not self.is_listening:
            self.start_voice_recognition()
        else:
            self.stop_voice_recognition()
    
    def start_voice_recognition(self):
        if self.voice_thread and self.voice_thread.isRunning():
            return
        
        self.is_listening = True
        self.voice_btn.setText("OFF")
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 68, 68, 180),
                    stop:1 rgba(204, 0, 0, 180));
                border: 2px solid rgba(255, 68, 68, 100);
                border-radius: 15px;
                padding: 10px 20px;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        
        self.status_label.setText("JARVIS LISTENING...")
        self.voice_thread = VoiceThread(self.jarvis)
        self.voice_thread.voice_result.connect(self.handle_voice_result)
        self.voice_thread.voice_status.connect(self.handle_voice_status)
        self.voice_thread.finished.connect(self.voice_recognition_finished)
        self.voice_thread.start()
    
    def stop_voice_recognition(self):
        self.is_listening = False
        self.voice_btn.setText("ON")
        self.voice_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 245, 255, 180),
                    stop:1 rgba(0, 128, 255, 180));
                border: 2px solid rgba(0, 245, 255, 100);
                border-radius: 15px;
                padding: 10px 20px;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        
        if self.voice_thread:
            self.voice_thread.quit()
    
    def handle_voice_result(self, text):
        self.message_input.setText(text)
        self.send_message()
    
    def handle_voice_status(self, status):
        self.status_label.setText(f"JARVIS - {status}")
    def voice_recognition_finished(self):
        self.stop_voice_recognition()
        self.status_label.setText("JARVIS READY")
    def speak_text(self, text):
        if self.is_speaking:
            return
        
        self.is_speaking = True
        self.status_label.setText("JARVIS SPEAKING...")
        
        self.speech_thread = SpeechThread(self.jarvis, text)
        self.speech_thread.speech_finished.connect(self.speech_finished)
        self.speech_thread.start()
    
    def speech_finished(self):
        self.is_speaking = False
        self.status_label.setText("JARVIS READY")
    
    def add_task(self):
        """Add a new task"""
        task_text = self.task_input.text().strip()
        if not task_text:
            return
        result = self.jarvis.add_task(task_text)
        self.add_message(f"Added task: {task_text}", "ai")
        self.task_input.clear()
        self.load_tasks()
        self.speak_text(f"Task added: {task_text}")
    
    def load_tasks(self):
        self.tasks_list.clear()
        
        tasks = self.jarvis.get_tasks()
        for task in tasks:
            item_text = f"{task['description']}"
            if task.get('due_date'):
                item_text += f" - Due: {task['due_date']}"
            
            item = QListWidgetItem(item_text)
            if task['status'] == 'completed':
                item.setStyleSheet("""
                    QListWidgetItem {
                        background: rgba(0, 255, 136, 30);
                        border: 1px solid rgba(0, 255, 136, 60);
                        text-decoration: line-through;
                        color: rgba(255, 255, 255, 150);
                    }
                """)
            elif task['priority'] == 'high':
                item.setStyleSheet("""
                    QListWidgetItem {
                        background: rgba(255, 170, 0, 30);
                        border: 1px solid rgba(255, 170, 0, 60);
                    }
                """)
            
            item.setData(Qt.UserRole, task['id'])
            self.tasks_list.addItem(item)
    
    def toggle_task(self, item):
        task_id = item.data(Qt.UserRole)
        for task in self.jarvis.tasks:
            if task['id'] == task_id:
                new_status = 'completed' if task['status'] == 'pending' else 'pending'
                result = self.jarvis.update_task_status(task_id, new_status)
                self.load_tasks()
                self.speak_text(f"Task {new_status}")
                break
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.globalPos()
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drag_start_position:
            self.move(self.pos() + event.globalPos() - self.drag_start_position)
            self.drag_start_position = event.globalPos()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)
    
    def closeEvent(self, event):
        if hasattr(self, 'jarvis'):
            self.jarvis.save_data()
        if self.voice_thread and self.voice_thread.isRunning():
            self.voice_thread.quit()
            self.voice_thread.wait()
        
        if self.response_thread and self.response_thread.isRunning():
            self.response_thread.quit()
            self.response_thread.wait()
        
        if self.speech_thread and self.speech_thread.isRunning():
            self.speech_thread.quit()
            self.speech_thread.wait()
        
        event.accept()


class ArcReactorWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setFixedSize(300, 300)
        self.rotation = 0
        self.pulse_scale = 1.0
        self.pulse_direction = 1
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center_x, center_y = self.width() // 2, self.height() // 2
        outer_gradient = QRadialGradient(center_x, center_y, 140)
        outer_gradient.setColorAt(0, QColor(0, 245, 255, 30))
        outer_gradient.setColorAt(0.7, QColor(0, 128, 255, 20))
        outer_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setBrush(QBrush(outer_gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_x - 140, center_y - 140, 280, 280)
        main_gradient = QRadialGradient(center_x, center_y, 110)
        main_gradient.setColorAt(0, QColor(255, 255, 255, 200))
        main_gradient.setColorAt(0.2, QColor(0, 245, 255, 180))
        main_gradient.setColorAt(0.5, QColor(0, 128, 255, 120))
        main_gradient.setColorAt(0.8, QColor(0, 26, 77, 100))
        main_gradient.setColorAt(1, QColor(10, 10, 15, 200))
        painter.setBrush(QBrush(main_gradient))
        painter.setPen(QPen(QColor(0, 245, 255, 150), 3))
        reactor_size = int(220 * self.pulse_scale)
        painter.drawEllipse(center_x - reactor_size//2, center_y - reactor_size//2, reactor_size, reactor_size)
        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(self.rotation)
        painter.setPen(QPen(QColor(0, 245, 255, 180), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(-90, -90, 180, 180)
        painter.rotate(-self.rotation * 1.5)
        painter.setPen(QPen(QColor(0, 245, 255, 120), 1))
        painter.drawEllipse(-70, -70, 140, 140)
        painter.restore()
        core_gradient = QRadialGradient(center_x, center_y, 35)
        core_gradient.setColorAt(0, QColor(255, 255, 255, 255))
        core_gradient.setColorAt(0.3, QColor(0, 245, 255, 200))
        core_gradient.setColorAt(1, QColor(0, 128, 255, 150))
        painter.setBrush(QBrush(core_gradient))
        painter.setPen(QPen(QColor(0, 245, 255, 200), 2))
        core_size = int(70 * self.pulse_scale)
        painter.drawEllipse(center_x - core_size//2, center_y - core_size//2, core_size, core_size)
        self.rotation += 2
        if self.rotation >= 360:
            self.rotation = 0
        self.pulse_scale += 0.005 * self.pulse_direction
        if self.pulse_scale >= 1.1:
            self.pulse_direction = -1
        elif self.pulse_scale <= 0.9:
            self.pulse_direction = 1

def main():
    try:
        logger.info("Starting JARVIS Neural Interface...")
        logger.info("Testing JarvisAI initialization...")
        jarvis = JarvisAI()
        logger.info("JarvisAI initialized successfully!")
        app = QApplication(sys.argv)
        app.setApplicationName("JARVIS Neural Interface")
        app.setApplicationVersion("2.3")
        logger.info("Starting GUI...")
        jarvis_gui = JarvisGUI(jarvis)
        jarvis_gui.show()
        logger.info("JARVIS Neural Interface ready!")
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Failed to start JARVIS: {e}")
        logger.error(f"Traceback: {e.__traceback__}")
        sys.exit(1)
if __name__ == "__main__":
    main()