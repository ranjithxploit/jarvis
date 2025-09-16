import sys
import os
from dotenv import load_dotenv
load_dotenv()
import json
from datetime import datetime, timedelta
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
import subprocess
import platform
import requests
import json
from datetime import datetime, timedelta

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

# Check for enhanced audio packages
try:
    import sounddevice as sd
    import speexdsp
    import webrtcvad
    AUDIO_ENHANCED = True
    logger.info("Enhanced audio packages available")
except ImportError:
    AUDIO_ENHANCED = False
    logger.warning("Enhanced audio packages not available. Install: pip install sounddevice speexdsp webrtcvad")

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
        
        # Initialize timers and system control
        self.active_timers = {}
        self.stopwatches = {}
        
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
            ),
            Tool(
                name="start_timer",
                func=self.agent_start_timer,
                description="Start a timer. Input should be 'duration_in_minutes,timer_name' or just 'duration_in_minutes'."
            ),
            Tool(
                name="start_stopwatch",
                func=self.agent_start_stopwatch,
                description="Start a stopwatch. Input should be stopwatch name."
            ),
            Tool(
                name="stop_timer",
                func=self.agent_stop_timer,
                description="Stop a timer. Input should be timer name or 'all' to stop all timers."
            ),
            Tool(
                name="get_system_info",
                func=self.agent_get_system_info,
                description="Get current date, time, and system information."
            ),
            Tool(
                name="control_volume",
                func=self.agent_control_volume,
                description="Control system volume. Input should be 'up', 'down', 'mute', or a number 0-100."
            ),
            Tool(
                name="daily_summary",
                func=self.agent_daily_summary,
                description="Get daily summary with date, time, weather, and tasks."
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
    
    def agent_start_timer(self, input_string: str) -> str:
        """Start a timer with specified duration"""
        try:
            parts = input_string.split(',')
            duration = float(parts[0].strip())
            timer_name = parts[1].strip() if len(parts) > 1 else f"Timer_{len(self.active_timers)+1}"
            
            if timer_name in self.active_timers:
                return f"Timer '{timer_name}' is already running"
            
            # Start timer in separate thread
            timer_thread = threading.Timer(duration * 60, self._timer_finished, [timer_name, duration])
            timer_thread.start()
            
            self.active_timers[timer_name] = {
                'duration': duration,
                'start_time': datetime.now(),
                'thread': timer_thread
            }
            
            return f"Timer '{timer_name}' started for {duration} minutes"
        except Exception as e:
            return f"Error starting timer: {str(e)}"
    
    def agent_start_stopwatch(self, input_string: str) -> str:
        """Start a stopwatch"""
        try:
            stopwatch_name = input_string.strip() or f"Stopwatch_{len(self.stopwatches)+1}"
            
            if stopwatch_name in self.stopwatches:
                return f"Stopwatch '{stopwatch_name}' is already running"
            
            self.stopwatches[stopwatch_name] = {
                'start_time': datetime.now(),
                'is_running': True
            }
            
            return f"Stopwatch '{stopwatch_name}' started"
        except Exception as e:
            return f"Error starting stopwatch: {str(e)}"
    
    def agent_stop_timer(self, input_string: str) -> str:
        """Stop a timer or stopwatch"""
        try:
            name = input_string.strip()
            
            if name == "all":
                for timer_name, timer_info in self.active_timers.items():
                    timer_info['thread'].cancel()
                self.active_timers.clear()
                self.stopwatches.clear()
                return "All timers and stopwatches stopped"
            
            if name in self.active_timers:
                self.active_timers[name]['thread'].cancel()
                del self.active_timers[name]
                return f"Timer '{name}' stopped"
            
            if name in self.stopwatches:
                elapsed = datetime.now() - self.stopwatches[name]['start_time']
                del self.stopwatches[name]
                return f"Stopwatch '{name}' stopped. Elapsed time: {str(elapsed)}"
            
            return f"No timer or stopwatch named '{name}' found"
        except Exception as e:
            return f"Error stopping timer: {str(e)}"
    
    def agent_get_system_info(self, input_string: str = "") -> str:
        """Get current system information"""
        try:
            now = datetime.now()
            info = f"Current date: {now.strftime('%Y-%m-%d')}\n"
            info += f"Current time: {now.strftime('%H:%M:%S')}\n"
            info += f"Day of week: {now.strftime('%A')}\n"
            info += f"System: {platform.system()} {platform.release()}\n"
            
            # Add active timers info
            if self.active_timers:
                info += f"Active timers: {len(self.active_timers)}\n"
            if self.stopwatches:
                info += f"Active stopwatches: {len(self.stopwatches)}\n"
            
            return info
        except Exception as e:
            return f"Error getting system info: {str(e)}"
    
    def agent_control_volume(self, input_string: str) -> str:
        """Enhanced volume control with more options"""
        try:
            command = input_string.strip().lower()
            
            if platform.system() == "Windows":
                if command in ["up", "increase", "raise", "louder"]:
                    subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]175)"], check=True)
                    return "Volume increased"
                elif command in ["down", "decrease", "lower", "quieter"]:
                    subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]174)"], check=True)
                    return "Volume decreased"
                elif command in ["mute", "silence", "quiet"]:
                    subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]173)"], check=True)
                    return "Volume muted/unmuted"
                elif command.replace("%", "").replace("percent", "").replace(" ", "").isdigit():
                    # Extract number for specific volume level
                    volume = int(''.join(filter(str.isdigit, command)))
                    volume = max(0, min(100, volume))
                    # Use NirCmd for specific volume setting if available
                    try:
                        subprocess.run(["nircmd.exe", "setsysvolume", str(int(volume * 655.35))], check=True)
                        return f"Volume set to {volume}%"
                    except:
                        # Fallback to multiple key presses
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]173)"], check=True)  # Mute first
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]173)"], check=True)  # Unmute
                        for _ in range(volume // 2):  # Approximate volume adjustment
                            subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]175)"], check=True)
                        return f"Volume adjusted to approximately {volume}%"
                elif "maximum" in command or "max" in command or "full" in command:
                    for _ in range(50):  # Max out volume
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]175)"], check=True)
                    return "Volume set to maximum"
                elif "minimum" in command or "min" in command or "lowest" in command:
                    for _ in range(50):  # Min volume
                        subprocess.run(["powershell", "-c", "(New-Object -comObject WScript.Shell).SendKeys([char]174)"], check=True)
                    return "Volume set to minimum"
                else:
                    return f"Unknown volume command: {command}. Try 'up', 'down', 'mute', or a percentage like '50%'"
            
            return "Volume control not supported on this system"
        except Exception as e:
            return f"Error controlling volume: {str(e)}"
    
    def agent_daily_summary(self, input_string: str = "") -> str:
        """Get daily summary with date, time, weather, and tasks"""
        try:
            now = datetime.now()
            summary = f"◊ DAILY BRIEFING ◊\n\n"
            summary += f"Date: {now.strftime('%A, %B %d, %Y')}\n"
            summary += f"Time: {now.strftime('%H:%M:%S')}\n\n"
            
            # Tasks summary
            pending_tasks = [t for t in self.tasks if t.get('status') == 'pending']
            completed_tasks = [t for t in self.tasks if t.get('status') == 'completed']
            
            summary += f"MISSION STATUS:\n"
            summary += f"• Pending missions: {len(pending_tasks)}\n"
            summary += f"• Completed missions: {len(completed_tasks)}\n\n"
            
            if pending_tasks:
                summary += "PRIORITY MISSIONS:\n"
                for task in pending_tasks[:3]:  # Show top 3
                    summary += f"• {task['description']}\n"
            
            # Active timers/stopwatches
            if self.active_timers or self.stopwatches:
                summary += f"\nACTIVE OPERATIONS:\n"
                for name, info in self.active_timers.items():
                    elapsed = datetime.now() - info['start_time']
                    remaining = timedelta(minutes=info['duration']) - elapsed
                    summary += f"• Timer '{name}': {str(remaining).split('.')[0]} remaining\n"
                
                for name, info in self.stopwatches.items():
                    elapsed = datetime.now() - info['start_time']
                    summary += f"• Stopwatch '{name}': {str(elapsed).split('.')[0]} elapsed\n"
            
            return summary
        except Exception as e:
            return f"Error generating daily summary: {str(e)}"
    
    def _timer_finished(self, timer_name: str, duration: float):
        """Called when a timer finishes"""
        if timer_name in self.active_timers:
            del self.active_timers[timer_name]
        
        # This will be called by GUI to speak the notification
        notification = f"Timer '{timer_name}' for {duration} minutes has finished!"
        logger.info(f"Timer finished: {notification}")
        
        # Store notification for GUI to pick up
        if not hasattr(self, 'timer_notifications'):
            self.timer_notifications = []
        self.timer_notifications.append(notification)

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
                'timestamp': datetime.now().isoformat()
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
            'before', 'earlier', 'context', 'search', 'find', 'look for', 'remember',
            'timer', 'stopwatch', 'start', 'stop', 'time', 'minute', 'hour',
            'volume', 'sound', 'audio', 'loud', 'quiet', 'mute', 'increase', 'decrease',
            'up', 'down', 'percent', '%', 'maximum', 'minimum', 'daily', 'summary',
            'briefing', 'date', 'system', 'info'
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
            'timestamp': datetime.now().isoformat(),
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
            'timestamp': datetime.now().isoformat()
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
            'created_at': datetime.now().isoformat()
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
                task['updated_at'] = datetime.now().isoformat()
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
    """Custom transparent widget with Iron Man holographic effect"""
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.animation_frame = 0
        
    def paintEvent(self, event):
        """Paint Iron Man-style holographic background with blue/cyan theme"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Dark Iron Man base with gradient
        painter.fillRect(self.rect(), QColor(5, 10, 25, 200))
        
        # Iron Man holographic grid lines (blue/cyan)
        painter.setPen(QPen(QColor(0, 162, 255, 60), 1))
        grid_size = 60
        for x in range(0, self.width(), grid_size):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), grid_size):
            painter.drawLine(0, y, self.width(), y)
        
        # Animated scan lines (Iron Man blue)
        scan_y = (self.animation_frame * 2) % self.height()
        painter.setPen(QPen(QColor(0, 245, 255, 120), 3))
        painter.drawLine(0, scan_y, self.width(), scan_y)
        painter.drawLine(0, (scan_y + 300) % self.height(), self.width(), (scan_y + 300) % self.height())
        
        # Iron Man corner brackets (blue/cyan theme)
        bracket_size = 40
        painter.setPen(QPen(QColor(0, 245, 255, 200), 4))
        # Top-left
        painter.drawLine(15, 15, 15 + bracket_size, 15)
        painter.drawLine(15, 15, 15, 15 + bracket_size)
        # Top-right
        painter.drawLine(self.width() - 15 - bracket_size, 15, self.width() - 15, 15)
        painter.drawLine(self.width() - 15, 15, self.width() - 15, 15 + bracket_size)
        # Bottom-left
        painter.drawLine(15, self.height() - 15, 15 + bracket_size, self.height() - 15)
        painter.drawLine(15, self.height() - 15 - bracket_size, 15, self.height() - 15)
        # Bottom-right
        painter.drawLine(self.width() - 15 - bracket_size, self.height() - 15, self.width() - 15, self.height() - 15)
        painter.drawLine(self.width() - 15, self.height() - 15 - bracket_size, self.width() - 15, self.height() - 15)
        
        # Additional Iron Man-style decorative elements
        painter.setPen(QPen(QColor(0, 162, 255, 100), 2))
        painter.drawLine(self.width()//4, 5, 3*self.width()//4, 5)
        painter.drawLine(self.width()//4, self.height()-5, 3*self.width()//4, self.height()-5)
        
        self.animation_frame += 1


class AnimatedButton(QPushButton):
    """Iron Man-style holographic button with advanced animations"""
    
    def __init__(self, text):
        super().__init__(text)
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)
        self.glow_intensity = 0
        self.glow_direction = 1
        
        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 20, 40, 180),
                    stop:0.5 rgba(0, 100, 200, 120),
                    stop:1 rgba(0, 245, 255, 180));
                border: 2px solid rgba(0, 245, 255, 150);
                border-radius: 8px;
                padding: 12px 24px;
                color: rgba(255, 255, 255, 220);
                font-weight: bold;
                font-size: 14px;
                font-family: 'Orbitron', 'Consolas', monospace;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 40, 80, 200),
                    stop:0.5 rgba(0, 150, 255, 160),
                    stop:1 rgba(0, 245, 255, 220));
                border: 2px solid rgba(0, 245, 255, 200);
                color: white;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 245, 255, 255),
                    stop:1 rgba(0, 150, 255, 180));
                border: 3px solid rgba(255, 255, 255, 150);
            }
        """)
    
    def update_animation(self):
        """Update button glow animation"""
        self.glow_intensity += self.glow_direction * 5
        if self.glow_intensity >= 100:
            self.glow_direction = -1
        elif self.glow_intensity <= 0:
            self.glow_direction = 1
        self.update()


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
        self.setGeometry(100, 100, 1600, 1000)  # Increased size for new panels
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        central_widget = TransparentWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Create panels layout
        left_panel = QVBoxLayout()
        center_panel = QVBoxLayout()
        right_panel = QVBoxLayout()
        
        self.create_chat_panel(left_panel)
        self.create_system_info_panel(left_panel)
        
        self.create_arc_reactor_panel(center_panel)
        
        self.create_tasks_panel(right_panel)
        self.create_timer_panel(right_panel)
        
        main_layout.addLayout(left_panel)
        main_layout.addLayout(center_panel)
        main_layout.addLayout(right_panel)
        self.apply_dark_theme()
        self.create_status_bar()
        self.setup_timers()
    
    def create_jarvis_button(self, text):
        """Create Iron Man-style button with blue/cyan theme"""
        button = QPushButton(text)
        button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 20, 40, 180),
                    stop:0.5 rgba(0, 100, 200, 120),
                    stop:1 rgba(0, 245, 255, 180));
                border: 3px solid rgba(0, 245, 255, 180);
                border-radius: 12px;
                padding: 15px 30px;
                color: rgba(255, 255, 255, 255);
                font-weight: bold;
                font-size: 15px;
                font-family: 'Orbitron', 'Consolas', monospace;
                text-transform: uppercase;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 40, 80, 200),
                    stop:0.5 rgba(0, 150, 255, 160),
                    stop:1 rgba(0, 245, 255, 220));
                border: 3px solid rgba(0, 245, 255, 220);
                color: white;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 245, 255, 255),
                    stop:1 rgba(0, 150, 255, 180));
                border: 4px solid rgba(255, 255, 255, 200);
            }
        """)
        return button
    def create_chat_panel(self, parent_layout):
        """Create authentic JARVIS-style holographic chat panel with white/blue theme"""
        chat_frame = QFrame()
        chat_frame.setFixedWidth(500)
        chat_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(240, 248, 255, 25),
                    stop:0.5 rgba(135, 206, 250, 35),
                    stop:1 rgba(0, 191, 255, 45));
                border: 3px solid rgba(255, 255, 255, 180);
                border-radius: 20px;
                background-clip: padding-box;
            }
        """)
        
        chat_layout = QVBoxLayout(chat_frame)
        chat_layout.setSpacing(25)
        chat_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header with authentic JARVIS styling
        header_label = QLabel("◊ J.A.R.V.I.S NEURAL INTERFACE ◊")
        header_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 255);
                font-family: 'Orbitron', 'Consolas', monospace;
                font-weight: bold;
                font-size: 20px;
                text-align: center;
                padding: 20px;
                border: none;
                border-bottom: 3px solid rgba(255, 255, 255, 150);
                margin-bottom: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 255, 255, 30),
                    stop:0.5 rgba(255, 255, 255, 60),
                    stop:1 rgba(255, 255, 255, 30));
                text-transform: uppercase;
                letter-spacing: 3px;
            }
        """)
        header_label.setAlignment(Qt.AlignCenter)
        chat_layout.addWidget(header_label)
        
        # Enhanced chat messages area with JARVIS styling
        self.chat_messages = QTextEdit()
        self.chat_messages.setReadOnly(True)
        self.chat_messages.setStyleSheet("""
            QTextEdit {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(240, 248, 255, 40),
                    stop:1 rgba(173, 216, 230, 60));
                border: 2px solid rgba(255, 255, 255, 120);
                border-radius: 15px;
                padding: 25px;
                color: rgba(25, 25, 112, 255);
                font-size: 16px;
                font-family: 'Consolas', 'Courier New', monospace;
                line-height: 1.8;
                selection-background-color: rgba(0, 191, 255, 150);
            }
            QScrollBar:vertical {
                background: rgba(240, 248, 255, 200);
                width: 16px;
                border-radius: 8px;
                border: 2px solid rgba(255, 255, 255, 120);
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(135, 206, 250, 220),
                    stop:1 rgba(255, 255, 255, 250));
                border-radius: 6px;
                min-height: 30px;
                border: 1px solid rgba(0, 191, 255, 100);
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 255);
                border: 2px solid rgba(0, 191, 255, 200);
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                border: none;
                background: none;
            }
        """)
        chat_layout.addWidget(self.chat_messages)
        
        # Control panel with JARVIS holographic styling
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            QFrame {
                background: rgba(240, 248, 255, 80);
                border: 2px solid rgba(255, 255, 255, 100);
                border-radius: 15px;
                padding: 15px;
            }
        """)
        controls_layout = QVBoxLayout(controls_frame)
        
        # Input layout
        input_layout = QHBoxLayout()
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("◊ Enter command sequence...")
        self.message_input.setStyleSheet("""
            QLineEdit {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 120),
                    stop:1 rgba(240, 248, 255, 140));
                border: 3px solid rgba(255, 255, 255, 150);
                border-radius: 12px;
                padding: 18px 25px;
                color: rgba(25, 25, 112, 255);
                font-size: 15px;
                font-family: 'Consolas', 'Courier New', monospace;
                selection-background-color: rgba(0, 191, 255, 150);
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 3px solid rgba(0, 191, 255, 220);
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 160),
                    stop:1 rgba(240, 248, 255, 180));
                color: rgba(25, 25, 112, 255);
            }
            QLineEdit::placeholder {
                color: rgba(70, 130, 180, 180);
                font-style: italic;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        # Buttons layout with JARVIS styling
        buttons_layout = QHBoxLayout()
        
        self.send_btn = self.create_jarvis_button("◊ TRANSMIT")
        self.send_btn.clicked.connect(self.send_message)
        buttons_layout.addWidget(self.send_btn)
        
        self.voice_btn = self.create_jarvis_button("🎤 VOICE")
        self.voice_btn.clicked.connect(self.toggle_voice)
        self.voice_btn.setFixedWidth(130)
        buttons_layout.addWidget(self.voice_btn)
        
        self.quit_btn = self.create_jarvis_button("◊ SHUTDOWN")
        self.quit_btn.clicked.connect(self.close)
        self.quit_btn.setFixedWidth(150)
        self.quit_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(220, 20, 60, 150),
                    stop:0.5 rgba(255, 99, 71, 120),
                    stop:1 rgba(255, 69, 0, 150));
                border: 3px solid rgba(255, 255, 255, 180);
                border-radius: 12px;
                padding: 15px 30px;
                color: rgba(255, 255, 255, 255);
                font-weight: bold;
                font-size: 15px;
                font-family: 'Orbitron', 'Consolas', monospace;
                text-transform: uppercase;
                letter-spacing: 2px;
                text-shadow: 0 0 8px rgba(255, 255, 255, 150);
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 69, 0, 200),
                    stop:0.5 rgba(255, 99, 71, 160),
                    stop:1 rgba(220, 20, 60, 200));
                border: 3px solid rgba(255, 255, 255, 220);
                box-shadow: 0 0 20px rgba(255, 69, 0, 100);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 0, 0, 255),
                    stop:1 rgba(220, 20, 60, 200));
                border: 4px solid rgba(255, 255, 255, 200);
            }
        """)
        buttons_layout.addWidget(self.quit_btn)
        
        controls_layout.addLayout(input_layout)
        controls_layout.addLayout(buttons_layout)
        chat_layout.addWidget(controls_frame)
        
        parent_layout.addWidget(chat_frame)
        
        # Welcome message with JARVIS flair
        self.add_message("◊ J.A.R.V.I.S NEURAL INTERFACE ONLINE ◊\n\nGood evening, Sir. All systems operational and standing by for your commands. How may I assist you today?", "ai")
    
    def create_system_info_panel(self, parent_layout):
        """Create system information and daily summary panel"""
        info_frame = QFrame()
        info_frame.setFixedWidth(500)
        info_frame.setMaximumHeight(200)
        info_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(240, 248, 255, 20),
                    stop:0.5 rgba(173, 216, 230, 30),
                    stop:1 rgba(135, 206, 250, 40));
                border: 2px solid rgba(255, 255, 255, 120);
                border-radius: 15px;
            }
        """)
        
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(15)
        info_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        info_header = QLabel("◊ SYSTEM STATUS ◊")
        info_header.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 255);
                font-family: 'Orbitron', monospace;
                font-weight: bold;
                font-size: 16px;
                text-align: center;
                padding: 10px;
                border-bottom: 2px solid rgba(255, 255, 255, 100);
                text-shadow: 0 0 8px rgba(255, 255, 255, 100);
            }
        """)
        info_header.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(info_header)
        
        # System info display
        self.system_info_display = QLabel()
        self.system_info_display.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 240);
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 15px;
                background: rgba(0, 20, 40, 100);
                border: 1px solid rgba(0, 162, 255, 80);
                border-radius: 8px;
            }
        """)
        self.system_info_display.setWordWrap(True)
        self.system_info_display.setTextFormat(Qt.RichText)  # Enable HTML formatting
        info_layout.addWidget(self.system_info_display)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        daily_summary_btn = self.create_jarvis_button("DAILY BRIEF")
        daily_summary_btn.clicked.connect(self.show_daily_summary)
        controls_layout.addWidget(daily_summary_btn)
        
        volume_up_btn = self.create_jarvis_button("VOL +")
        volume_up_btn.clicked.connect(lambda: self.control_system("volume_up"))
        volume_up_btn.setFixedWidth(80)
        controls_layout.addWidget(volume_up_btn)
        
        volume_down_btn = self.create_jarvis_button("VOL -")
        volume_down_btn.clicked.connect(lambda: self.control_system("volume_down"))
        volume_down_btn.setFixedWidth(80)
        controls_layout.addWidget(volume_down_btn)
        
        info_layout.addLayout(controls_layout)
        parent_layout.addWidget(info_frame)
        
        # Update system info initially
        self.update_system_info()
    
    def create_timer_panel(self, parent_layout):
        """Create timer and stopwatch control panel"""
        timer_frame = QFrame()
        timer_frame.setFixedWidth(480)
        timer_frame.setMaximumHeight(300)
        timer_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(240, 248, 255, 20),
                    stop:0.5 rgba(173, 216, 230, 30),
                    stop:1 rgba(135, 206, 250, 40));
                border: 2px solid rgba(255, 255, 255, 120);
                border-radius: 15px;
            }
        """)
        
        timer_layout = QVBoxLayout(timer_frame)
        timer_layout.setSpacing(15)
        timer_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        timer_header = QLabel("◊ TEMPORAL OPERATIONS ◊")
        timer_header.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 255);
                font-family: 'Orbitron', monospace;
                font-weight: bold;
                font-size: 16px;
                text-align: center;
                padding: 10px;
                border-bottom: 2px solid rgba(255, 255, 255, 100);
                text-shadow: 0 0 8px rgba(255, 255, 255, 100);
            }
        """)
        timer_header.setAlignment(Qt.AlignCenter)
        timer_layout.addWidget(timer_header)
        
        # Timer input
        timer_input_layout = QHBoxLayout()
        
        self.timer_input = QLineEdit()
        self.timer_input.setPlaceholderText("Timer duration (minutes)...")
        self.timer_input.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 80);
                border: 2px solid rgba(255, 255, 255, 120);
                border-radius: 8px;
                padding: 10px 15px;
                color: rgba(25, 25, 112, 255);
                font-size: 13px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
            }
            QLineEdit:focus {
                border: 2px solid rgba(0, 191, 255, 180);
                background: rgba(255, 255, 255, 120);
            }
        """)
        timer_input_layout.addWidget(self.timer_input)
        
        start_timer_btn = self.create_jarvis_button("START")
        start_timer_btn.clicked.connect(self.start_timer_gui)
        start_timer_btn.setFixedWidth(100)
        timer_input_layout.addWidget(start_timer_btn)
        
        timer_layout.addLayout(timer_input_layout)
        
        # Timer display
        self.timer_display = QLabel("No active timers")
        self.timer_display.setStyleSheet("""
            QLabel {
                color: rgba(25, 25, 112, 255);
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 10px;
                background: rgba(255, 255, 255, 40);
                border-radius: 8px;
            }
        """)
        self.timer_display.setWordWrap(True)
        timer_layout.addWidget(self.timer_display)
        
        # Control buttons
        timer_controls = QHBoxLayout()
        
        stopwatch_btn = self.create_jarvis_button("STOPWATCH")
        stopwatch_btn.clicked.connect(self.start_stopwatch_gui)
        timer_controls.addWidget(stopwatch_btn)
        
        stop_all_btn = self.create_jarvis_button("STOP ALL")
        stop_all_btn.clicked.connect(self.stop_all_timers)
        timer_controls.addWidget(stop_all_btn)
        
        timer_layout.addLayout(timer_controls)
        parent_layout.addWidget(timer_frame)
    
    def create_arc_reactor_panel(self, parent_layout):
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
        tasks_frame = QFrame()
        tasks_frame.setFixedWidth(480)
        tasks_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(240, 248, 255, 25),
                    stop:0.5 rgba(173, 216, 230, 35),
                    stop:1 rgba(135, 206, 250, 45));
                border: 3px solid rgba(255, 255, 255, 150);
                border-radius: 20px;
            }
        """)
        
        tasks_layout = QVBoxLayout(tasks_frame)
        tasks_layout.setSpacing(20)
        tasks_layout.setContentsMargins(25, 25, 25, 25)
        
        tasks_header = QLabel("◊ MISSION CONTROL CENTER ◊")
        tasks_header.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 255);
                font-family: 'Orbitron', monospace;
                font-weight: bold;
                font-size: 18px;
                text-align: center;
                padding: 15px;
                border-bottom: 3px solid rgba(255, 255, 255, 120);
                margin-bottom: 15px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 255, 255, 20),
                    stop:0.5 rgba(255, 255, 255, 40),
                    stop:1 rgba(255, 255, 255, 20));
                text-transform: uppercase;
                letter-spacing: 2px;
                text-shadow: 0 0 10px rgba(255, 255, 255, 100);
            }
        """)
        tasks_header.setAlignment(Qt.AlignCenter)
        tasks_layout.addWidget(tasks_header)
        
        self.tasks_list = QListWidget()
        self.tasks_list.setStyleSheet("""
            QListWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(240, 248, 255, 60),
                    stop:1 rgba(173, 216, 230, 80));
                border: 2px solid rgba(255, 255, 255, 120);
                border-radius: 15px;
                padding: 20px;
                color: rgba(25, 25, 112, 255);
                font-size: 15px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
            }
            QListWidget::item {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 255, 255, 80),
                    stop:1 rgba(135, 206, 250, 100));
                border: 2px solid rgba(255, 255, 255, 120);
                border-radius: 10px;
                padding: 15px 20px;
                margin: 6px 0;
                color: rgba(25, 25, 112, 255);
                min-height: 25px;
            }
            QListWidget::item:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(255, 255, 255, 120),
                    stop:1 rgba(0, 191, 255, 140));
                border: 2px solid rgba(255, 255, 255, 180);
                color: rgba(25, 25, 112, 255);
                box-shadow: 0 0 15px rgba(0, 191, 255, 80);
            }
            QListWidget::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 191, 255, 160),
                    stop:1 rgba(255, 255, 255, 140));
                border: 3px solid rgba(255, 255, 255, 200);
                color: rgba(25, 25, 112, 255);
            }
            QScrollBar:vertical {
                background: rgba(240, 248, 255, 200);
                width: 14px;
                border-radius: 7px;
                border: 2px solid rgba(255, 255, 255, 100);
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(135, 206, 250, 200),
                    stop:1 rgba(255, 255, 255, 220));
                border-radius: 6px;
                min-height: 30px;
                border: 1px solid rgba(0, 191, 255, 120);
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 255);
                border: 2px solid rgba(0, 191, 255, 200);
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                border: none;
                background: none;
            }
        """)
        self.tasks_list.itemDoubleClicked.connect(self.toggle_task)
        tasks_layout.addWidget(self.tasks_list)
        
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("◊ Add new mission objective...")
        self.task_input.setStyleSheet("""
            QLineEdit {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 100),
                    stop:1 rgba(240, 248, 255, 120));
                border: 3px solid rgba(255, 255, 255, 140);
                border-radius: 15px;
                padding: 18px 25px;
                color: rgba(25, 25, 112, 255);
                font-size: 15px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                selection-background-color: rgba(0, 191, 255, 150);
            }
            QLineEdit:focus {
                border: 3px solid rgba(0, 191, 255, 200);
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 140),
                    stop:1 rgba(240, 248, 255, 160));
                box-shadow: 0 0 15px rgba(0, 191, 255, 100);
            }
            QLineEdit::placeholder {
                color: rgba(70, 130, 180, 160);
                font-style: italic;
            }
        """)
        self.task_input.returnPressed.connect(self.add_task)
        tasks_layout.addWidget(self.task_input)

        add_task_btn = self.create_jarvis_button("◊ ADD MISSION")
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
        self.status_label = QLabel("◊ J.A.R.V.I.S ONLINE ◊")
        self.status_label.setStyleSheet("""
            QLabel {
                color: rgba(25, 25, 112, 255);
                font-family: 'Orbitron', monospace;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 150),
                    stop:1 rgba(240, 248, 255, 180));
                border: 2px solid rgba(255, 255, 255, 180);
                border-radius: 15px;
                text-shadow: 0 0 5px rgba(255, 255, 255, 100);
            }
        """)
        self.status_label.setParent(self)
        self.status_label.move(self.width()//2 - 100, self.height() - 60)
    
    def setup_timers(self):
        self.reactor_timer = QTimer()
        self.reactor_timer.timeout.connect(self.update_arc_reactor)
        self.reactor_timer.start(50)
        
        # Timer for checking notifications
        self.notification_timer = QTimer()
        self.notification_timer.timeout.connect(self.check_timer_notifications)
        self.notification_timer.start(1000)  # Check every second
        
        # Timer for updating system info
        self.system_info_timer = QTimer()
        self.system_info_timer.timeout.connect(self.update_system_info)
        self.system_info_timer.start(5000)  # Update every 5 seconds
        
    def update_arc_reactor(self):
        if hasattr(self, 'arc_reactor'):
            self.arc_reactor.update()
    
    def check_timer_notifications(self):
        """Check for timer notifications and speak them"""
        if hasattr(self.jarvis, 'timer_notifications') and self.jarvis.timer_notifications:
            notification = self.jarvis.timer_notifications.pop(0)
            self.add_message(notification, "ai")
            self.speak_text(notification)
        
        # Update timer display
        self.update_timer_display()
    
    def update_system_info(self):
        """Update system information display"""
        try:
            now = datetime.now()
            # Create bigger, more prominent date/time display
            info = f"<div style='text-align: center; margin-bottom: 15px;'>"
            info += f"<div style='font-size: 18px; font-weight: bold; color: #00f5ff; margin-bottom: 8px;'>"
            info += f"{now.strftime('%A, %B %d, %Y')}</div>"
            info += f"<div style='font-size: 22px; font-weight: bold; color: #ffffff; margin-bottom: 12px;'>"
            info += f"{now.strftime('%H:%M:%S')}</div>"
            info += f"</div>"
            
            # Add system info
            info += f"<div style='font-size: 12px; color: #00a2ff;'>"
            info += f"System: {platform.system()}<br>"
            
            # Add task summary
            pending = len([t for t in self.jarvis.tasks if t.get('status') == 'pending'])
            completed = len([t for t in self.jarvis.tasks if t.get('status') == 'completed'])
            info += f"Tasks: {pending} pending, {completed} completed</div>"
            
            self.system_info_display.setText(info)
        except Exception as e:
            self.system_info_display.setText(f"Error: {str(e)}")
    
    def update_timer_display(self):
        """Update timer display"""
        try:
            if not self.jarvis.active_timers and not self.jarvis.stopwatches:
                self.timer_display.setText("No active operations")
                return
            
            display_text = ""
            
            for name, info in self.jarvis.active_timers.items():
                elapsed = datetime.now() - info['start_time']
                remaining = timedelta(minutes=info['duration']) - elapsed
                if remaining.total_seconds() > 0:
                    display_text += f"Timer '{name}': {str(remaining).split('.')[0]}\n"
            
            for name, info in self.jarvis.stopwatches.items():
                elapsed = datetime.now() - info['start_time']
                display_text += f"Stopwatch '{name}': {str(elapsed).split('.')[0]}\n"
            
            self.timer_display.setText(display_text.strip() or "No active operations")
        except Exception as e:
            self.timer_display.setText(f"Error: {str(e)}")
    
    def show_daily_summary(self):
        """Show daily summary"""
        summary = self.jarvis.agent_daily_summary()
        self.add_message(summary, "ai")
        self.speak_text("Daily briefing ready, Sir.")
    
    def control_system(self, action):
        """Control system functions"""
        try:
            if action == "volume_up":
                result = self.jarvis.agent_control_volume("up")
            elif action == "volume_down":
                result = self.jarvis.agent_control_volume("down")
            else:
                result = f"Unknown action: {action}"
            
            self.add_message(result, "ai")
        except Exception as e:
            self.add_message(f"Error: {str(e)}", "ai")
    
    def start_timer_gui(self):
        """Start timer from GUI"""
        try:
            duration_text = self.timer_input.text().strip()
            if not duration_text:
                self.add_message("Please enter timer duration in minutes", "ai")
                return
            
            duration = float(duration_text)
            result = self.jarvis.agent_start_timer(f"{duration},GUI_Timer")
            self.add_message(result, "ai")
            self.speak_text(f"Timer started for {duration} minutes")
            self.timer_input.clear()
        except ValueError:
            self.add_message("Please enter a valid number for timer duration", "ai")
        except Exception as e:
            self.add_message(f"Error starting timer: {str(e)}", "ai")
    
    def start_stopwatch_gui(self):
        """Start stopwatch from GUI"""
        try:
            result = self.jarvis.agent_start_stopwatch("GUI_Stopwatch")
            self.add_message(result, "ai")
            self.speak_text("Stopwatch started")
        except Exception as e:
            self.add_message(f"Error starting stopwatch: {str(e)}", "ai")
    
    def stop_all_timers(self):
        """Stop all timers and stopwatches"""
        try:
            result = self.jarvis.agent_stop_timer("all")
            self.add_message(result, "ai")
            self.speak_text("All operations stopped")
        except Exception as e:
            self.add_message(f"Error stopping timers: {str(e)}", "ai")
    
    def add_message(self, message, sender):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if sender == "user":
            formatted_message = f"<div style='color: #00f5ff; margin: 15px 0; font-weight: bold;'><b>[{timestamp}] USER COMMAND:</b></div>"
            formatted_message += f"<div style='background: rgba(0, 245, 255, 20); padding: 15px; border-radius: 12px; margin: 8px 0 20px 25px; border-left: 4px solid #00f5ff; color: #ffffff; font-weight: bold;'>{message}</div>"
        else:
            formatted_message = f"<div style='color: #0080ff; margin: 15px 0; font-weight: bold;'><b>[{timestamp}] J.A.R.V.I.S RESPONSE:</b></div>"
            formatted_message += f"<div style='background: rgba(0, 128, 255, 20); padding: 15px; border-radius: 12px; margin: 8px 0 20px 25px; border-left: 4px solid #0080ff; color: #ffffff; font-weight: 500;'>{message}</div>"
        
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
        # Allow multiple speech instances but queue them properly
        self.status_label.setText("JARVIS SPEAKING...")
        
        # Clean up previous speech thread if it's finished
        if hasattr(self, 'speech_thread') and self.speech_thread and self.speech_thread.isFinished():
            self.speech_thread.deleteLater()
        
        self.speech_thread = SpeechThread(self.jarvis, text)
        self.speech_thread.speech_finished.connect(self.speech_finished)
        self.speech_thread.start()
    
    def speech_finished(self):
        self.status_label.setText("JARVIS READY")
        # Clean up the thread
        if hasattr(self, 'speech_thread') and self.speech_thread:
            self.speech_thread.deleteLater()
            self.speech_thread = None
    
    def add_task(self):
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
            
            # Use data to store styling info instead of setStyleSheet
            if task['status'] == 'completed':
                item.setData(Qt.UserRole + 1, 'completed')
            elif task['priority'] == 'high':
                item.setData(Qt.UserRole + 1, 'high')
            else:
                item.setData(Qt.UserRole + 1, 'normal')
                
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
        
        # Outer holographic field (JARVIS style)
        outer_gradient = QRadialGradient(center_x, center_y, 140)
        outer_gradient.setColorAt(0, QColor(255, 255, 255, 40))
        outer_gradient.setColorAt(0.5, QColor(135, 206, 250, 30))
        outer_gradient.setColorAt(0.8, QColor(0, 191, 255, 20))
        outer_gradient.setColorAt(1, QColor(240, 248, 255, 0))
        painter.setBrush(QBrush(outer_gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_x - 140, center_y - 140, 280, 280)
        
        # Main reactor core (JARVIS white/blue theme)
        main_gradient = QRadialGradient(center_x, center_y, 110)
        main_gradient.setColorAt(0, QColor(255, 255, 255, 220))
        main_gradient.setColorAt(0.2, QColor(240, 248, 255, 200))
        main_gradient.setColorAt(0.5, QColor(135, 206, 250, 160))
        main_gradient.setColorAt(0.8, QColor(0, 191, 255, 120))
        main_gradient.setColorAt(1, QColor(25, 25, 112, 150))
        painter.setBrush(QBrush(main_gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 180), 3))
        reactor_size = int(220 * self.pulse_scale)
        painter.drawEllipse(center_x - reactor_size//2, center_y - reactor_size//2, reactor_size, reactor_size)
        
        # Rotating energy rings (JARVIS style)
        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(self.rotation)
        painter.setPen(QPen(QColor(255, 255, 255, 200), 3))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(-90, -90, 180, 180)
        painter.rotate(-self.rotation * 1.5)
        painter.setPen(QPen(QColor(0, 191, 255, 150), 2))
        painter.drawEllipse(-70, -70, 140, 140)
        painter.rotate(self.rotation * 2)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        painter.drawEllipse(-50, -50, 100, 100)
        painter.restore()
        
        # Central core (bright white JARVIS core)
        core_gradient = QRadialGradient(center_x, center_y, 35)
        core_gradient.setColorAt(0, QColor(255, 255, 255, 255))
        core_gradient.setColorAt(0.3, QColor(240, 248, 255, 240))
        core_gradient.setColorAt(0.7, QColor(135, 206, 250, 200))
        core_gradient.setColorAt(1, QColor(0, 191, 255, 180))
        painter.setBrush(QBrush(core_gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
        core_size = int(70 * self.pulse_scale)
        painter.drawEllipse(center_x - core_size//2, center_y - core_size//2, core_size, core_size)
        
        # JARVIS-style energy particles
        painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
        for i in range(8):
            angle = (self.rotation + i * 45) * 3.14159 / 180
            x = center_x + 100 * np.cos(angle)
            y = center_y + 100 * np.sin(angle)
            painter.drawEllipse(int(x-2), int(y-2), 4, 4)
        
        self.rotation += 1.5
        if self.rotation >= 360:
            self.rotation = 0
        
        self.pulse_scale += 0.003 * self.pulse_direction
        if self.pulse_scale >= 1.08:
            self.pulse_direction = -1
        elif self.pulse_scale <= 0.92:
            self.pulse_direction = 1

def main():
    try:
        logger.info("Starting Jarvis Interface...")
        jarvis = JarvisAI()
        logger.info("Jarvis Online!")
        app = QApplication(sys.argv)
        app.setApplicationName("JARVIS Neural Interface")
        app.setApplicationVersion("2.3")
        logger.info("Starting GUI for Jarvis...")
        jarvis_gui = JarvisGUI(jarvis)
        jarvis_gui.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Failed to start JARVIS: {e}")
        logger.error(f"Traceback: {e.__traceback__}")
        sys.exit(1)
if __name__ == "__main__":
    main()