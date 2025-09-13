#!/usr/bin/env python3
import os
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
import schedule
from pathlib import Path
import logging
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import pyaudio
import wave
import audioop
import subprocess
import platform
import sys
import re

def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"
                               "\U0001F300-\U0001F5FF"
                               "\U0001F680-\U0001F6FF"
                               "\U0001F1E0-\U0001F1FF"
                               "\U00002600-\U000027BF"
                               "\U0001f900-\U0001f9ff"
                               "\U00002700-\U000027bf"
                               "\U0001f018-\U0001f270"
                               "\U00002B50"
                               "\U00002728"
                               "\U0001F916"
                               "\U0001F609"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text).strip()

from dotenv import load_dotenv

try:
    load_dotenv()
    print("Env loaded from dotenv.")
    print("Env loaded from dotenv.")
except ImportError:
    print("import failed or install dotenv")
except Exception as e:
    print(f"Error on loading dotenv file: {e}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  

class JarvisBrain:
    def __init__(self, groq_api_key: str, data_dir: str = "jarvis_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("check gemini api in env")
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        if not self.test_gemini_connection():
            raise ConnectionError("Failed to connect. Check gemini api and your netwoork!")
        
        #groq for agent process
        self.groq_model = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0.5)
        if not self.test_groq_connection():
            raise ConnectionError("Failed to connect. Check groq api and your netwoork!")
        
        
        #tts and stt
        self.tts_engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_speaking = False
        self.stop_speech = False
        self.wake_word = "jarvis"
        self.sleep_command = "bye jarvis"
        self.is_active = True
        self.listening_for_wake_word = True
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        #for data files
        self.conversations_file = self.data_dir / "conversations.xlsx"
        self.tasks_file = self.data_dir / "tasks.xlsx"
        self.reminders_file = self.data_dir / "reminders.xlsx"
        self.notes_file = self.data_dir / "notes.xlsx"
        self.vector_db_file = self.data_dir / "vector_db.pkl" #pkl
        self.faiss_index_file = self.data_dir / "faiss_index.index" #index from idx
        self.conversations = []
        self.tasks = []
        self.reminders = []
        self.notes = []
        self.vector_db = []
        self.faiss_index = None
        self.load_data()
        
        #initialize vector db
        self.initialize_vector_db()
        self.user_context = {
            "preferences": [],
            "interests": [],
            "recent_activities": [],
            "frequent_topics": {},
            "conversation_history": []
        }
        self.setup_agents()
        self.start_reminder_scheduler()
        self.conversation_chain = []
        self.setup_voice()
        logger.info("JarvisBrain initialized!!")
        
    def setup_voice(self):
        voices = self.tts_engine.getProperty('voices')
        if voices:
            voice_set = False
            
            # Prefer male voices for Iron Man JARVIS style
            preferred_voices = [
                'Microsoft David Desktop',    # Windows David (male)
                'Microsoft Mark Desktop',     # Windows Mark (male) 
                'Microsoft George Desktop',   # British male
                'Microsoft James Desktop',    # British male
                'Microsoft Daniel Desktop',   # British male
                'david',                      # Common male voice name
                'mark',                       # Common male voice name
                'male'                        # Any male voice
            ]
            
            # Try preferred voices in order
            for preferred in preferred_voices:
                for voice in voices:
                    if preferred.lower() in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        voice_set = True
                        logger.info(f"Voice set to: {voice.name}")
                        break
                if voice_set:
                    break
            
            if not voice_set:
                system = platform.system()
                if system == "Windows":
                    fallback_voices = ['david', 'mark', 'male', 'zira']
                else:
                    fallback_voices = ['male', 'english']
                
                for preferred in fallback_voices:
                    for voice in voices:
                        voice_name = voice.name.lower()
                        if preferred in voice_name:
                            self.tts_engine.setProperty('voice', voice.id)
                            voice_set = True
                            logger.info(f"Fallback voice set to: {voice.name}")
                            break
                    if voice_set:
                        break
            
            if not voice_set and voices:
                self.tts_engine.setProperty('voice', voices[0].id)
                logger.info(f"Using default voice: {voices[0].name}")
                
        speech_rate = int(os.getenv('JARVIS_SPEECH_RATE', '180'))  # Slightly slower for clarity
        speech_volume = float(os.getenv('JARVIS_SPEECH_VOLUME', '0.9'))
        
        self.tts_engine.setProperty('rate', speech_rate)
        self.tts_engine.setProperty('volume', speech_volume)
        
    def setup_agents(self):
        self.tools = [
            Tool(
                name="add_task",
                func=self.agent_add_task,
                description="Add a new task. Input should be 'task_description, due_date, priority' (due_date and priority are optional)."
            ),
            Tool(
                name="get_tasks",
                func=self.agent_get_tasks,
                description="Get tasks by status. Input should be 'status' (e.g., 'pending', 'completed')."
            ),
            Tool(
                name="list_tasks",
                func=self.agent_list_tasks,
                description="List all tasks. No input needed."
                ),
            Tool(    
                name="update_task",
                func=self.agent_update_task,
                description="Update a task. Input should be 'task_id, new_description, new_due_date, new_priority, new_status' (only task_id is mandatory)."
            ),
            Tool(
                name="add_reminder",
                func=self.agent_add_reminder,
                description="Add a new reminder. Input should be 'reminder_text, reminder_time'."
            ),
            Tool(
                name="get_reminders",
                func=self.agent_get_reminders,
                description="Get reminders by date. Input should be 'date' (format: YYYY-MM-DD)."
            ),
            Tool(
                name="list_reminders",
                func=self.agent_list_reminders,
                description="List all reminders. No input needed."
            ),
            Tool(
                name="update_reminder",
                func=self.agent_update_reminder,
                description="Update a reminder. Input should be 'reminder_id, new_text, new_time' (only reminder_id is mandatory)."
            ),
            Tool(
                name="add_note",
                func=self.agent_add_note,
                description="Add a new note. Input should be 'note_text, tags' (tags are optional, comma-separated)."
            ),
            Tool(
                name="get_notes",
                func=self.agent_get_notes,
                description="Get notes by tag. Input should be 'tag'."
            ),
            Tool(
                name="list_notes", 
                func=self.agent_list_notes,
                description="List all notes. No input needed."
            ),
            Tool(
                name="update_note",
                func=self.agent_update_note,
                description="Update a note. Input should be 'note_id, new_text, new_tags' (only note_id is mandatory)."
            ),
            Tool(
                name="search_conversations",
                func=self.agent_search_conversations,
                description="Search past conversations. Input should be 'query'."
            ),
            Tool(
                name="get_user_context",
                func=self.agent_get_user_context,
                description="Get user context information and frequent topics. No input needed."
            )
        ]
        react_template="""You are a personal assistant AI named Jarvis. You have access to the following tools:
{tools}
Use these tools to help the user with their requests. Always think step-by-step and use the tools when necessary. If you don't know the answer, it's okay to say you don't know. Be concise and to the point in your responses.

Question: the input question you must answer
Tought: you should always think what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
...(this Tought/Action/Action Input/Observation can repeat N times)        
Tought: I know the final answer
Final Answer: the final answer to the original input question

Begin!!

Question: {input}
Thought:{agent_scratchpad}"""

        self.prompt_react = PromptTemplate(
            template=react_template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )

        #react agent creation
        self.react_agent = create_react_agent(self.groq_model, tools=self.tools, prompt=self.prompt_react)
        self.agent_executor =  AgentExecutor(
            agent=self.react_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
    def agent_add_task(self,input_string: str) -> str:
        parts=input_string.split(',')
        task_desc=parts[0].strip()
        due_date=parts[1].strip() if len(parts) > 1 else None
        priority = parts[2].strip() if len(parts) > 2 else "Medium"
        return self.add_task(task_desc, due_date, priority)
    def agent_get_tasks(self, status: str = "all") -> str:
        if status == "all":
            tasks= self.get_tasks()
        else:
            tasks=self.get_tasks(status.strip())
        if not tasks:
            print(f"No tasks found {status}")
            
        task_list = []
        for task in tasks:
            task_list.append(f"ID: {task['id']}, Description: {task['description']}, Due: {task['due_date']}, Priority: {task['priority']}, Status: {task['status']}")
        return "\n".join(task_list)
    def agent_list_tasks(self, input_string: str = " ") -> str:
        tasks =self.get_tasks()
        if not tasks:
            return "No tasks found."
        task_list = []
        for task in tasks:
            task_list.append(f"ID: {task['id']}, Description: {task['description']}, Due: {task['due_date']}, Priority: {task['priority']}, Status: {task['status']}")
        return "\n".join(task_list)
    def agent_update_task(self, input_string: str) -> str:
        parts = input_string.split(',')
        if len(parts) != 2:
            return "Input is invalid, please tell task_id, new_status"
        try:
            task_id = int(parts[0].strip())
            new_status = parts[1].strip()
            return self.update_task_status(task_id, new_status)
        except ValueError:
            return "task id invalid, provide correct number."
    def agent_add_reminder(self, input_string: str) -> str:
        parts = input_string.split(',')
        if len(parts) < 2:
            return "Input is invalid, please tell reminder_text, reminder_time"
        reminder_text = parts[0].strip()
        reminder_time = parts[1].strip()
        return self.add_reminder(reminder_text, reminder_time)
    def agent_get_reminders(self, date_str: str) -> str:
        active_reminders = [ r for r in self.reminders if r['status'] == 'active']
        if not active_reminders:
            return "No active reminders found."
        reminder_list = []
        for reminder in active_reminders:
            reminder_list.append(f"ID: {reminder['id']}, Text: {reminder['text']}, Time: {reminder['time']}, Status: {reminder['status']}")
        return "\n".join(reminder_list)
    
    def agent_search_conversations(self, query: str) -> str:
        results = self.search_vector_db(query, k=3)
        if not results:
            return "No conversations found."
        context = "Previous conversations:\n"
        for i, result in enumerate(results, 1):
            context += f"{i}. {result['text'][:200]}...\n"
        return context
    def agent_list_reminders(self, input_string: str = " ") -> str:
        if not self.reminders:
            return "No reminders found."
        reminder_list = []
        for reminder in self.reminders:
            reminder_list.append(f"ID: {reminder['id']}, Text: {reminder['text']}, Time: {reminder['time']}, Status: {reminder['status']}")
        return "\n".join(reminder_list)
    def agent_update_reminder(self, input_string: str) -> str:
        parts = input_string.split(',')
        if len(parts) != 2:
            return "Input is invalid, please tell reminder_id, new_status"
        try:
            reminder_id = int(parts[0].strip())
            new_status = parts[1].strip()
            return self.update_reminder_status(reminder_id, new_status)
        except ValueError:
            return "reminder id invalid, provide correct number."
    def agent_add_note(self, input_string: str) -> str:
        parts = input_string.split(',')
        note_text = parts[0].strip()
        tags = [tag.strip() for tag in parts[1].split()] if len(parts) > 1 else []
        return self.add_note(note_text, tags)
    def agent_get_notes(self, tag: str) -> str:
        notes = self.get_notes(tag.strip())
        if not notes:
            return f"No notes found with tag '{tag}'."
        note_list = []
        for note in notes:
            note_list.append(f"ID: {note['id']}, Text: {note['text']}, Tags: {', '.join(note['tags'])}")
        return "\n".join(note_list)
    def agent_list_notes(self, input_string: str = " ") -> str:
        if not self.notes:
            return "No notes found."
        note_list = []
        for note in self.notes:
            note_list.append(f"ID: {note['id']}, Text: {note['text']}, Tags: {', '.join(note['tags'])}")
        return "\n".join(note_list)
    def agent_update_note(self, input_string: str) -> str:
        parts = input_string.split(',')
        if len(parts) < 2:
            return "Input is invalid, please tell note_id, new_text (new_tags optional)"
        try:
            note_id = int(parts[0].strip())
            new_text = parts[1].strip()
            new_tags = [tag.strip() for tag in parts[2].split()] if len(parts) > 2 else None
            return self.update_note(note_id, new_text, new_tags)
        except ValueError:
            return "note id invalid, provide correct number."
    def agent_get_user_context(self, input_string: str = " ") -> str:
        context = self.user_context
        context_sum  = (
            f"Preferences: {', '.join(context['preferences'])}\n"
            f"Interests: {', '.join(context['interests'])}\n"
            f"Recent Activities: {', '.join(context['recent_activities'])}\n"
            f"Frequent Topics: {', '.join(context['frequent_topics'])}\n"
        )
        return context_sum
    def load_data(self):
        try:
            if self.conversations_file.exists():
                self.conversations = pd.read_excel(self.conversations_file).to_dict(orient='records')
            if self.tasks_file.exists():
                self.tasks = pd.read_excel(self.tasks_file).to_dict(orient='records')
            if self.reminders_file.exists():
                self.reminders = pd.read_excel(self.reminders_file).to_dict(orient='records')
            if self.notes_file.exists():
                self.notes = pd.read_excel(self.notes_file).to_dict(orient='records')
            if self.vector_db_file.exists() and self.faiss_index_file.exists():
                with open(self.vector_db_file, 'rb') as f:
                    self.vector_db = pickle.load(f)
                self.faiss_index = faiss.read_index(str(self.faiss_index_file))
            logger.info("Data loaded Successfully.")
        except Exception as e:
                logger.error(f"error loading data: {e}")
            
    def save_data(self):
        try:
            pd.DataFrame(self.conversations).to_excel(self.conversations_file, index=False)
            pd.DataFrame(self.tasks).to_excel(self.tasks_file, index=False)
            pd.DataFrame(self.reminders).to_excel(self.reminders_file, index=False)
            pd.DataFrame(self.notes).to_excel(self.notes_file, index=False)
            with open(self.vector_db_file, 'wb') as f:
                pickle.dump(self.vector_db, f)
            if self.faiss_index:
                faiss.write_index(self.faiss_index, str(self.faiss_index_file))
            logger.info("Data saved Successfully.")
        except Exception as e:
            logger.error(f"error saving data: {e}")
    def initialize_vector_db(self):
        try:
            if not self.vector_db or self.faiss_index is None:
                self.vector_db = []
                self.faiss_index = faiss.IndexFlatL2(384)  # 384 for 'all-MiniLM-L6-v2'
                if self.vector_db:
                    vectors = np.array([item['embedding'] for item in self.vector_db]).astype('float32')
                    self.faiss_index.add(vectors)
                logger.info("Vector DB initialized.")
        except Exception as e:
            logger.error(f"error initializing vector db: {e}")
            
    def test_gemini_connection(self) -> bool:
        try:
            response = self.gemini_model.generate_content("Hello Gemini!")
            if response and response.text:
                logger.info("Gemini connection successful.")
                return True
            else:
                logger.error("Gemini connection failed.")
                return False
        except Exception as e:
            logger.error(f"Gemini connection error: {e}")
            return False
    
    def test_groq_connection(self) -> bool:
        try:
            response = self.groq_model.invoke("Hello Groq!")
            if response:
                logger.info("Groq connection successful.")
                return True
            else:
                logger.error("Groq connection failed.")
                return False
        except Exception as e:
            logger.error(f"Groq connection error: {e}")
            return False
    def ask_gemini(self, prompt: str) -> str:
        try:
            if not self.gemini_api_key:
                logger.error("gemini API not configured")
                return "My gemini AI is not properly configured sir!. Please check the API key."
            response = self.gemini_model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            elif response and hasattr(response, 'prompt_feedback'):
                feedback = response.prompt_feedback
                logger.error(f"blocked response: {feedback}")
                return "I apologize sir! but my response was blocked due to safety filters. Can you please rephrase your question sir?"
            else:
                logger.error("Gemini returned empty response")
                return "I apologize sir! but I couldn't generate a response at the moment. Please try again."
                
        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            error_type = type(e).__name__
            if "API" in str(e) or "key" in str(e).lower():
                return "I'm having trouble connecting to my AI service sir! Please check the API configuration."
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                return "I've reached my API limit sir! Please try again later."
            else:
                return f"I encountered a {error_type} error while processing your request sir! Please try again."
            
    def speak(self, text: str):
        try:
            self.is_speaking = True
            self.stop_speech= False
            self.tts_engine.runAndWait()
            sentences = text.replace('.', '.<break>').replace('!', '!<break>').replace('?', '?<break>').split('<break>')
            for sentence in sentences:
                if self.stop_speech:
                    break
                sentence = sentence.strip()
                if sentence:
                    print(f"Jarvis: {sentence}")
                    self.tts_engine.say(sentence)
                    self.tts_engine.runAndWait()
            self.is_speaking = False
        except Exception as e:
            logger.error(f"error in text to speech: {e}")
            self.is_speaking = False
            
    def stop_speaking(self):
        self.stop_speech = True
        try:
            self.tts_engine.stop()
        except:
            pass
        self.is_speaking = False
        print("Speech stopped.")
        
    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = self.recognizer.recognize_google(audio)
            return text.lower().strip()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            logger.error(f"error in speech to text: {e}")
            return None
        
    def detect_wake_word(self, text: str) -> bool:
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=4)
                text = self.recognizer.recognize_google(audio).lower().strip()
                print(text)
                return self.wake_word in text

        except:
            return False
    def add_to_vector_db(self, text: str):
        try:
            embedding = self.sentence_model.encode([text])[0]
            self.vector_db.append({
                "text": text,
                "embedding": embedding.tolist(),
                "timestamp": datetime.datetime.now().isoformat()
            })
            vector = np.array([embedding]).astype('float32')
            self.faiss_index.add(vector)
            self.save_data()
            logger.info("added to vector DB.")
        except Exception as e:
            logger.error(f"error adding to vector db: {e}")
    def search_vector_db(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            if not self.vector_db or self.faiss_index is None:
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
            logger.error(f"error searching vector database: {e}")
            return []
    def should_use_agent(self, user_input: str) -> bool:
        agent_keywords = [
            'task', 'reminder', 'schedule', 'add', 'complete', 'finish',
            'what did', 'previous', 'before', 'earlier', 'context',
            'search', 'find', 'look for', 'remember'
        ]
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in agent_keywords)

    def process_with_agent(self, user_input: str) -> str:
        try:
            result = self.agent_executor.invoke({"input": user_input})
            return result.get("output", "I got an error while processing your request.")
        except Exception as e:
            logger.error(f"error in agent processing: {e}")
            return "I got an error while processing your request. Let me try a different method."
    def generate_response_with_context(self, user_input: str) -> Dict[str, str]:
        """Generate AI response with JSON output for text and speech."""
        try:
            # Male movie-style system prompt (ASCII only)
            system_prompt = (
                "SYSTEM: You are J.A.R.V.I.S., created by Ranjith. Calm, professional, "
                "British-accented, witty yet restrained as in Iron Man films. "
                "Respond with JSON: {\"text\":\"...\", \"speech\":\"...\"}. "
                "Text: concise and structured. Speech: 1-2 sentences, max 25 words, formal tone. "
                f"User: {user_input}"
            )
            
            raw = self.ask_gemini(system_prompt).strip()
            try:
                data = json.loads(raw)
                text = data.get("text", "").strip()
                speech = data.get("speech", "").strip()
            except Exception:
                text = raw
                speech = ""
            return {"text": text, "speech": speech}
        except Exception as e:
            logger.error(f"error in generating response: {e}")
            return {"text": "I apologize sir! I encountered a technical difficulty. Could you please try that again?", "speech": "Technical difficulty encountered, sir."}
    def process_user_input(self, user_input: str) -> Dict[str, str]:
        if self.sleep_command in user_input.lower():
            self.is_active = False
            self.listening_for_wake_word = True
            return {"text": "Going back to sleep mode. Say 'Jarvis' to wake me up sir.", "speech": "Going to sleep mode, sir."}
        if self.should_use_agent(user_input):
            response_text = self.process_with_agent(user_input)
            response = {"text": response_text, "speech": ""}
        else:
            response = self.generate_response_with_context(user_input)
        
        # Extract text for conversation chain and saving
        response_text = response.get("text", "") if isinstance(response, dict) else str(response)
        self._update_conversation_chain(user_input, response_text)
        self.save_conversation(user_input, response_text)
        return response
    
    def _update_conversation_chain(self, user_input: str, ai_response: str):
        try:
            self.conversation_chain.append({
                'user': user_input,
                'jarvis': ai_response,
                'timestamp': datetime.datetime.now().isoformat()
            })
            if len(self.conversation_chain) > 10:
                self.conversation_chain = self.conversation_chain[-10:]
        except Exception as e:
            logger.error(f"error updating conversation chain: {e}")
            
    def save_conversation(self, user_input: str, ai_response: str):
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
        self.add_to_vector_db(full_text)
        self.update_user_context(user_input)
        self.save_data()
        
    def update_user_context(self, user_input: str):
        self.user_context['conversation_history'].append({
            'input': user_input,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        if len(self.user_context['conversation_history']) > 50:
            self.user_context['conversation_history'] = self.user_context['conversation_history'][-50:]
        words = user_input.lower().split()
        for word in words:
            if len(word) > 3:
                self.user_context['frequent_topics'][word] = self.user_context['frequent_topics'].get(word, 0) + 1
    
    def add_task(self, task_description: str, due_date: str = None, priority: str = "medium"):
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
    def add_reminder(self, reminder_text: str, reminder_time: str):
        reminder = {
            'id': len(self.reminders) + 1,
            'text': reminder_text,
            'time': reminder_time,
            'status': 'active',
            'created_at': datetime.datetime.now().isoformat()
        }
        self.reminders.append(reminder)
        self.save_data()
        return f"reminder set: {reminder_text} at {reminder_time}"

    def check_reminders(self):
        current_time = datetime.datetime.now().strftime("%H:%M")
        for reminder in self.reminders:
            if (reminder['status'] == 'active' and 
                reminder['time'] == current_time):
                reminder_msg = f"reminder: {reminder['text']}"
                print(reminder_msg)
                self.speak(f"Reminder: {reminder['text']}")
                reminder['status'] = 'completed'
                self.save_data()
    
    def start_reminder_scheduler(self):
        def run_scheduler():
            schedule.every().minute.do(self.check_reminders)
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def update_reminder_status(self, reminder_id: int, new_status: str):
        for reminder in self.reminders:
            if reminder['id'] == reminder_id:
                reminder['status'] = new_status
                reminder['updated_at'] = datetime.datetime.now().isoformat()
                self.save_data()
                return f"Reminder {reminder_id} status updated to {new_status}"
        return f"Reminder {reminder_id} not found"
    
    def add_note(self, note_text: str, tags: List[str] = None):
        if tags is None:
            tags = []
        note = {
            'id': len(self.notes) + 1,
            'text': note_text,
            'tags': tags,
            'created_at': datetime.datetime.now().isoformat()
        }
        self.notes.append(note)
        self.save_data()
        return f"Note added: {note_text}"
    
    def get_notes(self, tag: str = None):
        if tag:
            return [note for note in self.notes if tag in note.get('tags', [])]
        return list(self.notes)
    
    def update_note(self, note_id: int, new_text: str = None, new_tags: List[str] = None):
        for note in self.notes:
            if note['id'] == note_id:
                if new_text:
                    note['text'] = new_text
                if new_tags is not None:
                    note['tags'] = new_tags
                note['updated_at'] = datetime.datetime.now().isoformat()
                self.save_data()
                return f"Note {note_id} updated"
        return f"Note {note_id} not found"
    def run_continuous_mode(self):
        print("Continuous mode activated!")
        print(f"Say '{self.wake_word}' to activate")
        print(f"Say '{self.sleep_command}' to deactivate")
        print("Press 'S' + Enter to stop speech while Jarvis is talking")
        print("Press Ctrl+C to exit")
        
        def check_for_stop_input():
            import sys
            import select
            while True:
                if self.is_speaking:
                    try:
                        if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                            user_input = sys.stdin.readline().strip().lower()
                            if user_input in ['s', 'stop', 'quit speech']:
                                self.stop_speaking()
                    except:
                        pass
                time.sleep(0.1)
        stop_thread = threading.Thread(target=check_for_stop_input, daemon=True)
        stop_thread.start()
        
        try:
            while True:
                if self.listening_for_wake_word:
                    if self.detect_wake_word():
                        print("Wake word detected!")
                        self.speak("Yes sir! i'm heare!")
                        self.is_active = True
                        self.listening_for_wake_word = False
                
                elif self.is_active:
                    print("Listening...")
                    user_input = self.listen(timeout=10, phrase_time_limit=15)
                    
                    if user_input:
                        print(f"You said: {user_input}")
                        if user_input.lower() in ['stop talking', 'stop speech', 'quiet', 'shut up']:
                            self.stop_speaking()
                            continue
                        response = self.process_user_input(user_input)
                        response = remove_emojis(response)
                        print(f"Jarvis: {response}")
                        self.speak(response)
                        if not self.is_active:
                            print("Jarvis is now sleeping. Say 'Jarvis' to wake up.")
                    else:
                        print("No input detected. Going back to sleep mode.")
                        self.speak("Alrighty, I'll go back to sleep now. Just say Jarvis when you need me!")
                        self.is_active = False
                        self.listening_for_wake_word = True
                time.sleep(0.1)  
        except KeyboardInterrupt:
            print("\nI'm shutting down sir...")
            self.speak("I'm shutting down sir")
            self.speak("Server offline!")
            
    def run_interactive_mode(self):
        print("Jarvis interactive mode - type 'exit / quit' to quit")
        print("Type 'stop' while Jarvis is speaking.")
        print("Jarvis: Zira online. Systems nominal. Mood: caffeinated mischief. What ruin—er—what shall we beautify first, ghost?")
        def check_for_stop_command():
            while True:
                if self.is_speaking:
                    try:
                        stop_input = input()
                        if stop_input.lower() in ['stop', 's', 'quiet', 'quit']:
                            self.stop_speaking()
                    except:
                        pass
                time.sleep(0.1)
        stop_thread = threading.Thread(target=check_for_stop_command, daemon=True)
        stop_thread.start()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye', 'boi', 'xit', 'see you', 'seeyou']:
                    print("Jarvis: Aww, leaving already? Take care and see you soon sir!")
                    break
                if not user_input:
                    continue
                response = self.process_user_input(user_input)
                response = remove_emojis(response)
                print(f"\nJarvis: {response}")
                
            except KeyboardInterrupt:
                print("\nJarvis: Oops! Looks like you hit Ctrl+C. Catch you later!")
                break
            except EOFError:
                print("\nJarvis: Bye bye!")
                break

if __name__ == "__main__":
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

    if not GROQ_API_KEY:
        print("Please setup groq api in env")
        sys.exit(1)
    try:
        jarvis = JarvisBrain(GROQ_API_KEY)

        print("Choose mode for jarvis:")
        print("1. Continuous Voice Mode (with wake word detection)")
        print("2. Interactive Text Mode")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            jarvis.run_continuous_mode()
        else:
            jarvis.run_interactive_mode()
            
    except Exception as e:
        print(f"Error initializing Jarvis: {e}")
        print("Please check your APIs are correct sir.")