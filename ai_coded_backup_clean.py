#!/usr/bin/env python3

import sys
import os
import pandas as pd
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from datetime import datetime
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DATA_DIR = "vortex_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class VortexCore:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.tts_engine = pyttsx3.init('sapi5')
            self.tts_engine.setProperty('rate', 200)
            self.tts_engine.setProperty('volume', 0.9)
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
        except Exception as e:
            print(f"Voice initialization error: {e}")
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)
            except Exception as e2:
                self.tts_engine = None
        
        self.conversations_file = os.path.join(DATA_DIR, "conversations.xlsx")
        self.tasks_file = os.path.join(DATA_DIR, "tasks.xlsx")
        self.reminders_file = os.path.join(DATA_DIR, "reminders.xlsx")
        self._init_data_files()
        self.conversation_history = self._load_conversations()
    
    def _init_data_files(self):
        if not os.path.exists(self.conversations_file):
            df = pd.DataFrame(columns=['timestamp', 'user_input', 'ai_response'])
            df.to_excel(self.conversations_file, index=False)
        
        if not os.path.exists(self.tasks_file):
            df = pd.DataFrame(columns=['timestamp', 'task', 'status'])
            df.to_excel(self.tasks_file, index=False)
        
        if not os.path.exists(self.reminders_file):
            df = pd.DataFrame(columns=['timestamp', 'reminder', 'datetime', 'status'])
            df.to_excel(self.reminders_file, index=False)
    
    def _load_conversations(self):
        try:
            df = pd.read_excel(self.conversations_file)
            return df.tail(5).to_dict('records')
        except Exception:
            return []
    
    def save_conversation(self, user_input, ai_response):
        try:
            df = pd.read_excel(self.conversations_file)
            new_row = pd.DataFrame([{
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_input': user_input,
                'ai_response': ai_response
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(self.conversations_file, index=False)
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def speak(self, text):
        if self.tts_engine:
            try:
                clean_text = ''.join(char for char in text if char.isalnum() or char.isspace() or char in '.,!?')
                self.tts_engine.say(clean_text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
    
    def get_ai_response(self, user_input):
        try:
            if not self.model:
                return "AI model not available. Please configure GEMINI_API_KEY."
            
            prompt = f"You are VORTEX, an AI assistant. Answer this naturally and helpfully: {user_input}"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def listen(self):
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            text = self.recognizer.recognize_google(audio, language='en-US')
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"
        except Exception as e:
            return f"Error: {e}"

class VortexApp:
    def __init__(self):
        self.vortex = VortexCore()
        self.is_listening = False
        self.setup_ui()
        
    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("VORTEX")
        self.root.geometry("900x700")
        self.root.configure(bg='#0d1117')
        
        main_container = tk.Frame(self.root, bg='#0d1117')
        main_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        header_frame = tk.Frame(main_container, bg='#161b22', relief='solid', bd=1)
        header_frame.pack(fill='x', pady=(0, 15))
        
        title_label = tk.Label(header_frame, text="VORTEX", 
                              font=('Arial', 28, 'bold'), 
                              fg='#58a6ff', bg='#161b22')
        title_label.pack(pady=20)
        
        control_frame = tk.Frame(header_frame, bg='#161b22')
        control_frame.pack(pady=(0, 20))
        
        button_style = {
            'font': ('Arial', 11, 'bold'),
            'bg': '#21262d',
            'fg': '#f0f6fc',
            'activebackground': '#30363d',
            'activeforeground': '#58a6ff',
            'relief': 'flat',
            'bd': 1,
            'padx': 20,
            'pady': 8,
            'cursor': 'hand2'
        }
        
        self.voice_btn = tk.Button(control_frame, text="MIC", command=self.voice_input, **button_style)
        self.voice_btn.pack(side='left', padx=10)
        
        self.send_btn = tk.Button(control_frame, text="SEND", command=self.send_message, **button_style)
        self.send_btn.pack(side='left', padx=10)
        
        self.clear_btn = tk.Button(control_frame, text="CLEAR", command=self.clear_chat, **button_style)
        self.clear_btn.pack(side='left', padx=10)
        
        chat_frame = tk.Frame(main_container, bg='#0d1117')
        chat_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap='word',
            bg='#0d1117',
            fg='#f0f6fc',
            insertbackground='#58a6ff',
            font=('Consolas', 11),
            selectbackground='#264f78',
            selectforeground='#ffffff',
            relief='solid',
            bd=1
        )
        self.chat_display.pack(fill='both', expand=True)
        
        input_frame = tk.Frame(main_container, bg='#0d1117')
        input_frame.pack(fill='x')
        
        self.input_field = tk.Entry(
            input_frame,
            bg='#21262d',
            fg='#f0f6fc',
            insertbackground='#58a6ff',
            font=('Consolas', 12),
            relief='solid',
            bd=1
        )
        self.input_field.pack(fill='x', ipady=10)
        self.input_field.bind('<Return>', lambda event: self.send_message())
        
        self.add_message("VORTEX", "System ready. How may I assist you?")
        self.input_field.focus()
    
    def add_message(self, sender, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if sender == "You":
            formatted_msg = f"[{timestamp}] You: {message}\n\n"
            self.chat_display.insert('end', formatted_msg)
            start_idx = f"end-{len(formatted_msg)}c"
            end_idx = "end-1c"
            self.chat_display.tag_add("user", start_idx, end_idx)
            self.chat_display.tag_config("user", foreground="#58a6ff")
        else:
            formatted_msg = f"[{timestamp}] VORTEX: {message}\n\n"
            self.chat_display.insert('end', formatted_msg)
            start_idx = f"end-{len(formatted_msg)}c"
            end_idx = "end-1c"
            self.chat_display.tag_add("vortex", start_idx, end_idx)
            self.chat_display.tag_config("vortex", foreground="#ffa657")
        
        self.chat_display.see('end')
    
    def send_message(self):
        message = self.input_field.get().strip()
        if message:
            self.input_field.delete(0, 'end')
            self.add_message("You", message)
            
            def get_response():
                response = self.vortex.get_ai_response(message)
                self.root.after(0, lambda: self.handle_ai_response(message, response))
            
            threading.Thread(target=get_response, daemon=True).start()
    
    def handle_ai_response(self, user_message, ai_response):
        self.add_message("VORTEX", ai_response)
        self.vortex.save_conversation(user_message, ai_response)
        
        def speak_response():
            self.vortex.speak(ai_response)
        
        threading.Thread(target=speak_response, daemon=True).start()
    
    def voice_input(self):
        if not self.is_listening:
            self.is_listening = True
            self.voice_btn.configure(text="STOP", bg='#da3633')
            
            def listen_thread():
                message = self.vortex.listen()
                self.root.after(0, lambda: self.handle_voice_result(message))
            
            threading.Thread(target=listen_thread, daemon=True).start()
        else:
            self.is_listening = False
            self.voice_btn.configure(text="MIC", bg='#21262d')
    
    def handle_voice_result(self, message):
        self.is_listening = False
        self.voice_btn.configure(text="MIC", bg='#21262d')
        
        if "Could not understand" in message or "error" in message.lower():
            self.add_message("VORTEX", message)
        else:
            self.add_message("VORTEX", f"I heard: '{message}'. Processing your request...")
            self.input_field.delete(0, 'end')
            self.input_field.insert(0, message)
            self.send_message()
    
    def clear_chat(self):
        self.chat_display.delete(1.0, 'end')
        self.add_message("VORTEX", "Chat cleared. How can I help you?")
    
    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Shutting down...")

def main():
    try:
        app = VortexApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        print("VORTEX shutting down...")

if __name__ == "__main__":
    main()
