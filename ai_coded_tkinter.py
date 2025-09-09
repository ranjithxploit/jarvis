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
                print(f"TTS Voice: {voices[0].name}")
        except Exception as e:
            print(f"Voice initialization warning: {e}")
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)
                print("Fallback TTS initialized")
            except Exception as e2:
                print(f"TTS initialization failed: {e2}")
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
            print(f"AI Error: {e}")
            return f"Error: {str(e)}"
    
    def listen(self):
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            print("Processing speech...")
            text = self.recognizer.recognize_google(audio, language='en-US')
            print(f"Recognized: {text}")
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
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')
        
        header_frame = tk.Frame(self.root, bg='#2d2d2d', height=80)
        header_frame.pack(fill='x', padx=10, pady=(10, 5))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="VORTEX", 
                              font=('Arial', 24, 'bold'), 
                              fg='#00ffff', bg='#2d2d2d')
        title_label.pack(side='left', padx=20, pady=20)
        
        button_frame = tk.Frame(header_frame, bg='#2d2d2d')
        button_frame.pack(side='right', padx=20, pady=15)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Custom.TButton',
                       background='#404040',
                       foreground='#00ffff',
                       borderwidth=1,
                       focuscolor='none',
                       font=('Arial', 10, 'bold'))
        style.map('Custom.TButton',
                 background=[('active', '#505050')])
        
        self.voice_btn = ttk.Button(button_frame, text="MIC", 
                                   command=self.voice_input,
                                   style='Custom.TButton',
                                   width=8)
        self.voice_btn.pack(side='left', padx=5)
        
        self.send_btn = ttk.Button(button_frame, text="SEND", 
                                  command=self.send_message,
                                  style='Custom.TButton',
                                  width=8)
        self.send_btn.pack(side='left', padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="CLEAR", 
                                   command=self.clear_chat,
                                   style='Custom.TButton',
                                   width=8)
        self.clear_btn.pack(side='left', padx=5)
        
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.chat_display = scrolledtext.ScrolledText(
            main_frame,
            wrap='word',
            width=80,
            height=20,
            bg='#0d1117',
            fg='#c9d1d9',
            insertbackground='#00ffff',
            font=('Consolas', 11),
            borderwidth=2,
            relief='solid',
            highlightthickness=1,
            highlightcolor='#00ffff'
        )
        self.chat_display.pack(fill='both', expand=True, pady=(0, 10))
        
        input_frame = tk.Frame(main_frame, bg='#1a1a1a')
        input_frame.pack(fill='x', pady=(0, 10))
        
        self.input_field = tk.Entry(
            input_frame,
            bg='#21262d',
            fg='#c9d1d9',
            insertbackground='#00ffff',
            font=('Consolas', 12),
            borderwidth=2,
            relief='solid',
            highlightthickness=1,
            highlightcolor='#00ffff'
        )
        self.input_field.pack(fill='x', padx=(0, 10), ipady=8)
        self.input_field.bind('<Return>', lambda event: self.send_message())
        
        self.add_message("VORTEX", "System ready. How may I assist you?")
        self.input_field.focus()
    
    def add_message(self, sender, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {sender}: {message}\n\n"
        
        self.chat_display.insert('end', formatted)
        
        if sender != "You":
            self.chat_display.tag_add("vortex", f"end-{len(formatted)}c", "end-1c")
            self.chat_display.tag_config("vortex", foreground="#FFFF00", font=('Consolas', 11))
        
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
            try:
                self.vortex.speak(ai_response)
            except Exception as e:
                print(f"TTS Error: {e}")
        
        threading.Thread(target=speak_response, daemon=True).start()
    
    def voice_input(self):
        if not self.is_listening:
            self.is_listening = True
            self.voice_btn.configure(text="STOP")
            
            def listen_thread():
                try:
                    message = self.vortex.listen()
                    self.root.after(0, lambda: self.handle_voice_result(message))
                except Exception as e:
                    self.root.after(0, lambda: self.handle_voice_error(str(e)))
            
            threading.Thread(target=listen_thread, daemon=True).start()
        else:
            self.is_listening = False
            self.voice_btn.configure(text="MIC")
    
    def handle_voice_result(self, message):
        self.is_listening = False
        self.voice_btn.configure(text="MIC")
        
        if "Could not understand" in message or "error" in message.lower():
            self.add_message("VORTEX", message)
        else:
            self.add_message("VORTEX", f"I heard: '{message}'. Processing your request...")
            self.input_field.delete(0, 'end')
            self.input_field.insert(0, message)
            self.send_message()
    
    def handle_voice_error(self, error):
        self.is_listening = False
        self.voice_btn.configure(text="MIC")
        self.add_message("VORTEX", f"Voice recognition issue: {error}. Please try again.")
    
    def clear_chat(self):
        self.chat_display.delete(1.0, 'end')
        self.add_message("VORTEX", "Chat cleared. How can I help you?")
    
    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            if hasattr(self.vortex, 'tts_engine') and self.vortex.tts_engine:
                try:
                    self.vortex.tts_engine.stop()
                except:
                    pass

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("VORTEX AI Assistant")
        print("Usage: python ai_coded_tkinter.py")
        return
    
    try:
        app = VortexApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        print("VORTEX shutting down...")

if __name__ == "__main__":
    main()
