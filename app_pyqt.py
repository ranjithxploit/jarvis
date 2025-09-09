import sys
import os
import threading
from datetime import datetime
import pandas as pd
import speech_recognition as sr
import pyttsx3
import time

try:
    import google.generativeai as genai
except Exception:
    genai = None

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
                             QLabel, QListWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QPoint
from PyQt5.QtGui import QFont

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class VortexCore:
    def __init__(self):
        self.model = None
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key and genai:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print("Gemini AI configured successfully!")
            elif api_key and not genai:
                print("GEMINI_API_KEY provided but google.generativeai package not available.")
            else:
                print("No GEMINI_API_KEY found in environment. AI responses will be simulated.")
        except Exception as e:
            print(f"AI initialization failed: {e}")

        self.tts_engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.setup_tts()
        self.setup_data_storage()
        self.task_history = []
        self.tts_lock = threading.Lock()

    def setup_tts(self):
        try:
            voices = self.tts_engine.getProperty('voices')
            if voices:
                self.tts_engine.setProperty('voice', voices[0].id)
            self.tts_engine.setProperty('rate', 200)
            self.tts_engine.setProperty('volume', 0.8)
            print("TTS initialized successfully")
        except Exception as e:
            print(f"TTS setup error: {e}")
            self.tts_engine = None

    def setup_data_storage(self):
        os.makedirs('vortex_data', exist_ok=True)
        self.conversations_file = 'vortex_data/conversations.xlsx'
        self.tasks_file = 'vortex_data/tasks.xlsx'

        if not os.path.exists(self.conversations_file):
            df = pd.DataFrame(columns=['timestamp', 'user_input', 'ai_response'])
            df.to_excel(self.conversations_file, index=False)

        if not os.path.exists(self.tasks_file):
            df = pd.DataFrame(columns=['timestamp', 'task_type', 'description', 'status'])
            df.to_excel(self.tasks_file, index=False)

    def add_task_to_history(self, task_type, description):
        timestamp = datetime.now().strftime("%H:%M:%S")
        task = {'timestamp': timestamp, 'type': task_type, 'description': description, 'status': 'Completed'}
        self.task_history.append(task)
        try:
            df = pd.read_excel(self.tasks_file)
            new_row = pd.DataFrame([{
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'task_type': task_type,
                'description': description,
                'status': 'Completed'
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(self.tasks_file, index=False)
        except Exception as e:
            print(f"Error saving task: {e}")
            #just foor the sake of it

    def save_conversation(self, user_input, ai_response):
        try:
            df = pd.read_excel(self.conversations_file)
            new_row = pd.DataFrame([{
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'user_input': user_input,
                'ai_response': ai_response
            }])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(self.conversations_file, index=False)
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def speak(self, text):
        with self.tts_lock:  # Ensure only one TTS operation at a time
            try:
                if not self.tts_engine:
                    print("TTS engine not available")
                    return
                    
                import re
                clean_text = re.sub(r'[^\n\w\s\.,!\-:\(\)]', '', text)
                clean_text = clean_text.strip()
                
                if clean_text:
                    print(f"Speaking: {clean_text[:50]}...")
                    # Stop any current speech
                    try:
                        self.tts_engine.stop()
                        time.sleep(0.1)  # Small delay to ensure stop command is processed
                    except:
                        pass
                        
                    # Speak the text
                    self.tts_engine.say(clean_text)
                    self.tts_engine.runAndWait()
                    print("Speech completed")
            except Exception as e:
                print(f"TTS Error: {e}")
                # Try to reinitialize TTS engine if it fails
                try:
                    self.tts_engine = pyttsx3.init()
                    self.setup_tts()
                    print("TTS engine reinitialized")
                except Exception as e2:
                    print(f"TTS reinit failed: {e2}")
                    self.tts_engine = None

    def get_ai_response(self, user_input):
        try:
            if not self.model:
                user_lower = user_input.lower()
                if 'hello' in user_lower or 'hi' in user_lower or 'hey' in user_lower:
                    import random
                    return random.choice([
                        "Hello! I'm VORTEX, your AI assistant. How can I help you today?",
                        "Hi there! What can I assist you with?",
                        "Hey! I'm ready to help. What do you need?"
                    ])
                if 'time' in user_lower:
                    return f"The current time is {datetime.now().strftime('%I:%M %p on %B %d, %Y')}"
                if 'browser' in user_lower:
                    self.add_task_to_history('BROWSER', 'Opening web browser')
                    return 'Opening your default browser.'
                return f"I heard '{user_input}'. I can help, but full AI requires GEMINI_API_KEY."
            
            # If model is available, use AI
            prompt = f"You are VORTEX, an assistant. User request: {user_input}\nRespond concisely." 
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error connecting to AI: {e}"

    def listen(self):
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
            text = self.recognizer.recognize_google(audio, language='en-US').lower()
            if 'vortex' in text:
                return self.listen_for_command()
            return None
        except Exception:
            return None

    def listen_for_command(self):
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
            text = self.recognizer.recognize_google(audio, language='en-US')
            return text
        except Exception as e:
            print(f"Command recognition error: {e}")
            return None


class TaskHistoryWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        title = QLabel("Task History")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #00FFFF; margin: 5px;")
        layout.addWidget(title)
        self.task_list = QListWidget()
        self.task_list.setStyleSheet("background: rgba(0,0,0,120); color: #fff; border:none;")
        layout.addWidget(self.task_list)
        self.setLayout(layout)

    def add_task(self, timestamp, task_type, description):
        self.task_list.addItem(f"[{timestamp}] {task_type}: {description}")
        if self.task_list.count() > 20:
            self.task_list.takeItem(0)


class ContinuousListenThread(QThread):
    command_detected = pyqtSignal(str)

    def __init__(self, core):
        super().__init__()
        self.core = core
        self._active = False

    def run(self):
        self._active = True
        while self._active:
            try:
                command = self.core.listen()
                if command:
                    self.command_detected.emit(command)
                    self.msleep(800)
                else:
                    self.msleep(200)
            except Exception as e:
                print(f"Continuous listening error: {e}")
                self.msleep(1000)

    def stop(self):
        self._active = False


class VoiceThread(QThread):
    voice_result = pyqtSignal(str)

    def __init__(self, vortex_core):
        super().__init__()
        self.vortex_core = vortex_core

    def run(self):
        result = self.vortex_core.listen_for_command()
        self.voice_result.emit(result)


class AIThread(QThread):
    ai_response = pyqtSignal(str, str)

    def __init__(self, vortex_core, message):
        super().__init__()
        self.vortex_core = vortex_core
        self.message = message

    def run(self):
        response = self.vortex_core.get_ai_response(self.message)
        self.ai_response.emit(self.message, response)


class VortexMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.vortex = VortexCore()
        self.listening = False
        self.init_ui()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.drag_position = QPoint()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        screen = QApplication.primaryScreen().geometry()
        window_width = 500
        window_height = 450
        self.setGeometry(screen.width() - window_width - 20, 20, window_width, window_height)

        # Task history window
        self.task_window = QWidget()
        self.task_window.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.task_window.setAttribute(Qt.WA_TranslucentBackground)
        self.task_window.setGeometry(20, 20, 300, 400)
        self.task_history = TaskHistoryWidget()
        task_layout = QVBoxLayout()
        task_layout.addWidget(self.task_history)
        self.task_window.setLayout(task_layout)
        self.task_window.show()

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        header_layout = QHBoxLayout()
        title = QLabel("VORTEX")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: #00FFFF; margin: 5px;")
        header_layout.addWidget(title)

        button_style = """
            QPushButton {
                background: rgba(0, 0, 0, 120);
                color: #00FFFF;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: normal;
                font-size: 14px;
                min-width: 40px;
                min-height: 40px;
            }
            QPushButton:hover {
                background: rgba(0, 255, 255, 50);
            }
            QPushButton:pressed {
                background: rgba(0, 255, 255, 100);
            }
        """

        self.voice_btn = QPushButton("MIC")
        self.voice_btn.setStyleSheet(button_style)
        self.voice_btn.clicked.connect(self.toggle_listening)
        self.listening_active = False
        header_layout.addWidget(self.voice_btn)

        send_btn = QPushButton("SEND")
        send_btn.setStyleSheet(button_style)
        send_btn.clicked.connect(self.send_message)
        header_layout.addWidget(send_btn)

        clear_btn = QPushButton("CLEAR")
        clear_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 0, 0, 100);
                color: #FF4444;
                border: none;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
                font-size: 14px;
                min-width: 40px;
                min-height: 40px;
            }
            QPushButton:hover {
                background: rgba(255, 68, 68, 50);
            }
            QPushButton:pressed {
                background: rgba(255, 68, 68, 100);
            }
        """)
        clear_btn.clicked.connect(self.clear_chat)
        header_layout.addWidget(clear_btn)

        self.status_label = QLabel("ONLINE")
        self.status_label.setFont(QFont("Segoe UI", 16))
        self.status_label.setStyleSheet("color: #00FF00; margin: 5px;")
        header_layout.addWidget(self.status_label)

        close_btn = QPushButton("X")
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 0, 0, 100);
                color: #FF4444;
                border: none;
                border-radius: 8px;
                padding: 5px;
                font-weight: bold;
                font-size: 16px;
                min-width: 30px;
                min-height: 30px;
            }
            QPushButton:hover {
                background: rgba(255, 68, 68, 100);
            }
        """)
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)

        layout.addLayout(header_layout)

        self.input_field = QLineEdit()
        self.input_field.setStyleSheet("""
            QLineEdit {
                background: rgba(0, 0, 0, 200);
                color: #FFFFFF;
                border: 2px solid #00FFFF;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas';
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit:focus {
                border-color: #00FFFF;
                background: rgba(0, 0, 0, 250);
            }
        """)
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        layout.addWidget(self.input_field)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 0, 0, 150);
                color: #FFFFFF;
                border: 2px solid #00FFFF;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Consolas';
                font-size: 12px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.chat_display)

        main_widget.setLayout(layout)

        self.add_message("VORTEX", "System ready. How may I assist you?")
        self.input_field.setFocus()

    def add_message(self, sender, message):
        timestamp = datetime.now().strftime("%H:%M")
        if sender == "You":
            formatted = f'<span style="color: #00FFFF; font-weight: bold;">[{timestamp}]</span> <span style="color: #FFFFFF;">{message}</span><br><br>'
        else:
            formatted = f'<span style="color: #FFFF00; font-weight: bold;">[{timestamp}]</span> <span style="color: #FFFFFF;">{message}</span><br><br>'
        self.chat_display.append(formatted)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        self.chat_display.setTextCursor(cursor)

    def send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return
        self.input_field.clear()
        self.add_message("You", message)
        self.status_label.setStyleSheet("color: #FFAA00; margin: 5px;")
        self.ai_thread = AIThread(self.vortex, message)
        self.ai_thread.ai_response.connect(self.handle_ai_response)
        self.ai_thread.start()

    def handle_ai_response(self, user_message, ai_response):
        self.add_message("VORTEX", ai_response)
        self.vortex.save_conversation(user_message, ai_response)
        for task in self.vortex.task_history[-5:]:
            self.task_history.add_task(task['timestamp'], task['type'], task['description'])

        def speak_thread():
            try:
                self.vortex.speak(ai_response)
            except Exception as e:
                print(f"TTS Error: {e}")

        threading.Thread(target=speak_thread, daemon=True).start()
        self.status_label.setStyleSheet("color: #00FF00; margin: 5px;")
        self.input_field.setFocus()

    def toggle_listening(self):
        if not hasattr(self, 'listening_active'):
            self.listening_active = False
        if not self.listening_active:
            self.listening_active = True
            self.voice_btn.setText("STOP")
            self.voice_btn.setStyleSheet("background: rgba(255,0,0,120); color: white; border:none; border-radius:20px;")
            self.update_status("Continuous listening active - Say 'VORTEX' to activate")
            self.listen_thread = ContinuousListenThread(self.vortex)
            self.listen_thread.command_detected.connect(self.process_voice_command)
            self.listen_thread.start()
        else:
            self.listening_active = False
            self.voice_btn.setText("MIC")
            self.voice_btn.setStyleSheet("")
            if hasattr(self, 'listen_thread'):
                self.listen_thread.stop()

    def process_voice_command(self, command):
        if command:
            self.input_field.setText(command)
            self.send_message()

    def update_status(self, message):
        print(message)

    def clear_chat(self):
        self.chat_display.clear()
        self.add_message("VORTEX", "Chat cleared. How may I assist you?")

    def closeEvent(self, event):
        if hasattr(self, 'task_window'):
            self.task_window.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = VortexMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
