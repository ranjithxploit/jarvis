#!/usr/bin/env python3
import sys
import os
import json
import asyncio
import threading
import time
import logging
import re
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Optional
from io import StringIO
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import speech_recognition as sr
import pyttsx3
from jarvis_brain import JarvisBrain


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


class LogHandler(logging.Handler):
    def __init__(self, log_widget):
        super().__init__()
        self.log_widget = log_widget
        
    def emit(self, record):
        log_message = self.format(record)
        QMetaObject.invokeMethod(
            self.log_widget, 
            "add_log", 
            Qt.QueuedConnection,
            Q_ARG(str, log_message)
        )
class TransparentTextWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool
        )
        self.setStyleSheet("""
            QWidget {
                background: transparent;
                background-color: transparent;
            }
        """)
        
    def paintEvent(self, event):
        pass


class ChatWidget(TransparentTextWidget):
    message_received = pyqtSignal(str, str)
    def __init__(self, jarvis_ai: JarvisBrain, parent=None):
        super().__init__(parent)
        self.jarvis_ai = jarvis_ai
        self.is_listening = False
        self.setFixedSize(550, 500) 
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_thread = None
        self.speech_lock = threading.Lock()
        
        # Enhanced microphone setup with better ambient noise adjustment
        self.setup_microphone()
        
        self.init_ui()
        self.message_received.connect(self.add_message)
    
    def setup_microphone(self):
        """Setup microphone with proper ambient noise adjustment"""
        try:
            print("Setting up microphone...")
            with self.microphone as source:
                print("Adjusting for ambient noise...")
                # Better ambient noise adjustment with longer duration
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                print("Microphone setup complete!")
        except Exception as e:
            print(f"Microphone setup warning: {e}")
            # Continue anyway - we'll handle errors during listening
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        header = QLabel("JARVIS")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            color: #00FFFF;
            font-family: 'Consolas', 'Courier New', monospace;
            font-weight: bold;
            font-size: 26px;
            padding: 16px 0px;
            background: transparent;
        """)
        layout.addWidget(header)
        # messages space
        self.messages_area = QScrollArea()
        self.messages_area.setWidgetResizable(True)
        self.messages_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 12px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 255, 255, 0.5);
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(0, 255, 255, 0.7);
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                border: none;
                background: none;
            }
        """)
        
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setAlignment(Qt.AlignTop)
        self.messages_layout.setSpacing(16)
        self.messages_area.setWidget(self.messages_widget)

        self.add_message("JARVIS online!. Ready for commands sir!", "system")
        layout.addWidget(self.messages_area, 1)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        #voice button
        self.voice_btn = QPushButton("🎤 MIC OFF")
        self.voice_btn.setFixedHeight(45)
        self.voice_btn.setStyleSheet("""
            QPushButton {
                color: #00FF00;
                background-color: rgba(0, 0, 0, 100);
                border: 2px solid #00FF00;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                font-size: 18px;
                padding: 12px;
            }
            QPushButton:hover {
                color: #00FFFF;
                border-color: #00FFFF;
                background-color: rgba(0, 255, 255, 50);
            }
            QPushButton:pressed {
                color: #FFFFFF;
                border-color: #FFFFFF;
                background-color: rgba(255, 255, 255, 50);
            }
        """)
        self.voice_btn.clicked.connect(self.toggle_voice_listening)
        
        # Wake word button
        self.wake_word_btn = QPushButton("💤 WAKE WORD")
        self.wake_word_btn.setFixedHeight(45)
        self.wake_word_btn.setStyleSheet("""
            QPushButton {
                color: #FFFF00;
                background-color: rgba(0, 0, 0, 100);
                border: 2px solid #FFFF00;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton:hover {
                color: #FFFF99;
                border-color: #FFFF99;
                background-color: rgba(255, 255, 0, 50);
            }
        """)
        self.wake_word_btn.clicked.connect(self.start_wake_word_mode)
        
        #clear button
        clear_btn = QPushButton("CLEAR")
        clear_btn.setFixedHeight(45)
        clear_btn.setStyleSheet("""
            QPushButton {
                color: #FF6600;
                background-color: rgba(0, 0, 0, 100);
                border: 2px solid #FF6600;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                font-size: 18px;
                padding: 12px;
            }
            QPushButton:hover {
                color: #FF9900;
                border-color: #FF9900;
                background-color: rgba(255, 153, 0, 50);
            }
        """)
        clear_btn.clicked.connect(self.clear_messages)
        
        #quit button
        quit_btn = QPushButton("❌ QUIT")
        quit_btn.setFixedHeight(45)
        quit_btn.setStyleSheet("""
            QPushButton {
                color: #FF0000;
                background-color: rgba(0, 0, 0, 100);
                border: 2px solid #FF0000;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                font-size: 18px;
                padding: 12px;
            }
            QPushButton:hover {
                color: #FF6666;
                border-color: #FF6666;
                background-color: rgba(255, 102, 102, 50);
            }
            QPushButton:pressed {
                color: #FFFFFF;
                border-color: #FFFFFF;
                background-color: rgba(255, 255, 255, 100);
            }
        """)
        quit_btn.clicked.connect(self.quit_application)
        
        controls_layout.addWidget(self.voice_btn, 1)
        controls_layout.addWidget(self.wake_word_btn, 1)
        controls_layout.addWidget(clear_btn, 1)
        controls_layout.addWidget(quit_btn, 1)
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Enter command...")
        self.message_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                background-color: rgba(0, 0, 0, 150);
                border: 2px solid #00FFFF;
                border-radius: 8px;
                color: #FFFFFF;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 20px;
            }
            QLineEdit:focus {
                border-color: #00FF00;
                background-color: rgba(0, 255, 0, 30);
            }
            QLineEdit::placeholder {
                color: rgba(255, 255, 255, 0.5);
                font-size: 18px;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        send_btn = QPushButton("SEND")
        send_btn.setFixedHeight(45)
        send_btn.setStyleSheet("""
            QPushButton {
                color: #00FFFF;
                background-color: rgba(0, 0, 0, 100);
                border: 2px solid #00FFFF;
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                font-size: 18px;
                padding: 12px 24px;
            }
            QPushButton:hover {
                color: #00FF00;
                border-color: #00FF00;
                background-color: rgba(0, 255, 0, 50);
            }
        """)
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.message_input, 1)
        input_layout.addWidget(send_btn)
        layout.addLayout(controls_layout)
        layout.addLayout(input_layout)
        self.setLayout(layout)
        
    def add_message(self, message: str, sender: str):
        colors = {
            'user': '#00FFFF',
            'system': '#00FF00',
            'ai': '#FFFF00',  
            'error': '#FF0000'
        }
        
        color = colors.get(sender, '#FFFFFF')
        prefix = {
            'user': '>>> USER: ',
            'system': '>>> SYSTEM: ',
            'ai': '>>> JARVIS: ',
            'error': '>>> ERROR: '
        }.get(sender, '>>> ')
        
        message_label = QLabel(f"{prefix}{message}")
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"""
            color: {color};
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 20px;
            padding: 12px 0px;
            background: transparent;
        """)
        
        self.messages_layout.addWidget(message_label)
        
        # Auto-scroll to bottom
        QApplication.processEvents()
        self.messages_area.verticalScrollBar().setValue(
            self.messages_area.verticalScrollBar().maximum()
        )
    def send_message(self):
        message = self.message_input.text().strip()
        if not message:
            return
        self.add_message(message, "user")
        self.message_input.clear()
        threading.Thread(
            target=self.process_message,
            args=(message,),
            daemon=True
        ).start()
        
    def process_message(self, message: str):
        try:
            print(f"Processing: {message}")
            
            response = self.jarvis_ai.process_user_input(message)
            print(f"Jarvis response received: {response}")
            
            # Handle both Dict and string responses
            if isinstance(response, dict):
                display_text = response.get("text", "")
                speech_text = response.get("speech", "") or display_text
            else:
                display_text = str(response)
                speech_text = display_text
            
            # Remove emojis from display text
            display_text = remove_emojis(display_text)
            speech_text = remove_emojis(speech_text)
            
            self.message_received.emit(display_text, "ai")
            
            # Use speech text for TTS
            if speech_text:
                speech_thread = threading.Thread(
                    target=self.speak_response,
                    args=(speech_text,),
                    daemon=True,
                    name=f"SpeechThread-{int(time.time())}"
                )
                speech_thread.start()
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            print(f"Error processing message: {error_msg}")
            self.message_received.emit(error_msg, "error")
    
    def speak_response(self, response: str):
        with self.speech_lock:
            try:
                print(f"Speaking: {response}")
                
                preferred_voice = os.getenv('JARVIS_VOICE', 'Microsoft David Desktop')
                speech_rate = int(os.getenv('JARVIS_SPEECH_RATE', '180'))
                speech_volume = float(os.getenv('JARVIS_SPEECH_VOLUME', '0.9'))
                
                import pyttsx3
                engine = pyttsx3.init('sapi5')
                
                voices = engine.getProperty('voices')
                selected_voice = None
                
                print(f"Available voices: {len(voices) if voices else 0}")
                for voice in voices or []:
                    print(f"Voice: {voice.name}")
                    if preferred_voice.lower() in voice.name.lower():
                        selected_voice = voice
                        break
                
                if not selected_voice and voices:
                    for voice in voices:
                        if 'zira' in voice.name.lower():
                            selected_voice = voice
                            break
                
                if selected_voice:
                    engine.setProperty('voice', selected_voice.id)
                    print(f"Using voice: {selected_voice.name}")
                elif voices:
                    engine.setProperty('voice', voices[0].id)
                    print(f"Using default voice: {voices[0].name}")
                
                engine.setProperty('rate', speech_rate)
                engine.setProperty('volume', speech_volume)
                
                engine.say(response)
                engine.runAndWait()
                engine.stop()
                
                print("Speech completed successfully!")
                
            except Exception as e:
                print(f"Speech error: {e}")
                try:
                    import subprocess
                    cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\'{response}\')"'
                    subprocess.run(cmd, shell=True, timeout=30)
                    print("PowerShell speech fallback completed!")
                except Exception as fallback_error:
                    print(f"All speech methods failed: {fallback_error}")
    
    def speak_system_message(self, message: str):
        with self.speech_lock:
            try:
                print(f"Speaking system message: {message}")
                
                preferred_voice = os.getenv('JARVIS_VOICE', 'Microsoft David Desktop')
                speech_rate = int(os.getenv('JARVIS_SPEECH_RATE', '180'))
                speech_volume = float(os.getenv('JARVIS_SPEECH_VOLUME', '0.9'))
                
                import pyttsx3
                engine = pyttsx3.init('sapi5')
                
                voices = engine.getProperty('voices')
                selected_voice = None
                
                # First try the preferred voice
                for voice in voices or []:
                    if preferred_voice.lower() in voice.name.lower():
                        selected_voice = voice
                        break
                
                # If preferred voice not found, try common male voices
                if not selected_voice and voices:
                    male_voices = ['david', 'mark', 'george', 'james', 'paul']
                    for male_name in male_voices:
                        for voice in voices:
                            if male_name in voice.name.lower() and 'desktop' in voice.name.lower():
                                selected_voice = voice
                                break
                        if selected_voice:
                            break
                
                # If still no male voice found, use first available voice
                if not selected_voice and voices:
                    selected_voice = voices[0]
                
                if selected_voice:
                    engine.setProperty('voice', selected_voice.id)
                    print(f"System voice: {selected_voice.name}")
                elif voices:
                    engine.setProperty('voice', voices[0].id)
                    print(f"Using default voice: {voices[0].name}")
                
                engine.setProperty('rate', speech_rate)
                engine.setProperty('volume', speech_volume)
                
                engine.say(message)
                engine.runAndWait()
                engine.stop()
                
                print("System speech completed!")
                
            except Exception as e:
                print(f"System speech error: {e}")
                try:
                    import subprocess
                    cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\'{message}\')"'
                    subprocess.run(cmd, shell=True, timeout=30)
                    print("System speech fallback completed!")
                except Exception as fallback_error:
                    print(f"System speech fallback failed: {fallback_error}")
    
    def listen_for_voice(self):
        """Enhanced voice listening with better parameters and error handling"""
        while self.is_listening:
            try:
                print("Listening for voice...")
                
                # Re-adjust for ambient noise periodically
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Better listening parameters
                    audio = self.recognizer.listen(
                        source, 
                        timeout=2,          # Wait up to 2 seconds for speech to start
                        phrase_time_limit=8  # Allow up to 8 seconds for complete phrase
                    )
                
                try:
                    # Use Google Speech Recognition for better accuracy
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    
                    # Process the recognized text
                    QMetaObject.invokeMethod(
                        self,
                        "process_voice_input",
                        Qt.QueuedConnection,
                        Q_ARG(str, text)
                    )
                    
                except sr.UnknownValueError:
                    # Speech was unintelligible - continue listening
                    print("Could not understand audio, continuing...")
                    continue
                    
                except sr.RequestError as e:
                    print(f"Google Speech Recognition error: {e}")
                    # Try to continue with local recognition if available
                    continue
                    
            except sr.WaitTimeoutError:
                # No speech detected within timeout - this is normal
                continue
                
            except OSError as e:
                print(f"Microphone access error: {e}")
                # Try to reinitialize microphone
                try:
                    self.setup_microphone()
                    time.sleep(1)
                except:
                    print("Failed to reinitialize microphone")
                    break
                    
            except Exception as e:
                print(f"Voice recognition error: {e}")
                # Small delay before retrying
                time.sleep(0.5)
                continue
    
    def listen_once(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """Single voice input with improved settings"""
        try:
            print("Listening for single input...")
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen with better parameters
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            print(f"Single input recognized: {text}")
            return text.strip()
            
        except sr.WaitTimeoutError:
            print("Listening timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand the audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"Error in single listen: {e}")
            return None
    
    def detect_wake_word(self, wake_word: str = "jarvis") -> bool:
        """Detect wake word in speech input - inspired by example implementation"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Short listening for wake word detection
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
            
            text = self.recognizer.recognize_google(audio).lower().strip()
            print(f"Wake word detection heard: {text}")
            return wake_word.lower() in text
            
        except sr.WaitTimeoutError:
            return False
        except sr.UnknownValueError:
            return False
        except Exception as e:
            print(f"Wake word detection error: {e}")
            return False
    
    def start_wake_word_mode(self):
        """Start continuous wake word detection mode"""
        print("Starting wake word detection mode...")
        
        def wake_word_loop():
            while True:
                try:
                    if not self.is_listening:  # Only listen for wake word when not actively listening
                        if self.detect_wake_word():
                            print("Wake word detected! Activating voice mode...")
                            # Activate voice listening mode
                            QMetaObject.invokeMethod(
                                self,
                                "toggle_voice_listening",
                                Qt.QueuedConnection
                            )
                            time.sleep(2)  # Prevent immediate re-triggering
                    else:
                        time.sleep(0.5)  # Wait while in active listening mode
                except Exception as e:
                    print(f"Wake word loop error: {e}")
                    time.sleep(1)
        
        wake_word_thread = threading.Thread(target=wake_word_loop, daemon=True)
        wake_word_thread.start()
        
        self.add_message("Wake word detection started. Say 'Jarvis' to activate voice mode.", "system")
    
    @pyqtSlot(str)
    def process_voice_input(self, text: str):
        self.add_message(text, "user")
        threading.Thread(
            target=self.process_message,
            args=(text,),
            daemon=True
        ).start()
    def toggle_voice_listening(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            # Re-setup microphone when starting voice listening
            self.setup_microphone()
            
            self.voice_btn.setText("🎤 MIC ON")
            self.voice_btn.setStyleSheet("""
                QPushButton {
                    color: #FF0000;
                    background-color: rgba(0, 0, 0, 100);
                    border: 2px solid #FF0000;
                    border-radius: 8px;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-weight: bold;
                    font-size: 12px;
                    padding: 8px;
                }
                QPushButton:hover {
                    color: #FF6666;
                    border-color: #FF6666;
                    background-color: rgba(255, 102, 102, 50);
                }
            """)
            self.add_message("Voice listening activated - Speak now!", "system")
            self.voice_thread = threading.Thread(
                target=self.listen_for_voice,
                daemon=True
            )
            self.voice_thread.start()
            
            threading.Thread(
                target=self.speak_system_message,
                args=("Voice Mode Activated. I'm listening for your commands!",),
                daemon=True
            ).start()
        else:
            self.voice_btn.setText("MIC OFF")
            self.voice_btn.setStyleSheet("""
                QPushButton {
                    color: #00FF00;
                    background-color: rgba(0, 0, 0, 100);
                    border: 2px solid #00FF00;
                    border-radius: 8px;
                    font-family: 'Consolas', 'Courier New', monospace;
                    font-weight: bold;
                    font-size: 8px;
                    padding: 7px;
                }
                QPushButton:hover {
                    color: #00FFFF;
                    border-color: #00FFFF;
                    background-color: rgba(0, 255, 255, 50);
                }
            """)
            self.add_message("Voice listening deactivated", "system")
            threading.Thread(
                target=self.speak_system_message,
                args=("Voice mode deactivated.",),
                daemon=True
            ).start()
    def clear_messages(self):
        for i in reversed(range(self.messages_layout.count())):
            self.messages_layout.itemAt(i).widget().setParent(None)
        self.add_message("Messages cleared. Ready for commands.", "system")
    
    def quit_application(self):
        try:
            print("Quitting JARVIS application...")
            self.add_message("Shutting down JARVIS...", "system")
            self.speak_system_message("Shutting down JARVIS. Goodbye!")
            time.sleep(2)
            QApplication.quit()
        except Exception as e:
            print(f"Error quitting application: {e}")
            QApplication.quit()

class RemindersWidget(TransparentTextWidget):
    def __init__(self, jarvis_ai: JarvisBrain, parent=None):
        super().__init__(parent)
        self.jarvis_ai = jarvis_ai
        self.setFixedSize(500, 450) 
        self.init_ui()
        self.load_reminders()
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        header = QLabel("REMINDERS")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            color: #FFFF00;
            font-family: 'Consolas', 'Courier New', monospace;
            font-weight: bold;
            font-size: 16px;
            padding: 10px 0px;
            background: transparent;
        """)
        layout.addWidget(header)
        self.reminders_area = QScrollArea()
        self.reminders_area.setWidgetResizable(True)
        self.reminders_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 12px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 0, 0.5);
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 0, 0.7);
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                border: none;
                background: none;
            }
        """)
        self.reminders_widget = QWidget()
        self.reminders_layout = QVBoxLayout(self.reminders_widget)
        self.reminders_layout.setAlignment(Qt.AlignTop)
        self.reminders_layout.setSpacing(8)
        self.reminders_area.setWidget(self.reminders_widget)
        layout.addWidget(self.reminders_area, 1)
        self.reminder_input = QLineEdit()
        self.reminder_input.setPlaceholderText("Add new reminder...")
        self.reminder_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                background-color: rgba(0, 0, 0, 150);
                border: 2px solid #FFFF00;
                border-radius: 8px;
                color: #FFFFFF;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #00FF00;
                background-color: rgba(255, 255, 0, 30);
            }
            QLineEdit::placeholder {
                color: rgba(255, 255, 255, 0.5);
            }
        """)
        self.reminder_input.returnPressed.connect(self.add_reminder)
        layout.addWidget(self.reminder_input)
        self.setLayout(layout)

    def add_reminder_to_list(self, text: str, time_str: str):
        reminder_text = f"[{time_str}] {text}"
        
        reminder_label = QLabel(reminder_text)
        reminder_label.setWordWrap(True)
        reminder_label.setStyleSheet("""
            color: #FFFF00;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 13px;
            padding: 5px 0px;
            background: transparent;
        """)
        self.reminders_layout.addWidget(reminder_label)
    def add_reminder(self):
        text = self.reminder_input.text().strip()
        if not text:
            return
        time_str = datetime.now().strftime("%H:%M")
        try:
            self.jarvis_ai.add_reminder(text, time_str)
            self.add_reminder_to_list(text, time_str)
            self.reminder_input.clear()
        except Exception as e:
            error_label = QLabel(f"Error: {str(e)}")
            error_label.setStyleSheet("""
                color: #FF0000;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                background: transparent;
            """)
            self.reminders_layout.addWidget(error_label)
            
    def load_reminders(self):
        try:
            for reminder in getattr(self.jarvis_ai, 'reminders', []):
                if reminder.get('status') == 'active':
                    self.add_reminder_to_list(
                        reminder.get('text', ''),
                        reminder.get('time', '')
                    )
        except Exception as e:
            pass

class LogsWidget(TransparentTextWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(600, 400) 
        self.max_logs = 60
        self.init_ui()
        self.setup_logging()
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        header_layout = QHBoxLayout()
        header = QLabel("SYSTEM LOGS")
        header.setAlignment(Qt.AlignLeft)
        header.setStyleSheet("""
            color: #FF6600;
            font-family: 'Consolas', 'Courier New', monospace;
            font-weight: bold;
            font-size: 14px;
            padding: 10px 0px;
            background: transparent;
        """)
        # clearlogs button
        clear_btn = QPushButton("CLEAR")
        clear_btn.setFixedSize(80, 30)
        clear_btn.setStyleSheet("""
            QPushButton {
                color: #FF6600;
                background-color: rgba(0, 0, 0, 100);
                border: 2px solid #FF6600;
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #FF9900;
                border-color: #FF9900;
                background-color: rgba(255, 153, 0, 50);
            }
        """)
        clear_btn.clicked.connect(self.clear_logs)
        header_layout.addWidget(header, 1)
        header_layout.addWidget(clear_btn)
        layout.addLayout(header_layout)
        #logs area
        self.logs_area = QScrollArea()
        self.logs_area.setWidgetResizable(True)
        self.logs_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 12px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 102, 0, 0.5);
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 102, 0, 0.7);
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                border: none;
                background: none;
            }
        """)
        
        self.logs_widget = QWidget()
        self.logs_layout = QVBoxLayout(self.logs_widget)
        self.logs_layout.setAlignment(Qt.AlignTop)
        self.logs_layout.setSpacing(2)
        self.logs_area.setWidget(self.logs_widget)
        layout.addWidget(self.logs_area, 1)
        self.setLayout(layout)

        self.add_log("Jarvis initialized")
        
    def setup_logging(self):
        self.log_handler = LogHandler(self)
        self.log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self.log_handler)
        sys.stdout = LogCapture(self)
        
    @pyqtSlot(str)
    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {message}"
        log_label = QLabel(log_text)
        log_label.setWordWrap(True)
        log_label.setStyleSheet("""
            color: #FF6600;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            padding: 2px 0px;
            background: transparent;
        """)
        
        self.logs_layout.addWidget(log_label)
        if self.logs_layout.count() > self.max_logs:
            old_widget = self.logs_layout.itemAt(0).widget()
            old_widget.setParent(None)
        QApplication.processEvents()
        self.logs_area.verticalScrollBar().setValue(
            self.logs_area.verticalScrollBar().maximum()
        )
        
    def clear_logs(self):
        for i in reversed(range(self.logs_layout.count())):
            self.logs_layout.itemAt(i).widget().setParent(None)
        self.add_log("Logs cleared")


class LogCapture: #for logs    
    def __init__(self, logs_widget):
        self.logs_widget = logs_widget
        self.original_stdout = sys.stdout
    def write(self, message):
        self.original_stdout.write(message)
        if message.strip() and not message.strip().startswith('\n'):
            self.logs_widget.add_log(message.strip())
    def flush(self):
        self.original_stdout.flush()


class JarvisCleanGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            groq_api_key = os.getenv('GROQ_API_KEY', 'dummy')
            self.jarvis_ai = JarvisBrain(groq_api_key)
            print("Jarvis initialized successfully!")
        except Exception as e:
            print(f"Error in jarvis initialization {e}")
            sys.exit(1)
        self.init_ui()
    def init_ui(self):
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool
        )
        self.setGeometry(0, 0, 1920, 1080)
        self.setStyleSheet("""
            QMainWindow {
                background: transparent;
                background-color: transparent;
            }
        """)
        self.hide()
        self.setup_widgets()

    def setup_widgets(self):
        # Chat Widget
        self.chat_widget = ChatWidget(self.jarvis_ai)
        self.chat_widget.move(1920 - 530 - 20, 20)
        self.chat_widget.show()
        
        # Reminders Widget - Top Left
        self.reminders_widget = RemindersWidget(self.jarvis_ai)
        self.reminders_widget.move(20, 20)
        self.reminders_widget.show()
        
        # Logs Widget - Bottom Right
        self.logs_widget = LogsWidget()
        self.logs_widget.move(1920 - 560 - 20, 1080 - 380 - 20)
        self.logs_widget.show()
        
        # Exit Button - Bottom Left Corner
        self.exit_widget = self.create_exit_button()
        self.exit_widget.move(20, 1080 - 80 - 20)
        print("Jarvis initialized, all operation success!!")
        threading.Thread(
            target=self.chat_widget.speak_system_message,
            args=("Jarvis online, all operation successful.",),
            daemon=True
        ).start()
        
    def create_exit_button(self):
        exit_widget = TransparentTextWidget()
        exit_widget.setFixedSize(120, 80)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # Exit button
        exit_btn = QPushButton("❌ EXIT\nJARVIS")
        exit_btn.setFixedSize(100, 60)
        exit_btn.setStyleSheet("""
            QPushButton {
                color: #FF0000;
                background-color: rgba(0, 0, 0, 120);
                border: 3px solid #FF0000;
                border-radius: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-weight: bold;
                font-size: 11px;
                padding: 5px;
            }
            QPushButton:hover {
                color: #FF6666;
                border-color: #FF6666;
                background-color: rgba(255, 0, 0, 80);
            }
            QPushButton:pressed {
                color: #FFFFFF;
                border-color: #FFFFFF;
                background-color: rgba(255, 255, 255, 100);
            }
        """)
        exit_btn.clicked.connect(self.close_all)
        layout.addWidget(exit_btn, 0, Qt.AlignCenter)
        exit_widget.setLayout(layout)
        return exit_widget
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close_all()
        super().keyPressEvent(event)
        
    def close_all(self):
        try:
            print("Shutting down jarvis!")
            if hasattr(self, 'chat_widget') and self.chat_widget.is_listening:
                self.chat_widget.is_listening = False
                self.chat_widget.speak_system_message("Shutting down Jarvis! Server Offline!")
                time.sleep(2) 
            self.chat_widget.close()
            self.reminders_widget.close()
            self.logs_widget.close()
            self.exit_widget.close()
            self.close()
            print("Exited!")
        except Exception as e:
            print(f"Unable to end GUI: {e}")
        finally:
            QApplication.quit()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Jarvis")
    app.setApplicationVersion("2.2")
    try:
        gui = JarvisCleanGUI()
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error starting Jarvis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
