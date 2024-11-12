# alert.py
import pygame

class Alert:
    def __init__(self, sound_file):
        pygame.mixer.init()  
        self.alert_sound = pygame.mixer.Sound(sound_file)  # Cargar el sonido de alerta

    def play_alert(self):
        """Reproduce el sonido de alerta"""
        self.alert_sound.play()
