import os
from typing import Any

from elevenlabs import play, save
from elevenlabs.client import ElevenLabs

# MODEL = "eleven_multilingual_v2"
MODEL = "eleven_flash_v2_5"
VOICE = "Chris"

class Speech:
    def __init__(self, api_key: str, repo_path: str):
        self.client = ElevenLabs(api_key=api_key)
        self.repo_path = repo_path

    def run(self, text: str):
        path = f"{self.repo_path}/{text.replace(' ', '_')}.mp3"
        if self._play_sound_if_exists(path):
            return

        audio = self._generate_speech(text)
        save(audio, path)

        self._play_sound_if_exists(path)

    def _play_sound_if_exists(self, text: str) -> bool:
        path = f"{self.repo_path}/{text.replace(' ', '_')}.mp3"

        if not os.path.isfile(path):
            return False

        with open(path, "rb") as f:
            play(f.read())

        return True

    def _generate_speech(self, text: str) -> Any:
        return self.client.generate(
            text=text,
            voice=VOICE,
            model=MODEL
        )