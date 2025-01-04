import pyaudio
import wave
from googletrans import Translator
from gtts import gTTS
import whisper
import os
from pydub import AudioSegment
from pydub.playback import play
from collections import deque
from threading import Thread, Event

# 配置参数
INPUT_DEVICE_INDEX = 2  # 麦克风阵列设备索引
OUTPUT_DEVICE_INDEX = 5  # 虚拟音频设备索引
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
CHANNELS = 2  # 适配麦克风的最大输入通道数
RECORD_SECONDS = 10
FORMAT = pyaudio.paInt16


class FIFOQueue:
    def __init__(self):
        self.queue = deque()
        self.new_data_event = Event()

    def enqueue(self, item):
        """
        向队列尾部添加元素
        :param item: 要添加的元素
        """
        self.queue.append(item)
        self.new_data_event.set()  # 触发新数据事件

    def dequeue(self):
        """
        从队列头部移除元素并返回
        :return: 队列头部的元素，如果队列为空返回 None
        """
        if not self.is_empty():
            return self.queue.popleft()
        return None

    def is_empty(self):
        """
        检查队列是否为空
        :return: True 表示队列为空，False 表示不为空
        """
        return len(self.queue) == 0

    def size(self):
        """
        获取队列的长度
        :return: 队列的元素数量
        """
        return len(self.queue)

    def listen_for_new_data(self, callback):
        """
        监听队列，一旦有新数据就调用回调函数
        :param callback: 回调函数
        """
        # def listener():
        #     while True:
        #         self.new_data_event.wait()  # 等待新数据事件
        #         while not self.is_empty():
        #             item = self.dequeue()
        #             callback(item)
        #         self.new_data_event.clear()  # 清除事件标志
        #
        # listener_thread = Thread(target=listener)
        # listener_thread.daemon = True
        # listener_thread.start()
        pass


def record_audio():
    """从麦克风录制音频"""
    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 打开输入流
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=INPUT_DEVICE_INDEX,
    )

    print("Recording...")
    frames = []
    for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b"".join(frames)


def save_audio(audio_data, filename="temp.wav"):
    """保存音频数据到文件"""
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_data)
    wf.close()


def recognize_speech(model, filename):
    """使用 Whisper 将音频转文本"""

    def transcribe_audio(file_path):
        # 转录音频
        result = model.transcribe(file_path, language="zh")
        return result["text"]

    # 测试
    file_path = filename  # 替换为你的音频文件路径
    try:
        transcript = transcribe_audio(file_path)
        print("转录结果：", transcript)
        return transcript
    except Exception as e:
        print(f"处理失败: {e}")


def translate_to_english(text):
    """将文本翻译为英文"""
    translator = Translator()
    translated = translator.translate(text, src='zh-CN', dest='en')
    return translated.text


def text_to_speech(text, output_file="output.mp3"):
    """将文本转语音并保存"""
    tts = gTTS(text, lang='en')
    tts.save(output_file)
    return output_file
    # os.system(f"start {output_file}")  # Windows 使用 start 播放，Linux/macOS 可用 open


def output_audio(output_file):
    audio = AudioSegment.from_mp3(output_file)
    play(audio)


# 主流程
# 加载 Whisper 模型
model = whisper.load_model("medium")  # 模型大小可选：tiny, base, small, medium, large
audio_data = record_audio()
save_audio(audio_data, "temp.wav")
text = recognize_speech(model, "temp.wav")
print("识别的文本:", text)

if text:
    english_text = translate_to_english(text)
    print("翻译后的文本:", english_text)
    output_file = text_to_speech(english_text)
    output_audio(output_file)
