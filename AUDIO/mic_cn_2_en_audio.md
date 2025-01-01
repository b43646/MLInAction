

```
import pyaudio
import wave
from googletrans import Translator
from gtts import gTTS
import whisper
import os
from pydub import AudioSegment

# 配置参数
INPUT_DEVICE_INDEX = 2  # 麦克风阵列设备索引
OUTPUT_DEVICE_INDEX = 5  # 虚拟音频设备索引
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
CHANNELS = 2  # 适配麦克风的最大输入通道数
RECORD_SECONDS = 10
FORMAT = pyaudio.paInt16


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
    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    input_device_index = OUTPUT_DEVICE_INDEX
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if 'CABLE Output (VB-Audio Virtual Cable)' in device_info['name']:
            input_device_index = i

    try:
        # 将转译后生成的音频数据输出到会议软件
        wf = AudioSegment.from_mp3(output_file)
        # 打开一个音频流
        output_stream = p.open(format=p.get_format_from_width(wf.sample_width),
                               channels=wf.channels,
                               rate=wf.frame_rate,
                               output=True,
                               frames_per_buffer=CHUNK_SIZE,
                               output_device_index=input_device_index,
                               )

        # 将音频数据写入到音频流中进行播放
        while len(data := wf.readframes(CHUNK_SIZE)):  # Requires Python 3.8+ for :=
            output_stream.write(data)

    except KeyboardInterrupt:
        print("\n音频输出失败\n")

    finally:
        # 关闭流和 PyAudio
        output_stream.stop_stream()
        output_stream.close()
        p.terminate()


# 主流程
# 加载 Whisper 模型
model = whisper.load_model("base")  # 模型大小可选：tiny, base, small, medium, large
audio_data = record_audio()
save_audio(audio_data, "temp.wav")
text = recognize_speech(model, "temp.wav")
print("识别的文本:", text)

if text:
    english_text = translate_to_english(text)
    print("翻译后的文本:", english_text)
    output_file = text_to_speech(english_text)
    # TODO: 将翻译后的英文音频发送给会议软件的输入端
    # output_audio(output_file)

```
