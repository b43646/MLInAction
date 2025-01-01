


```

"""PyAudio Example: Play a wave file."""

import wave
import sys

import pyaudio
# 每次从音频文件中读取的音频数据大小为1024字节
CHUNK = 1024

if len(sys.argv) < 2:
    print(f'Plays a wave file. Usage: {sys.argv[0]} filename.wav')
    sys.exit(-1)
# 打开音频文件
with wave.open(sys.argv[1], 'rb') as wf:
    # 创建一个 pyaudio.PyAudio 类的实例，用于处理音频流，这将初始化 PortAudio 系统资源。
    p = pyaudio.PyAudio()

    # 打开一个音频流
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 将音频数据写入到音频流中进行播放
    while len(data := wf.readframes(CHUNK)):  # Requires Python 3.8+ for :=
        stream.write(data)

    # 关闭音频流
    stream.close()

    # 释放系统资源
    p.terminate()
```
