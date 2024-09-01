import numpy as np
from zhconv import convert
from faster_whisper import VadOptions, get_speech_timestamps, WhisperModel, collect_chunks
import time
import threading
import queue
import gradio as gr
import av


# 加载模型
path = "./models/faster-whisper-large-v2"
model = WhisperModel(model_size_or_path=path, device="cuda", compute_type="float16")

# 初始化参数
channels = 1
sampling_rate = 48000
duration = 10  # 定义录音时长
recording = False


total_frames = np.array([], dtype=np.float32)
vad_parameters = VadOptions(max_speech_duration_s=float(5))
resArr = []
    

def transcribe(audio):
    segments, info = model.transcribe(audio, beam_size=5, language='zh',vad_filter=True)
    result = ','.join(segment.text for segment in segments)
    result = convert(result, 'zh-cn')
    return result


def update_data():
    global recording
    while recording:
        res = ','.join(resArr)
        yield res
        time.sleep(0.2)  # 每秒更新

def ai_summary(transcription):
    # AI总结逻辑，这里对转录结果进行总结
    return f"总结: {transcription[:50]}..."  # 示例总结逻辑


audio_file = None
audio_stream_data = None

data_queue = queue.Queue()

# 假设 audio_stream 是一个形状为 (48000, ) 的 numpy 数组
def resample_audio(audio_stream, sample_rate):
    """
    将音频流从原始采样率重采样到目标采样率。

    :param audio_stream: 输入音频数据（numpy数组）
    :param original_rate: 原始采样率
    :param target_rate: 目标采样率
    :return: 重采样后的音频数据（numpy数组）
    """
    # 创建音频重采样器
    resampler = av.audio.resampler.AudioResampler(
        format='s16',  # 输入音频格式
        layout='mono',  # 输入音频布局
        rate=sample_rate,
    )

    # 将 numpy 数组转换为 AVFrame
    frame = av.AudioFrame.from_ndarray(audio_stream.reshape(-1, 1).T, layout='mono')
    frame.sample_rate = 48000

    # 重采样音频
    resampled_frames  = resampler.resample(frame)

   # 获取重采样后的音频数据
    resampled_audio = np.concatenate([f.to_ndarray() for f in resampled_frames ])
    
    return resampled_audio.reshape(-1)

def transcribe_process():
    print("transcribe_process")
    global recording
    while recording:
        # 每500毫秒进行一次VAD
        time.sleep(0.5)
        if not data_queue.empty():
            datas = []
            while not data_queue.empty():
                datas.append(data_queue.get())  # 从队列中获取数据
            global total_frames
            datas = resample_audio(np.concatenate(datas),16000)
            datas = datas.astype(np.float32) / 32768.0  # 转换为float32
            total_frames = np.concatenate((total_frames,datas))
            datas = []
            speech_chunks = get_speech_timestamps(total_frames, vad_parameters)
            if(len(speech_chunks)):
                transcribe_frames = collect_chunks(total_frames, [speech_chunks[0]])
                res = transcribe(transcribe_frames)
                if len(resArr):resArr.pop() 
                resArr.append(res)
                if(len(speech_chunks)>1):
                    total_frames = total_frames[speech_chunks[1]['start']:]
                    resArr.append("....")
    
def input_audio_change(data):
    global audio_stream_data, audio_file
    # 检查并调整音频数据的形状
    # 将音频数据分块写入文件
    sample_rate,audio_data = data
    try:
        data_int16 = audio_data.astype(np.int16)
        data_queue.put(data_int16, timeout=1)  # 设置超时，避免无休止阻塞
    except queue.Full:
        print("队列已满，丢弃旧数据")

    
    frame_size = 24000  # 每次写入的数据块大小（样本数）
    for start in range(0, len(audio_data), frame_size):
        end = min(start + frame_size, len(audio_data))
        chunk = audio_data[start:end].reshape(-1, 1).T

        # 创建一个 AVFrame 来存储音频数据
        frame = av.AudioFrame.from_ndarray(chunk, layout='mono')
        frame.sample_rate = sample_rate
        # 编码并写入音频帧
        packet = audio_stream_data.encode(frame)
        if packet:
            audio_file.mux(packet)

def start_recording():
    print("start_recording")
    global audio_file, audio_stream_data,recording,total_frames,resArr
    total_frames = np.array([], dtype=np.float32)
    resArr = []
    recording = True
    audio_file = av.open("./temp.wav", mode="w")
    audio_stream_data = audio_file.add_stream("pcm_s16le", rate=48000,channels = 1)
    threading.Thread(target=transcribe_process, daemon=True).start()
    for res in update_data():
        yield res

def stop_recording():
    global audio_stream_data,recording,audio_file
    print("stop_recording")
    time.sleep(2)
    recording = False
    audio_file.close()
    return "./temp.wav"

def file_uploaded(file):
    global resArr
    if(not file): return
    container = av.open(file.name)
    
    # 查找音频流
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if not audio_stream:
        raise ValueError("未找到音频流。")
    # 音频参数
    sample_rate = audio_stream.rate
    # 准备音频数据列表
    audio_data = []

    # 提取音频帧
    for frames in container.decode(audio_stream):
        samples = frames.to_ndarray()
        # 处理单声道或双声道
        if samples.shape[0] == 1:  # 单声道
            audio_data.append(samples)
        elif samples.shape[0] >= 2:  # 双声道
            audio_data.append(samples[0, :][np.newaxis, :])  # 只取左声道

    # 将音频数据拼接成一个 NumPy 数组
    audio_data = np.concatenate(audio_data, axis=1)
    container = None
    save2wav(audio_data=audio_data,sample_rate=sample_rate)
    text = transcribe("temp.wav")
    resArr = [text]
    output_wav_path = "temp.wav"
    return output_wav_path, text

        

def save2wav(audio_data, sample_rate,output_path="temp.wav"):
    audio_file = av.open(output_path, mode="w")
    audio_stream_data = audio_file.add_stream("pcm_s16le", rate=sample_rate, channels=1)
    # 规范化音频数据
    if(audio_data.dtype=="float32"):
        audio_data = np.clip(audio_data, -1.0, 1.0)  # 确保在范围内
        audio_data = (audio_data*32767).astype(np.int16)  # 转换为 int16
    else:
        audio_data = (audio_data).astype(np.int16)

    # 写入音频流
    frame_size = 24000  # 每次写入的数据块大小（样本数）
    for start in range(0, len(audio_data), frame_size):
        end = min(start + frame_size, len(audio_data))
        chunk = audio_data[start:end].reshape(-1, 1).T  # 保持为单声道

        # 创建一个 AVFrame 来存储音频数据
        frame = av.AudioFrame.from_ndarray(chunk, layout = 'mono')
        frame.sample_rate = sample_rate

        # 编码并写入音频帧
        packet = audio_stream_data.encode(frame)
        if packet:
            audio_file.mux(packet)

    audio_file.close()

def re_transcribe():
    return transcribe("temp.wav")
     
css = """
        button {
            height:100%
        }
        """
with gr.Blocks() as iface:
    gr.Markdown("<h4>GitHub仓库地址(欢迎star):<a href='https://github.com/caiwuu?tab=repositories'>https://github.com/caiwuu?tab=repositories</a></h4>")
    with gr.Row():
        input_record = gr.Audio(label="实时转录",sources=["microphone"], streaming=True)
        upload_file = gr.File(label="上传音频或者视频")  
    
    with gr.Row():
        with gr.Column():
            play_audio = gr.Audio(label="提取音频", type="filepath")
    regen = gr.Button("重新转录")
            
    output_text = gr.Textbox(label="转录结果", interactive=True)            

    summary_button = gr.Button("AI总结")
    summary_output = gr.Textbox(label="AI 总结结果", interactive=True)
    summary_button.click(ai_summary,inputs=output_text,outputs=summary_output)

    regen.click(re_transcribe,outputs=output_text)
    upload_file.upload(file_uploaded, inputs=upload_file, outputs=[play_audio,output_text])
    input_record.change(input_audio_change, inputs=input_record)
    input_record.start_recording(start_recording,outputs=output_text)
    input_record.stop_recording(stop_recording,outputs=play_audio)
    


iface.launch()
