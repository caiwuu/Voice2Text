import numpy as np
from zhconv import convert
from faster_whisper import VadOptions, get_speech_timestamps, WhisperModel, collect_chunks
import time
import threading
import queue
import gradio as gr
import av
import subprocess
import requests
import json

def check_cuda_available():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("CUDA is available.")
            return True
        else:
            print("CUDA is not available.")
            return False
    except FileNotFoundError:
        print("CUDA is not available.")
        return False



cuda_available = check_cuda_available()
# 加载模型
path = "./models/faster-whisper-large-v2"
model = WhisperModel(model_size_or_path=path, device="cuda", compute_type="float16") if cuda_available else WhisperModel(model_size_or_path=path, device="cpu", compute_type="int8")
# 初始化参数
channels = 1
sampling_rate = 48000
duration = 10  # 定义录音时长
recording = False
audio_file = None
audio_stream_data = None
data_queue = queue.Queue()
total_frames = np.array([], dtype=np.float32)
vad_parameters = VadOptions(max_speech_duration_s=float(5))
text_arr = []
beam_size = 5
language = "zh"
is_subtitles = True,
ai_srvice = 'http://gpt.wyzhp.site/backend-anon/conversations'
language_arr = ["af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "zh-TW"]
language_label = ["南非荷兰语", "阿姆哈拉语", "阿拉伯语", "阿萨姆语", "阿塞拜疆语", "巴什基尔语", "白俄罗斯语", "保加利亚语", "孟加拉语", "藏语", "布列塔尼语", "波斯尼亚语", "加泰罗尼亚语", "捷克语", "威尔士语", "丹麦语", "德语", "希腊语", "英语", "西班牙语", "爱沙尼亚语", "巴斯克语", "波斯语", "芬兰语", "法罗语", "法语", "加利西亚语", "古吉拉特语", "豪萨语", "夏威夷语", "希伯来语", "印地语", "克罗地亚语", "海地克里奥尔语", "匈牙利语", "亚美尼亚语", "印度尼西亚语", "冰岛语", "意大利语", "日语", "爪哇语", "格鲁吉亚语", "哈萨克语", "高棉语", "卡纳达语", "韩语", "拉丁语", "卢森堡语", "林加拉语", "老挝语", "立陶宛语", "拉脱维亚语", "马达加斯加语", "毛利语", "马其顿语", "马拉雅拉姆语", "蒙古语", "马拉地语", "马来语", "马耳他语", "缅甸语", "尼泊尔语", "荷兰语", "挪威语（新挪威语）", "挪威语", "奥克西唐语", "旁遮普语", "波兰语", "普什图语", "葡萄牙语", "罗马尼亚语", "俄语", "梵语", "信德语", "僧伽罗语", "斯洛伐克语", "斯洛文尼亚语", "修纳语", "索马里语", "阿尔巴尼亚语", "塞尔维亚语", "巽他语", "瑞典语", "斯瓦希里语", "泰米尔语", "泰卢固语", "塔吉克语", "泰语", "土库曼语", "塔加洛语", "土耳其语", "鞑靼语", "乌克兰语", "乌尔都语", "乌兹别克语", "越南语", "意第绪语", "约鲁巴语", "中文", "中文繁体"]
sys_prompt = "用户给你提供的是语音识别的一些文字，这些文字可能存在断句问题和错别字。错别字可能是识别成拼音相近的字了，请你从读音和上下文，对它进行修复，直接返回修复结果，语言类型保持和用户的一致"





def format_time(seconds):
    # 将输入的时间（秒）转换为总秒数
    total_seconds = float(seconds)
    
    # 计算小时、分钟和秒
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    
    # 将秒拆分为整数秒和毫秒部分
    int_secs = int(secs)
    milliseconds = int((secs - int_secs) * 1000)
    
    # 格式化为00:00:00.000的形式
    formatted_time = f"{hours:02}:{minutes:02}:{int_secs:02}.{milliseconds:03}"
    
    return formatted_time

def transcribe(audio):
    lang = 'zh' if language=='zh-TW' else language
    segments, info = model.transcribe(audio, beam_size=beam_size, language=lang,vad_filter=True)
    result = ''
    if(is_subtitles and not recording):
        # result = ','.join(segment.text for segment in segments)
        for index,segment in enumerate(segments):
            result += f'{index+1}\r\n{format_time(segment.start)} --> {format_time(segment.end)}\r\n{segment.text}\r\n\r\n'
    else:
        result = ','.join(segment.text for segment in segments)

    if(language=='zh'):
        result = convert(result, 'zh-cn')
    return result


def update_data():
    global recording
    while recording:
        text = ','.join(text_arr)
        yield text
        time.sleep(0.5)  # 每秒更新

def ai_summary(text):
    ai_text = ''
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5c'
    }
    data = {
            "messages":  [
                {
                    "role":"system",
                    "content": sys_prompt
                },
                {
                    "role":"user",
                    "content": text
                }
            ],
            "model": "gpt-4o-mini",
            "stream":True
    }
    response = requests.post(ai_srvice, headers=headers, json=data)
    # 如果使用流模式，你需要逐步读取响应
    try:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if(line[6]=='{'):
                    ai_text += json.loads(line[6:])["choices"][0]["delta"].get("content",'')
                    yield ai_text
    except Exception as e:
        print(e)
        yield "调用AI服务发送错误"


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
                if len(text_arr):text_arr.pop() 
                text_arr.append(res)
                if(len(speech_chunks)>1):
                    total_frames = total_frames[speech_chunks[1]['start']:]
                    text_arr.append("....")
    
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
    global audio_file, audio_stream_data,recording,total_frames,text_arr
    total_frames = np.array([], dtype=np.float32)
    text_arr = []
    recording = True
    audio_file = av.open("./temp.wav", mode="w")
    audio_stream_data = audio_file.add_stream("pcm_s16le", rate=48000,channels = 1)
    threading.Thread(target=transcribe_process, daemon=True).start()
    for text in update_data():
        yield text

def stop_recording():
    global audio_stream_data,recording,audio_file
    print("stop_recording")
    time.sleep(1)
    recording = False
    audio_file.close()
    return "./temp.wav"

def file_uploaded(file):
    global text_arr
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
    text_arr = [text]
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

def settings_change(beam_size_v,language_index_v,is_subtitles_v):
    global beam_size,language,is_subtitles,language_arr
    beam_size = beam_size_v
    language = language_arr[language_index_v]
    is_subtitles = is_subtitles_v
def language_value():
    global language,language_arr,language_label
    return language_label[language_arr.index(language)]
def sys_prompt_change(prompt):
    global sys_prompt
    sys_prompt = prompt

md_notice = "检测到你的计算机CUDA可用,已经为你切换到GPU模式" if cuda_available else "检测到你的计算机CUDA不可用,已经为你切换到CPU模式;该模式速度较慢,实时转录可能无法流畅使用"
with gr.Blocks() as iface:
    gr.Markdown("<h4>GitHub仓库地址(欢迎star、贡献代码)：<a href='https://github.com/caiwuu/Voice2Text'>Voice2Text</a>；模型仓库：<a href='https://huggingface.co/Systran'>https://huggingface.co/Systran</a></h4>")
    gr.Markdown(f"<h4>{md_notice}</h4>")
    with gr.Row():
        input_record = gr.Audio(label="实时转录",sources=["microphone"], streaming=True)
        upload_file = gr.File(label="上传音频或者视频")
        with gr.Column():
            beam_size_slider = gr.Slider(2, 10,step=1.0, value=beam_size,info="增加可提高识别率，也会牺牲性能", label="beam_size",interactive=True,)
            language_label_selector = gr.Dropdown(language_label,type="index", value=language_value, label="输出语言",interactive=True)
            output_type_checkbox  = gr.Checkbox(label="字幕格式",value=is_subtitles, interactive=True,info="实时转录无法使字幕格式")
            beam_size_slider.change(settings_change,inputs=[beam_size_slider,language_label_selector,output_type_checkbox])
            language_label_selector.change(settings_change,inputs=[beam_size_slider,language_label_selector,output_type_checkbox])
            output_type_checkbox.change(settings_change,inputs=[beam_size_slider,language_label_selector,output_type_checkbox])
    
    with gr.Row():
        with gr.Column():
            play_audio = gr.Audio(label="提取音频", type="filepath")

    with gr.Row():
        regen = gr.Button("重新转录",variant="primary")
            
    output_text = gr.Textbox(label="转录结果", interactive=True)            

    summary_button = gr.Button("AI润色",variant="primary")
    prompt_text = gr.Textbox(label="提示词", interactive=True,value=sys_prompt) 
    summary_output = gr.Textbox(label="AI 润色结果", interactive=True)
    summary_button.click(ai_summary,inputs=output_text,outputs=summary_output)

    prompt_text.change(sys_prompt_change,inputs=[prompt_text])
    regen.click(re_transcribe,outputs=output_text)
    upload_file.upload(file_uploaded, inputs=upload_file, outputs=[play_audio,output_text])
    input_record.change(input_audio_change, inputs=input_record)
    input_record.start_recording(start_recording,outputs=output_text)
    input_record.stop_recording(stop_recording,outputs=play_audio)
    


iface.launch()
