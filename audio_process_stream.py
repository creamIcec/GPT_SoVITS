from app.GPT_SoVITS.audio_process import get_tts_wav
import tempfile;
from pydub import AudioSegment;
import io, wave

from app.GPT_SoVITS.tools.i18n.i18n import I18nAuto;

i18n = I18nAuto();

# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()

# from https://github.com/RVC-Boss/GPT-SoVITS/blob/a82f61393290f2b1a629417c22d0e364b6264c3a/GPT_SoVITS/inference_stream.py
def get_streaming_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    how_to_cut=i18n("按标点符号切"),
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free = False,
    byte_stream=True,
):
    """生成流式语音。

    Args:
        ref_wav_path (str): 参考音频的路径。
        prompt_text (str): 参考文本的内容。
        prompt_language (str): 参考文本的语言代号。必须存在于 `dict_language` 字典中。
        text (str): 待合成的文本内容。
        text_language (str): 待合成的文本的语言代号。必须存在于 `dict_language` 字典中。
        how_to_cut (str): 切分方式。必须存在于 `i18n` 字典中。
        top_k (int): 单步累计采用Token数。越大单次预测生成的音频越长; 过小可能导致生成不连续, 过大可能导致生成效果变差。
        top_p (int): 单步累计采用的概率阈值。越大时将会有"更远"的Token被采用; 和top_k的作用类似, 使用时, 优先考虑top_k, 再考虑top_p。
        temperature (_type_): 控制生成音频的随机性。以1为基准, 越大于1的值越随机, 越小于1的值越稳定。
        ref_free (_type_): 是否使用参考音频。
        byte_stream (bool, optional): 是否按照字节流式传输。默认为True, 否则以小文件的方式流式传输。

    Yields:
        chunk (bytes): 字节流式传输的音频;
        file(str): 小文件式传输的音频文件路径

    """
    chunks = get_tts_wav(
        ref_wav_path=ref_wav_path,
        prompt_text=prompt_text,
        prompt_language=prompt_language,
        text=text,
        text_language=text_language,
        how_to_cut=how_to_cut,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        ref_free=ref_free,
        stream=True,
    );

    if byte_stream:
        #yield wave_header_chunk();
        for chunk in chunks:
            yield chunk;
    else:
        i = 0;
        format = "wav";
        for chunk in chunks:
            i += 1;
            file = f"{tempfile.gettempdir()}/{i}.{format}";
            segment = AudioSegment(chunk, frame_rate=32000, sample_width=2, channels=1);
            segment.export(file, format=format);
            print("file:" + file);
            with open(file, 'rb') as f:
                yield f.read()  # Yield 文件内容



# 在server中添加一个测试路由，用于流式返回音频