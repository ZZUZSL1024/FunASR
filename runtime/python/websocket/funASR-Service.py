'''
功能概述：音频分段摘要
步骤：
1、传入mp3,wav格式音频文件
2、调用funasr生成分段文本内容
3、调用大模型生成摘要
4、输出结果

'''
from flask import Flask, request, jsonify
import re
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from werkzeug.utils import secure_filename
import torch
import os
import json
import numpy as np
from openai import OpenAI
from funasr import AutoModel

app = Flask(__name__)

# 避免中文在日志/响应中被转义为 ASCII 码
app.config["JSON_AS_ASCII"] = False

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SPEAKER_DB_PATH = os.path.join(os.path.dirname(__file__), "speaker_db.json")

auto_model = AutoModel(model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                  vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                  punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                  spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                    device="cuda:0" if torch.cuda.is_available() else "cpu"
                       )
# 声纹对比
sv_pipeline = pipeline(
                task='speaker-verification',
                model='damo/speech_campplus_sv_zh-cn_16k-common',
                model_revision='v1.0.0',
                device="cuda:0" if torch.cuda.is_available() else "cpu"
                )


def load_speaker_db():
    if not os.path.exists(SPEAKER_DB_PATH):
        return {}
    try:
        with open(SPEAKER_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_speaker_db(db):
    os.makedirs(os.path.dirname(SPEAKER_DB_PATH), exist_ok=True)
    with open(SPEAKER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)


def make_json_response(data, status=200):
    """Return JSON with ensure_ascii=False so 控制台/客户端都能看到中文"""
    return app.response_class(
        response=json.dumps(data, ensure_ascii=False),
        status=status,
        mimetype="application/json",
    )


# 注册声纹：传音频wav文件，实现注册音频返回音频的embedding数据
@app.route('/Register_Speaker', methods=['POST'])
def Register_Speaker():
    # 检查文件上传
    if 'file' not in request.files:
        return make_json_response({"error": "No audio file provided"}, 400)

    speaker_name = request.form.get('speaker_name') or request.args.get('speaker_name')
    if not speaker_name:
        return make_json_response({"error": "speaker_name is required"}, 400)

    file = request.files['file']
    if file.filename == '':
        return make_json_response({"error": "Empty filename"}, 400)

    # 保存上传文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        # 执行音频文件Embeding
        result = sv_pipeline([filepath], output_emb=True)
        embedding = result['embs'][0]
        # 删除临时文件
        os.remove(filepath)
        if len(embedding) == 0:
            return make_json_response({
                "status": "error",
                "result": "音频解析结果为空"
            })
        else:
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
            speaker_db = load_speaker_db()
            speaker_db[speaker_name] = embedding_list
            save_speaker_db(speaker_db)
            return make_json_response({
                "status": "success",
                "result": embedding_list,
                "speaker_name": speaker_name
            })

    except Exception as e:
        # 清理文件
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return make_json_response({"error": str(e)}, 500)

# 会议撰写
@app.route('/AsrCamWithIdentify', methods=['POST'])
def speech_recognition_Timestamp_cam_identify_speakers():
    # 检查文件上传
    if 'file' not in request.files:
        return make_json_response({"error": "No audio file provided"}, 400)

    file = request.files['file']
    if file.filename == '':
        return make_json_response({"error": "Empty filename"}, 400)

    # 保存上传文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # 2. 从FormData读取参数并解析JSON
    # 读取identify_speakers（默认False）
    identify_speakers_str = request.form.get('identify_speakers', 'false')
    identify_speakers = json.loads(identify_speakers_str.lower())  # 转为bool

    # 读取speaker_db（默认空字典）
    speaker_db = load_speaker_db()
    speaker_db_str = request.form.get('speaker_db', '{}')
    try:
        provided_db = json.loads(speaker_db_str)
        if isinstance(provided_db, dict):
            speaker_db.update(provided_db)
    except Exception:
        pass

    # 3. 验证声纹库（如果需要对比）
    if identify_speakers and (not isinstance(speaker_db, dict) or len(speaker_db) == 0):
        return make_json_response({
            "status": "error",
            "result": "声纹库为空或格式错误，无法进行对比"
        }, 400)
    try:
        # 执行语音识别
        result = auto_model.generate(input=filepath,
                                     batch_size_s=300,
                                     hotword='',
                                     )

        # 处理结果
        processed_result = process_cam_result_with_identify_speakers(result,speaker_db,filepath,identify_speakers)

        os.remove(filepath)
        if len(processed_result) == 0:
            return make_json_response({
                "status": "error",
                "result": "音频解析结果为空"
            })
        else:
            # 控制台打印时保持中文不被转义
            print(json.dumps(processed_result, ensure_ascii=False, indent=2))
            return make_json_response({
                "status": "success",
                "result": processed_result
            })

    except Exception as e:
        # 清理文件
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return make_json_response({"error": str(e)}, 500)

import torchaudio
def _extract_audio_segment(audio_path, start_sec, end_sec):
    """根据时间戳提取音频片段"""
    waveform, sample_rate = torchaudio.load(audio_path)

        # 计算起止采样点
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

        # 提取片段
    segment = waveform[:, start_sample:end_sample]

    # 保存为临时文件（pipeline需要文件路径）
    temp_path = f"/tmp/temp_segment_{start_sec}_{end_sec}.wav"
    torchaudio.save(temp_path, segment, sample_rate)

    return temp_path
# 是否使用声纹转化的结果处理
def process_cam_result_with_identify_speakers(result,speaker_db,filepath,identify_speakers=False,threshold=0.45):
    """处理ASR结果，返回包含时间和内容的JSON对象列表"""
    if not isinstance(result, list) or len(result) == 0:
        return []

    data = result[0]
    best_match = "unknown"
    best_score = 0.0
    # 创建JSON格式的输出
    output = []
    sentence_infos = data.get('sentence_info',[])

    current_sentence = {
        "spk": sentence_infos[0]["spk"],
        "spk_name": best_match,
        "confidence":best_score,
        "start": sentence_infos[0]["start"],
        "end": sentence_infos[0]["end"],
        "text": sentence_infos[0]["text"]
    }


    for i in range(1, len(sentence_infos)):
        sentence_info = sentence_infos[i]
        # 检查合并条件：相同说话人且时间间隔≤1000ms
        if (current_sentence["spk"] == sentence_info["spk"] and
                sentence_info["start"] - current_sentence["end"] <= 1000):

            # 合并文本内容（中文无需加空格）
            current_sentence["text"] += sentence_info["text"]

            # 更新整句结束时间
            current_sentence["end"] = sentence_info["end"]

        else:
            # 保存合并完成的句子
            if identify_speakers: # 提取说话人
                segment_audio = _extract_audio_segment( #提取对应的音频
                    filepath, current_sentence['start']/1000, current_sentence['end']/1000
                )
                result_b = sv_pipeline([segment_audio], output_emb=True)['embs'][0] #获取音频向量
                os.remove(segment_audio)
                # 遍历声纹库

                for name, db_emb in speaker_db.items():
                    # 计算余弦相似度
                    if isinstance(db_emb, str):
                        data_list = json.loads(db_emb)
                    else:
                        data_list = db_emb
                    arr = np.array(data_list, dtype=np.float32)
                    similarity = 1 - cosine(result_b, arr)
                    similarity = float(similarity)
                    if similarity > best_score and similarity > threshold:
                        best_score = similarity
                        best_match = name
            output.append({
                "spk": current_sentence["spk"],
                "spk_name": best_match,
                "confidence":best_score,
                "text": current_sentence["text"],
                "start": current_sentence["start"],
                "end": current_sentence["end"]
            })
            # 重新开始新句子
            current_sentence = sentence_info.copy()
            best_match = "unknown"
            best_score = 0.0

    # 保存合并完成的句子
    if identify_speakers:  # 提取说话人
        segment_audio = _extract_audio_segment(  # 提取对应的音频
                    filepath, current_sentence['start']/1000, current_sentence['end']/1000
        )
        result_b = sv_pipeline([segment_audio], output_emb=True)['embs'][0]  # 获取音频向量
        os.remove(segment_audio)
        # 遍历声纹库

        for name, db_emb in speaker_db.items():
            # 计算余弦相似度
            if isinstance(db_emb, str):
                data_list = json.loads(db_emb)
            else:
                data_list = db_emb
            arr = np.array(data_list, dtype=np.float32)
            similarity = 1 - cosine(result_b, arr)
            similarity = float(similarity)
            if similarity > best_score and similarity > threshold:
                best_score = similarity
                best_match = name
    output.append({
        "spk": current_sentence["spk"],
        "spk_name": best_match,
        "confidence":best_score,
        "text": current_sentence["text"],
        "start": current_sentence["start"],
        "end": current_sentence["end"]
    })

    return output  # 返回JSON对象列表


from scipy.spatial.distance import cosine
@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    """计算两个特征向量的余弦相似度"""
    data = request.get_json(force=True)
    emb1 = np.array(data['emb1'], dtype=np.float32)
    emb2 = np.array(data['emb2'], dtype=np.float32)
    print(1-cosine(emb1,emb2))
    return make_json_response({
                "status": "success",
                "result": float(1 - cosine(emb1, emb2))
            })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10099, debug=True)
