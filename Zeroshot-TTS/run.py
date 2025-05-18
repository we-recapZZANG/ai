from gradio_client import Client, handle_file
import re
import os
from pydub import AudioSegment
import tempfile

def split_into_sentences(text):
    # 문장 분리 패턴 (마침표, 물음표, 느낌표 등으로 문장 구분)
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    # 빈 문장 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def generate_tts_for_sentence(client, sentence, speaker_audio_path):
    result = client.predict(
		model_choice="Zyphra/Zonos-v0.1-transformer",
		text=sentence,
		language="ko",
		speaker_audio=handle_file(speaker_audio_path),
		e1=1,
		e2=0.05,
		e3=0.05,
		e4=0.05,
		e5=0.05,
		e6=0.05,
		e7=0.1,
		e8=0.2,
		vq_single=0.78,
		fmax=24000,
		pitch_std=45,
		speaking_rate=15,
		dnsmos_ovrl=4,
		speaker_noised=False,
		cfg_scale=2,
		top_p=0,
		top_k=0,
		min_p=0,
		linear=0.5,
		confidence=0.4,
		quadratic=0,
		seed=420,
		randomize_seed=True,
		unconditional_keys=["emotion"],
		api_name="/generate_audio"
    )
    return result[0]  # 오디오 파일 경로 반환

def merge_audio_files(audio_paths, output_path="output/hybrid_audio.wav", gap_duration_ms=500):
    if not audio_paths:
        return None

    # 0.5초(기본값) 무음 생성
    silence = AudioSegment.silent(duration=gap_duration_ms)

    # 첫 번째 오디오로 시작
    combined = AudioSegment.from_wav(audio_paths[0])

    # 이후 오디오들에는 무음 간격 추가
    for path in audio_paths[1:]:
        audio = AudioSegment.from_wav(path)
        combined += silence + audio

    # 출력 경로 지정 또는 기본 경로 사용
    if output_path is None:
        output_path = os.path.join("output", "combined_audio.wav")

    # 결합된 오디오 저장
    combined.export(output_path, format="wav")
    return output_path

# Gradio 클라이언트 초기화
client = Client("http://localhost:7860/")

# 입력 텍스트와 화자 오디오 경로
input_text = """정말 지독하게 추운 날이었어. 눈도 펑펑 내리고 어둠도 내려앉았어. 그해 마지막 저녁이었지.
한 가엾은 소녀가 모자도 안 쓰고 맨발로 거리를 걷고 있었어. 얼마나 추웠겠어. 사실 집을 나설 땐 신발을 신고 있었대. 근데 그게 엄마 신발이라 너무 컸던 거야. 그래서 길을 뛰어다니다가 그만 신발 한 짝을 잃어버렸어. 마차 두 대가 엄청 빠르게 지나가면서 신발이 어디로 갔는지 찾을 수도 없었대. 더 웃긴 건 어떤 꼬마가 남은 한 짝을 주워가면서 아기 낳으면 요람으로 쓰겠다고 했다는 거야. 결국 소녀는 맨발로 돌아다녀야 했지. 두 발은 꽁꽁 얼어붙어서 빨갛게 변했어.
소녀는 낡은 앞치마에 성냥 몇 갑을 들고 있었어. 그걸 팔려고 했지만 하루 종일 단 한 사람도 성냥을 사지 않았대. 아무도 관심이 없었던 거지.
춥고 배고파서 덜덜 떨면서 소녀는 거리를 기어가고 있었어. 정말 딱한 모습이었지.
눈송이가 소녀의 긴 머리카락 위로 떨어져 목까지 둘러싸더래. 창문에서는 불빛이 새어나오고 거위를 굽는 고소한 냄새도 났어. 오늘이 한 해의 마지막 날이라서 사람들이 다들 따뜻하게 지내는 게 부러웠던 거야.
어느 집 모퉁이에 쭈그리고 앉은 소녀는 더 이상 걸을 힘도 없었어. 게다가 집에 돌아갈 엄두도 안 났지. 성냥을 하나도 팔지 못했으니 아버지가 분명 화낼 게 뻔했거든. 집도 춥긴 마찬가지였어. 바람 막을 것도 별로 없었대.
손이 너무 시려워서 거의 움직이지도 않았어. 그래서 소녀는 성냥 하나를 켜보기로 했어. 혹시 조금이라도 따뜻해질까 싶어서 말이야.
치익! 성냥이 타오르자 정말 작은 촛불처럼 밝게 빛났어. 그 순간 소녀는 황금빛 손잡이가 달린 커다란 난로 앞에 앉아 있는 것 같은 착각이 들었대. 발도 녹이려 했는데, 이내 불꽃이 꺼지면서 환상도 사라졌지. 손엔 타버린 성냥만 남았어.
소녀는 또 성냥 하나를 켰어. 이번엔 벽이 투명해지면서 방 안이 훤히 보였대. 거기엔 맛있는 저녁 식사가 차려져 있었고, 거위가 접시에서 뛰어나와 소녀에게 다가왔어. 하지만 또 불이 꺼지자 모든 게 사라지고 차가운 벽만 남았지.
세 번째 성냥을 켰을 땐 크리스마스트리가 보였어. 정말 아름다웠대. 수천 개의 초가 빛나고, 알록달록한 그림들이 소녀를 내려다보았어. 소녀가 손을 뻗자마자 또 불이 꺼졌어. 그 빛들은 하늘로 올라가 별처럼 반짝였지.
그걸 본 소녀는 별 하나가 떨어지면 누군가 세상을 떠난다고 했던 할머니 말을 떠올렸어. 이미 세상을 떠난 할머니가 너무 그리웠던 거야.
소녀는 또 성냥을 켰어. 그러자 할머니가 밝게 웃으며 소녀 앞에 나타났어.
“할머니! 제발 절 데려가 주세요! 성냥이 꺼지면 할머니도 사라질 거잖아요!”
소녀는 할머니 곁에 있고 싶어서 남은 성냥을 모두 켜버렸어. 세상보다 더 환한 빛 속에서 할머니는 정말 아름답게 보였고, 소녀를 품에 안았지.
그리고 둘은 점점 위로, 위로 날아올랐어. 더 이상 추위도, 배고픔도, 아픔도 없는 곳으로 말이야. 하느님 곁으로.
다음 날 아침, 사람들은 모퉁이에서 차갑게 얼어붙은 채 앉아 있는 소녀를 발견했어. 입가에는 미소가 남아 있었고, 손에는 다 타버린 성냥 꾸러미가 쥐어져 있었지.
그 누구도 소녀가 얼마나 아름다운 꿈 속에서 할머니와 함께했는지는 알지 못했어."""

speaker_audio_path = 'ham.wav'

# 문장 분리
sentences = split_into_sentences(input_text)
print(f"분리된 문장 수: {len(sentences)}")

# 각 문장에 대해 TTS 생성
audio_paths = []
for i, sentence in enumerate(sentences):
    print(f"문장 {i+1} 처리 중: {sentence}")
    audio_path = generate_tts_for_sentence(client, sentence, speaker_audio_path)
    audio_paths.append(audio_path)
    print(f"문장 {i+1} 오디오 생성 완료: {audio_path}")

# 모든 오디오 파일 병합
final_audio_path = merge_audio_files(audio_paths)
print(f"최종 오디오 파일 경로: {final_audio_path}")

# 결과 반환
result = (final_audio_path, os.path.getsize(final_audio_path))
print(result)