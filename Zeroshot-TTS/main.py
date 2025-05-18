import fastapi
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import tempfile
import re
from gradio_client import Client, handle_file
from pydub import AudioSegment
import logging # 로깅 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- run.py의 함수들을 여기에 가져오거나 별도 모듈로 분리하여 import ---
# (가독성을 위해 run.py의 함수들을 약간 수정하거나 그대로 가져올 수 있습니다)

def split_into_sentences(text: str):
    pattern = r'(?<=[.!?])\s+'
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# Gradio TTS API 서버 URL (환경 변수 등에서 설정 가능)
TTS_API_URL = os.getenv("TTS_API_URL", "http://localhost:7860/") # run.sh에서 설정한 Gradio 주소

def generate_tts_for_sentence(client: Client, sentence: str, speaker_audio_path: str):
    try:
        # run.py와 동일한 파라미터 사용
        result = client.predict(
            model_choice="Zyphra/Zonos-v0.1-transformer",
            text=sentence,
            language="ko",
            speaker_audio=handle_file(speaker_audio_path), # 임시 저장된 화자 오디오 파일 경로
            e1=1, e2=0.05, e3=0.05, e4=0.05, e5=0.05, e6=0.05, e7=0.1, e8=0.2,
            vq_single=0.78, fmax=24000, pitch_std=45, speaking_rate=15,
            dnsmos_ovrl=4, speaker_noised=False, cfg_scale=2, top_p=0, top_k=0,
            min_p=0, linear=0.5, confidence=0.4, quadratic=0,
            seed=420, randomize_seed=True, # run.py에서 randomize_seed=True로 사용하셨으므로 반영
            unconditional_keys=["emotion"],
            api_name="/generate_audio"
        )
        return result[0]  # 생성된 오디오 파일 경로 반환
    except Exception as e:
        logger.error(f"Error in generate_tts_for_sentence for '{sentence}': {e}")
        raise

def merge_audio_files(audio_paths: list, base_output_path: str, gap_duration_ms: int = 500):
    if not audio_paths:
        return None
    
    silence = AudioSegment.silent(duration=gap_duration_ms)
    
    try:
        combined = AudioSegment.from_wav(audio_paths[0])
        for path in audio_paths[1:]:
            audio = AudioSegment.from_wav(path)
            combined += silence + audio # 수정된 로직 반영
        
        # 디렉토리 존재 확인 및 생성
        output_dir = os.path.dirname(base_output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        combined.export(base_output_path, format="wav")
        return base_output_path
    except Exception as e:
        logger.error(f"Error merging audio files: {e}")
        raise
# --- 함수 정의 끝 ---

app = FastAPI()

@app.post("/process-audio-text/")
async def process_audio_text(
    text: str = Form(...), 
    speaker_wav: UploadFile = File(...)
):
    # 각 요청 처리를 위한 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    uploaded_speaker_path = os.path.join(temp_dir, speaker_wav.filename or "speaker.wav")
    final_output_filename = "final_story_audio.wav"
    final_output_path = os.path.join(temp_dir, final_output_filename)

    try:
        # 1. 업로드된 화자 WAV 파일을 임시 디렉토리에 저장
        with open(uploaded_speaker_path, "wb") as buffer:
            shutil.copyfileobj(speaker_wav.file, buffer)
        logger.info(f"Speaker WAV saved to: {uploaded_speaker_path}")

        # 2. Gradio 클라이언트 초기화
        #    (실제 프로덕션에서는 애플리케이션 시작 시 초기화하거나,
        #     FastAPI의 dependency injection을 사용하는 것이 더 효율적일 수 있습니다.)
        gradio_tts_client = Client(TTS_API_URL)
        logger.info(f"Connected to Gradio client at {TTS_API_URL}")

        # 3. 텍스트를 문장으로 분리
        sentences = split_into_sentences(text)
        if not sentences:
            logger.warning("No sentences found in input text.")
            raise HTTPException(status_code=400, detail="No sentences found in input text.")
        logger.info(f"Split into {len(sentences)} sentences.")

        # 4. 각 문장에 대해 TTS 생성
        sentence_audio_paths = []
        for i, sentence_text in enumerate(sentences):
            logger.info(f"Generating TTS for sentence {i+1}/{len(sentences)}: '{sentence_text[:30]}...'")
            # generate_tts_for_sentence 함수가 생성된 오디오 파일의 경로를 반환한다고 가정
            generated_sentence_audio_path_from_gradio = generate_tts_for_sentence(
                gradio_tts_client,
                sentence_text,
                uploaded_speaker_path
            )
            
            # Gradio 클라이언트가 생성한 임시 파일을 우리 임시 디렉토리로 복사 (관리 용이)
            # 파일 이름에 순서를 부여하여 고유하게 만듭니다.
            copied_sentence_path = os.path.join(temp_dir, f"sentence_{i+1}.wav")
            shutil.copy(generated_sentence_audio_path_from_gradio, copied_sentence_path)
            sentence_audio_paths.append(copied_sentence_path)
            logger.info(f"Sentence {i+1} audio saved to: {copied_sentence_path}")


        if not sentence_audio_paths:
            logger.error("TTS generation failed for all sentences.")
            raise HTTPException(status_code=500, detail="TTS generation failed for all sentences.")

        # 5. 생성된 오디오 파일들을 병합 (0.3초 무음 간격 포함)
        logger.info("Merging sentence audio files...")
        merged_audio_file_path = merge_audio_files(
            sentence_audio_paths,
            base_output_path=final_output_path, # 요청별 임시 디렉토리 내 최종 경로
            gap_duration_ms=300
        )

        if not merged_audio_file_path or not os.path.exists(merged_audio_file_path):
            logger.error("Failed to merge audio files or merged file not found.")
            raise HTTPException(status_code=500, detail="Failed to merge audio files.")
        logger.info(f"Final merged audio saved to: {merged_audio_file_path}")

        # 6. 최종 생성된 오디오 파일을 응답으로 반환
        #    FileResponse는 파일을 스트리밍하며, 작업 완료 후 파일이 삭제되도록 처리할 수 있습니다.
        #    여기서는 finally 블록에서 전체 임시 디렉토리를 삭제합니다.
        return FileResponse(
            path=merged_audio_file_path, 
            media_type="audio/wav", 
            filename=final_output_filename # 클라이언트에게 전달될 파일 이름
        )

    except HTTPException as http_exc:
        logger.error(f"HTTPException: {http_exc.detail}")
        raise http_exc # FastAPI가 처리하도록 다시 발생
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") # 스택 트레이스 포함 로깅
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        # 7. 요청 처리 후 임시 디렉토리와 그 내용 삭제
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")

# 이 파일을 main.py로 저장하고 다음 명령어로 실행:
# uvicorn main:app --reload
#
# 그런 다음 HTTP POST 요청을 /process-audio-text/ 엔드포인트로 보냅니다.
# 요청 본문에는 'text' (form data)와 'speaker_wav' (file upload)가 포함되어야 합니다.