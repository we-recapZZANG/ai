#!/bin/bash

# ... (기존 환경 변수 설정 부분) ...
cd "$(dirname "$0")" || exit

echo "INFO: Assuming you are already in the correct Python environment (e.g., Conda zonos)."

# Set environment variables
export HF_HOME="$(pwd)/huggingface"
export TORCH_HOME="$(pwd)/torch"
# export HF_ENDPOINT="https://hf-mirror.com"
export XFORMERS_FORCE_DISABLE_TRITON=1
export PHONEMIZER_ESPEAK_LIBRARY="$HOME/.local/lib/libespeak-ng.so"
export GRADIO_HOST="http://localhost:7860/" # Gradio가 사용할 주소 (main.py의 TTS_API_URL과 일치해야 함)
export TTS_API_URL="http://localhost:7860/" # main.py가 Gradio 서버를 찾을 주소

# (Optional) export CUDA_HOME if not automatically set
if [ -z "$CUDA_HOME" ] && [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
fi

echo "Starting Gradio TTS server in the background..."
python -m gradio_interface & # 백그라운드에서 Gradio 실행
GRADIO_PID=$! # Gradio 프로세스 ID 저장

echo "Gradio server_started with PID: $GRADIO_PID"
echo "Waiting for Gradio server to be ready..."
sleep 10 # Gradio 서버가 시작될 시간을 충분히 줍니다. 필요에 따라 조절하세요.

echo "Starting FastAPI server..."
# uvicorn Zonos.main:app --host 0.0.0.0 --port 8000 --reload # 개발용 (코드 변경 시 자동 재시작)
uvicorn main:app --host localhost --port 3002 # FastAPI 서버를 localhost의 3002 포트에서 실행

# 스크립트 종료 시 백그라운드 Gradio 프로세스도 종료 (선택적)
# FastAPI 서버가 Ctrl+C 등으로 종료되면 이 부분이 실행됩니다.
echo "FastAPI server stopped. Stopping Gradio server with PID: $GRADIO_PID..."
kill $GRADIO_PID
wait $GRADIO_PID 2>/dev/null # Gradio 프로세스가 완전히 종료될 때까지 대기
echo "Gradio server stopped."

echo "All servers stopped."
