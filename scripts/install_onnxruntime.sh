#!/bin/bash
# ONNX Runtime C API 설치 스크립트

set -e

VERSION="1.16.0"
ARCH="linux-x64"
PACKAGE="onnxruntime-${ARCH}-${VERSION}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PACKAGE}.tgz"

echo "📦 Installing ONNX Runtime C API ${VERSION}..."

# 임시 디렉토리 생성
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# 다운로드
echo "⬇️  Downloading from $URL..."
wget -q --show-progress "$URL"

# 압축 해제
echo "📂 Extracting..."
tar -xzf "${PACKAGE}.tgz"

# 설치
echo "📥 Installing to /usr/local..."
sudo cp ${PACKAGE}/lib/* /usr/local/lib/
sudo cp -r ${PACKAGE}/include/* /usr/local/include/
sudo ldconfig

# 정리
cd -
rm -rf "$TMP_DIR"

echo "✅ ONNX Runtime C API installed successfully!"
echo ""
echo "Verify installation:"
echo "  ls -l /usr/local/lib/libonnxruntime*"
echo "  ls -l /usr/local/include/onnxruntime_c_api.h"
