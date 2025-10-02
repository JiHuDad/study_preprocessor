# C 추론 엔진 Makefile
# 하이브리드 로그 이상탐지 시스템

# 컴파일러 및 플래그
CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 -fPIC
DEBUG_FLAGS = -g -DDEBUG -O0
INCLUDES = -I./include -I/usr/local/include
LIBS = -lonnxruntime -lm -lpthread

# 디렉토리
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# 소스 파일
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/inference_engine

# 라이브러리 타겟
LIBNAME = libinference_engine
STATIC_LIB = $(BINDIR)/$(LIBNAME).a
SHARED_LIB = $(BINDIR)/$(LIBNAME).so

# 기본 타겟
.PHONY: all clean debug install test help

all: $(TARGET)

# 실행 파일 빌드
$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(OBJECTS) -o $@ $(LIBS)
	@echo "✅ Built executable: $@"

# 정적 라이브러리 빌드
static: $(STATIC_LIB)

$(STATIC_LIB): $(filter-out $(OBJDIR)/main.o,$(OBJECTS)) | $(BINDIR)
	ar rcs $@ $^
	@echo "✅ Built static library: $@"

# 동적 라이브러리 빌드
shared: $(SHARED_LIB)

$(SHARED_LIB): $(filter-out $(OBJDIR)/main.o,$(OBJECTS)) | $(BINDIR)
	$(CC) -shared -o $@ $^ $(LIBS)
	@echo "✅ Built shared library: $@"

# 오브젝트 파일 빌드
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 디렉토리 생성
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

# 디버그 빌드
debug: CFLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET)
	@echo "✅ Built debug version"

# 정리
clean:
	rm -rf $(OBJDIR) $(BINDIR)
	@echo "🧹 Cleaned build files"

# 설치 (시스템 전역)
install: $(TARGET) $(STATIC_LIB) $(SHARED_LIB)
	@echo "📦 Installing inference engine..."
	sudo cp $(TARGET) /usr/local/bin/
	sudo cp $(STATIC_LIB) /usr/local/lib/
	sudo cp $(SHARED_LIB) /usr/local/lib/
	sudo cp $(INCDIR)/*.h /usr/local/include/
	sudo ldconfig
	@echo "✅ Installation complete"

# 제거
uninstall:
	@echo "🗑️ Uninstalling inference engine..."
	sudo rm -f /usr/local/bin/inference_engine
	sudo rm -f /usr/local/lib/$(LIBNAME).*
	sudo rm -f /usr/local/include/inference_engine.h
	sudo ldconfig
	@echo "✅ Uninstallation complete"

# 테스트 빌드 및 실행
test: $(TARGET)
	@echo "🧪 Running basic tests..."
	@if [ ! -f "test_models/vocab.json" ]; then \
		echo "⚠️ Test models not found. Creating dummy files..."; \
		mkdir -p test_models; \
		echo '{"0": "INFO User <*> logged in", "1": "ERROR Authentication failed"}' > test_models/vocab.json; \
	fi
	@echo "Running test mode..."
	./$(TARGET) -d test_models/deeplog.onnx -v test_models/vocab.json -t || echo "⚠️ Test requires actual ONNX models"

# 의존성 확인
check-deps:
	@echo "🔍 Checking dependencies..."
	@pkg-config --exists onnxruntime 2>/dev/null && echo "✅ ONNX Runtime found" || echo "❌ ONNX Runtime not found"
	@which gcc >/dev/null && echo "✅ GCC found" || echo "❌ GCC not found"
	@echo "💡 To install ONNX Runtime:"
	@echo "   wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz"
	@echo "   tar -xzf onnxruntime-linux-x64-1.16.0.tgz"
	@echo "   sudo cp onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/"
	@echo "   sudo cp -r onnxruntime-linux-x64-1.16.0/include/* /usr/local/include/"
	@echo "   sudo ldconfig"

# 벤치마크
benchmark: $(TARGET)
	@echo "⚡ Running performance benchmark..."
	@echo "Generating test data..."
	@for i in {1..10000}; do \
		echo "2024-01-01 10:00:$$i INFO Test log line $$i"; \
	done > test_data.log
	@echo "Running benchmark..."
	@time ./$(TARGET) -d test_models/deeplog.onnx -v test_models/vocab.json -i test_data.log -o /dev/null || echo "⚠️ Benchmark requires actual models"
	@rm -f test_data.log

# 메모리 검사 (valgrind 필요)
memcheck: debug
	@echo "🔍 Running memory check..."
	@which valgrind >/dev/null || (echo "❌ valgrind not found. Install with: sudo apt-get install valgrind"; exit 1)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET) -d test_models/deeplog.onnx -v test_models/vocab.json -t

# 코드 포맷팅 (clang-format 필요)
format:
	@echo "🎨 Formatting code..."
	@which clang-format >/dev/null || (echo "❌ clang-format not found"; exit 1)
	find $(SRCDIR) $(INCDIR) -name "*.c" -o -name "*.h" | xargs clang-format -i
	@echo "✅ Code formatted"

# 정적 분석 (cppcheck 필요)
analyze:
	@echo "🔬 Running static analysis..."
	@which cppcheck >/dev/null || (echo "❌ cppcheck not found. Install with: sudo apt-get install cppcheck"; exit 1)
	cppcheck --enable=all --std=c99 $(SRCDIR)

# 도움말
help:
	@echo "🛠️ C Inference Engine Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build executable (default)"
	@echo "  static     - Build static library"
	@echo "  shared     - Build shared library"
	@echo "  debug      - Build debug version"
	@echo "  clean      - Remove build files"
	@echo "  install    - Install system-wide"
	@echo "  uninstall  - Remove system installation"
	@echo "  test       - Run basic tests"
	@echo "  benchmark  - Run performance benchmark"
	@echo "  memcheck   - Run memory leak check"
	@echo "  format     - Format source code"
	@echo "  analyze    - Run static analysis"
	@echo "  check-deps - Check dependencies"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                    # Build executable"
	@echo "  make debug              # Build with debug info"
	@echo "  make static shared      # Build libraries"
	@echo "  make test               # Run tests"
	@echo "  make install            # Install system-wide"
	@echo ""
	@echo "Dependencies:"
	@echo "  - GCC compiler"
	@echo "  - ONNX Runtime C API"
	@echo "  - pthread library"
	@echo "  - math library"

# 의존성 정보
$(OBJDIR)/onnx_engine.o: $(INCDIR)/inference_engine.h
$(OBJDIR)/log_parser.o: $(INCDIR)/inference_engine.h
$(OBJDIR)/anomaly_detector.o: $(INCDIR)/inference_engine.h
$(OBJDIR)/main.o: $(INCDIR)/inference_engine.h
