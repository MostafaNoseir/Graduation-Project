import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:graduation_project/services/camera_service.dart';
import 'package:graduation_project/services/color_detector.dart';
import 'package:graduation_project/services/text_detector.dart';
import 'package:graduation_project/services/ttf_service.dart';
import 'package:graduation_project/services/voice_command_service.dart';
import 'package:graduation_project/services/yolo_detector.dart';
import 'package:graduation_project/main.dart';
import 'package:graduation_project/services/currency_detector.dart';
enum DetectionMode { objects, color, text, currency }

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final CameraService _cameraService = CameraService();

  late final TtsService _tts;
  late final VoiceCommandService _voice;
  late final YoloDetector _yolo;
  late final ColorDetector _colorDetector;
  late final TextDetector _textDetector;
  late final CurrencyDetector _currencyDetector;
  DetectionMode _mode = DetectionMode.objects;
  bool _busy = false;
  CameraDescription? _currentCamera;

  // حالة الـHold
  bool _holding = false;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    _tts = TtsService();
    await _tts.init();

    _voice = VoiceCommandService();
    await _voice.init();

    _yolo = YoloDetector(_tts);
    await _yolo.loadModel();

    _colorDetector = ColorDetector();
    _textDetector = TextDetector();

    _currentCamera = cameras.first;
    await _cameraService.init(_currentCamera!);

    _currencyDetector = CurrencyDetector();
    await _currencyDetector.loadModel();

    setState(() {});
    // مش بنشغل الاستماع تلقائي، كله عن طريق Hold
  }

  // دالة _speak توقف الاستماع أثناء الكلام
  Future<void> _speak(String text) async {
    if (!_holding) _voice.disable(); // وقف الاستماع لو مش Hold
    await _tts.speak(text);
    if (_holding) {
      _voice.startListening(_onVoiceCommand);
    } // رجع الاستماع لو مش Hold
  }

  Future<void> _switchCamera() async {
    if (cameras.length < 2) {
      await _speak("لا توجد كاميرا أخرى متاحة");
      return;
    }

    final newLensDirection =
        _currentCamera!.lensDirection == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;

    _currentCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == newLensDirection,
      orElse: () => cameras.first,
    );

    await _cameraService.init(_currentCamera!);
    setState(() {});

    await _speak(
      _tts.isArabic
          ? (_currentCamera!.lensDirection == CameraLensDirection.front
                ? "تم التبديل إلى الكاميرا الأمامية"
                : "تم التبديل إلى الكاميرا الخلفية")
          : (_currentCamera!.lensDirection == CameraLensDirection.front
                ? "Switched to front camera"
                : "Switched to back camera"),
    );
  }

  Future<void> _captureAndProcess(
    Future<String?> Function(Uint8List bytes, String path) task,
  ) async {
    if (_busy) return;
    _busy = true;

    try {
      await _speak(_tts.isArabic ? "لحظة واحدة" : "One moment");

      final XFile? file = await _cameraService.takePicture();
      if (file == null) {
        await _speak("فشل التصوير");
        _busy = false;
        return;
      }

      final bytes = await file.readAsBytes();
      final result = await task(bytes, file.path);

      if (result != null) await _speak(result);
    } catch (e) {
      if (_tts.isArabic) {
        await _speak("حدث خطأ");
      } else {
        await _speak("An error occurred");
      }
    }

    _busy = false;
  }

  void _onVoiceCommand(String text) async {
    if (text.contains("عربي") || text.contains("arabic")) {
      if (!_tts.isArabic) await _tts.toggleLanguage();
      await _yolo.loadLabels();
      setState(() {});
      return;
    }

    if (text.contains("english") || text.contains("انجليزي")) {
      if (_tts.isArabic) await _tts.toggleLanguage();
      await _yolo.loadLabels();
      setState(() {});
      return;
    }

    if (text.contains("شوف") ||
        text.contains("ايه") ||
        text.contains("what") ||
        text.contains("object")) {
      await _captureAndProcess((bytes, _) => _yolo.detectImage(bytes));
      return;
    }

    if (text.contains("اقرا") || text.contains("read")) {
      await _captureAndProcess((_, path) => _textDetector.extractText(path, isArabic: _tts.isArabic));
      return;
    }

    if (text.contains("لون") || text.contains("color")) {
      await _captureAndProcess(
        (bytes, _) =>
            _colorDetector.detectDominantColor(bytes, isArabic: _tts.isArabic),
      );
      return;
    }

    if (text.contains("عملة") || text.contains("فلوس") || text.contains("جنيه") ||
        text.contains("currency") || text.contains("money") || text.contains("note")) {
      await _captureAndProcess(
            (bytes, _) => _currencyDetector.detectCurrency(bytes, isArabic: _tts.isArabic),
      );
      return;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraService.controller == null ||
        !_cameraService.controller!.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
            CameraPreview(_cameraService.controller!),
            Positioned.fill(
              child: GestureDetector(
                behavior: HitTestBehavior.opaque, // 🔥 مهم جدًا
                onLongPressStart: (_) {
                  _holding = true;
                  _voice.enable(_onVoiceCommand);
                },
                onLongPressEnd: (_) {
                  _holding = false;
                  _voice.disable();
                },
                onDoubleTap: _switchCamera
              ),
            ),
            // زر تبديل الكاميرا
            Align(
              alignment: Alignment.topLeft,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: IconButton(
                  icon: Icon(
                    _currentCamera?.lensDirection == CameraLensDirection.front
                        ? Icons.camera_front
                        : Icons.camera_rear,
                    color: Colors.white,
                    size: 36,
                  ),
                  onPressed: _switchCamera,
                ),
              ),
            ),

            // زر تغيير اللغة
            Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: IconButton(
                  icon: const Icon(
                    Icons.language,
                    color: Colors.white,
                    size: 36,
                  ),
                    onPressed: () async {
                      await _tts.toggleLanguage();
                      await _yolo.loadLabels();
                      setState(() {});
                    },


                ),
              ),
            ),

            // أزرار اختيار المود
            Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 20),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _buildModeButton(
                      DetectionMode.objects,
                      'assets/icons/recognition.png',
                      _tts.isArabic ? "أشياء" : "objects",
                    ),
                    _buildModeButton(
                      DetectionMode.color,
                      'assets/icons/color.png',
                      _tts.isArabic ? "الوان" : "colors",
                    ),
                    _buildModeButton(
                      DetectionMode.text,
                      'assets/icons/text.png',
                      _tts.isArabic ? "نص" : "text",
                    ),
                    _buildModeButton(
                      DetectionMode.currency,
                      'assets/icons/money.png',  // ← أضف أيقونة مناسبة
                      _tts.isArabic ? "عملة" : "currency",
                    ),
                  ],
                ),
              ),
            ),

            // زر التقاط الصورة
            Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 160),
                child: FloatingActionButton(
                  onPressed: () async {
                    if (_mode == DetectionMode.objects) {
                      await _captureAndProcess(
                        (bytes, _) => _yolo.detectImage(bytes),

                      );
                    } else if (_mode == DetectionMode.color) {
                      await _captureAndProcess(
                        (bytes, _) => _colorDetector.detectDominantColor(
                          bytes,
                          isArabic: _tts.isArabic,
                        ),
                      );
                    } else if (_mode == DetectionMode.text) {
                      await _captureAndProcess(
                        (_, path) => _textDetector.extractText(path, isArabic: _tts.isArabic),
                      );
                    } else if (_mode == DetectionMode.currency) {
                      await _captureAndProcess(
                            (bytes, _) => _currencyDetector.detectCurrency(bytes, isArabic: _tts.isArabic),
                      );
                    }
                  },
                  backgroundColor: Colors.white,
                  child: CircleAvatar(
                    radius: 35,
                    backgroundColor: Colors.white,
                    child: Image.asset(
                      'assets/icons/eye.png',
                      width: 50,
                      height: 50,
                      color: Colors.black,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModeButton(DetectionMode mode, String imagePath, String label) {
    final isSelected = _mode == mode;

    return GestureDetector(
      onTap: () async {
        setState(() => _mode = mode);
        if (_tts.isArabic) {
          await _speak("تم تفعيل $label");
        } else {
          await _speak("Mode changed to $label");
        }
      },
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            radius: 35,
            backgroundColor: isSelected
                ? Colors.white
                : Colors.white.withValues(alpha: 0.3),
            child: Image.asset(
              imagePath,
              width: 40,
              height: 40,
              color: isSelected ? Colors.black : Colors.white,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            label,
            style: const TextStyle(color: Colors.white, fontSize: 16),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraService.dispose();
    _voice.dispose();
    _tts.dispose();
    _yolo.dispose();
    TextDetector.dispose();
    _currencyDetector.dispose();
    super.dispose();
  }
}
