import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:graduation_project/core/services/camera_service.dart';
import 'package:graduation_project/core/services/color_detector.dart';
import 'package:graduation_project/core/services/text_detector.dart';
import 'package:graduation_project/core/services/ttf_service.dart';
import 'package:graduation_project/core/services/voice_command_service.dart';
import 'package:graduation_project/core/services/yolo_detector.dart';
import 'package:graduation_project/main.dart';
enum DetectionMode { objects, color, text }

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

  DetectionMode _mode = DetectionMode.objects;
  bool _busy = false;
  CameraDescription? _currentCamera;

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

    setState(() {});
    _voice.startListening(_onVoiceCommand);
  }

  Future<void> _switchCamera() async {
    if (cameras.length < 2) {
      await _tts.speak("لا توجد كاميرا أخرى متاحة");
      return;
    }

    final newLensDirection = _currentCamera!.lensDirection == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;

    _currentCamera = cameras.firstWhere(
          (camera) => camera.lensDirection == newLensDirection,
      orElse: () => cameras.first,
    );

    await _cameraService.init(_currentCamera!);
    setState(() {});

    await _tts.speak(
      _currentCamera!.lensDirection == CameraLensDirection.front
          ? "تم التبديل إلى الكاميرا الأمامية"
          : "تم التبديل إلى الكاميرا الخلفية",
    );
  }

  Future<void> _captureAndProcess(
      Future<String?> Function(Uint8List bytes, String path) task,
      ) async {
    if (_busy) return;
    _busy = true;

    try {
      await _tts.speak(_tts.isArabic ? "ثانية واحدة" : "One moment");

      final XFile? file = await _cameraService.takePicture();
      if (file == null) {
        await _tts.speak("فشل التصوير");
        _busy = false;
        return;
      }

      final bytes = await file.readAsBytes();
      final result = await task(bytes, file.path);

      if (result != null) await _tts.speak(result);
    } catch (e) {
      await _tts.speak("حدث خطأ");
    }

    _busy = false;
  }

  void _onVoiceCommand(String text) async {
    if (text.contains("عربي") || text.contains("arabic")) {
      if (!_tts.isArabic) await _tts.toggleLanguage();
      await _yolo.loadLabels();
      return;
    }

    if (text.contains("english") || text.contains("انجليزي")) {
      if (_tts.isArabic) await _tts.toggleLanguage();
      await _yolo.loadLabels();
      return;
    }

    if (text.contains("شوف") || text.contains("ايه") || text.contains("what")) {
      await _captureAndProcess((bytes, _) => _yolo.detectImage(bytes));
      return;
    }

    if (text.contains("اقرا") || text.contains("read")) {
      await _captureAndProcess((_, path) => _textDetector.extractText(path));
      return;
    }

    if (text.contains("لون") || text.contains("color")) {
      await _captureAndProcess((bytes, _) => _colorDetector.detectDominantColor(bytes));
      return;
    }

    await _tts.speak("لم أفهم الأمر");
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraService.controller == null ||
        !_cameraService.controller!.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          CameraPreview(_cameraService.controller!),

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
                icon: const Icon(Icons.language, color: Colors.white, size: 36),
                onPressed: _tts.toggleLanguage,
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
                  _buildModeButton(DetectionMode.objects, Icons.visibility, "أشياء"),
                  _buildModeButton(DetectionMode.color, Icons.color_lens, "لون"),
                  _buildModeButton(DetectionMode.text, Icons.text_fields, "نص"),
                ],
              ),
            ),
          ),

          // زر المايك
          // Align(
          //   alignment: Alignment.bottomCenter,
          //   child: Padding(
          //     padding: const EdgeInsets.only(bottom: 90),
          //     child: FloatingActionButton.extended(
          //       icon: const Icon(Icons.mic),
          //       label: const Text("تحدث"),
          //       onPressed: () => _voice.startListening(_onVoiceCommand),
          //     ),
          //   ),
          // ),

          // زر التقاط الصورة
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: const EdgeInsets.only(bottom: 160),
              child: FloatingActionButton(
                onPressed: () async {
                  if (_mode == DetectionMode.objects) {
                    await _captureAndProcess((bytes, _) => _yolo.detectImage(bytes));
                  } else if (_mode == DetectionMode.color) {
                    await _captureAndProcess((bytes, _) => _colorDetector.detectDominantColor(bytes));
                  } else if (_mode == DetectionMode.text) {
                    await _captureAndProcess((_, path) => _textDetector.extractText(path));
                  }
                },
                backgroundColor: Colors.white,
                child: const Icon(Icons.camera_alt, size: 40, color: Colors.black),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildModeButton(DetectionMode mode, IconData icon, String label) {
    final isSelected = _mode == mode;
    return GestureDetector(
      onTap: () async {
        setState(() => _mode = mode);
        await _tts.speak("تم تفعيل $label"); // هنا بيتكلم عند الضغط
      },
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            radius: 35,
            backgroundColor: isSelected ? Colors.white : Colors.white.withValues(alpha: 0.3),
            child: Icon(icon, size: 40, color: isSelected ? Colors.black : Colors.white),
          ),
          const SizedBox(height: 8),
          Text(label, style: const TextStyle(color: Colors.white, fontSize: 16)),
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
    super.dispose();
  }
}
