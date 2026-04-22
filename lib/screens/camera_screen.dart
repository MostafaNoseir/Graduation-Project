// camera_screen.dart  –  نسخة معدّلة تضيف DetectionMode.face
// التغييرات موضّحة بـ "// ✅ NEW"

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
import 'package:graduation_project/services/face_recognition_service.dart'; // ✅ NEW

// ✅ NEW – أضفنا face
enum DetectionMode { objects, color, text, currency, face }

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
  late final FaceRecognitionService _faceService; // ✅ NEW

  DetectionMode _mode = DetectionMode.objects;
  bool _busy = false;
  CameraDescription? _currentCamera;
  bool _holding = false;
  // ✅ NEW – لتتبع عملية التسجيل
  bool _enrolling = false;

  @override
  void initState() {
    super.initState();
    _init();
  }

  bool _isReady = false;

  Future<void> _init() async {
    try {
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

      _faceService = FaceRecognitionService();
      await _faceService.init();

      _isReady = true;
      setState(() {});
    } catch (e) {
      _isReady = false;
      print("Init failed: $e");
    }
  }  // ─────────────────────────────────────────────────────────────────────────
  //  SPEAK
  // ─────────────────────────────────────────────────────────────────────────
  Future<void> _speak(String text) async {
    if (!_holding) _voice.disable();
    await _tts.speak(text);
    if (_holding) _voice.startListening(_onVoiceCommand);
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  SWITCH CAMERA
  // ─────────────────────────────────────────────────────────────────────────
  Future<void> _switchCamera() async {
    if (cameras.length < 2) {
      await _speak(_tts.isArabic ? "لا توجد كاميرا أخرى متاحة" : "No other camera available");
      return;
    }

    final newDir = _currentCamera!.lensDirection == CameraLensDirection.front
        ? CameraLensDirection.back
        : CameraLensDirection.front;

    _currentCamera = cameras.firstWhere(
          (c) => c.lensDirection == newDir,
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

  // ─────────────────────────────────────────────────────────────────────────
  //  CAPTURE + PROCESS  (للأوضاع العادية)
  // ─────────────────────────────────────────────────────────────────────────
  Future<void> _captureAndProcess(
      Future<String?> Function(Uint8List bytes, String path) task,
      ) async {
    if (_busy) return;
    _busy = true;

    try {
      // await _speak(_tts.isArabic ? "لحظة واحدة" : "One moment");

      final XFile? file = await _cameraService.takePicture();
      if (file == null) {
        await _speak(_tts.isArabic ? "فشل التصوير" : "Capture failed");
        _busy = false;
        return;
      }

      final bytes  = await file.readAsBytes();
      final result = await task(bytes, file.path);
      if (result != null) await _speak(result);
    } catch (_) {
      await _speak(_tts.isArabic ? "حدث خطأ" : "An error occurred");
    }

    _busy = false;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // ✅ NEW –  تسجيل وجه جديد (Enroll)
  // ─────────────────────────────────────────────────────────────────────────
  Future<void> _startEnroll() async {
    if (_busy || _enrolling) return;

    // 1. اطلب الاسم
    final name = await _showNameDialog();
    if (name == null || name.trim().isEmpty) return;

    _enrolling = true;
    _busy      = true;

    final isAr = _tts.isArabic;

    await _speak(
      isAr
          ? "سألتقط عشر صور، ابق ثابتا أمام الكاميرا"
          : "I will capture 10 photos, stay still",
    );

    final samples = <({Uint8List bytes, String path})>[];

    for (int i = 0; i < 10; i++) {
      await Future.delayed(const Duration(milliseconds: 600));
      final file = await _cameraService.takePicture();
      if (file == null) continue;
      samples.add((bytes: await file.readAsBytes(), path: file.path));
    }

    final ok = await _faceService.enrollPerson(name.trim(), samples);

    await _speak(
      ok
          ? (isAr
          ? "تم تسجيل $name بنجاح"
          : "Enrolled $name successfully")
          : (isAr
          ? "فشل التسجيل، حاول في إضاءة أفضل"
          : "Enrollment failed, try better lighting"),
    );

    _enrolling = false;
    _busy      = false;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // ✅ NEW –  Dialog لإدخال الاسم
  // ─────────────────────────────────────────────────────────────────────────
  Future<String?> _showNameDialog() async {
    final controller = TextEditingController();

    return showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => AlertDialog(
        backgroundColor: Colors.grey[900],
        title: Text(
          _tts.isArabic ? "أدخل اسم الشخص" : "Enter person's name",
          style: const TextStyle(color: Colors.white),
        ),
        content: TextField(
          controller: controller,
          autofocus: true,
          style: const TextStyle(color: Colors.white),
          decoration: InputDecoration(
            hintText: _tts.isArabic ? "مثال: أحمد" : "e.g. Ahmed",
            hintStyle: const TextStyle(color: Colors.grey),
            enabledBorder: const UnderlineInputBorder(
              borderSide: BorderSide(color: Colors.white54),
            ),
            focusedBorder: const UnderlineInputBorder(
              borderSide: BorderSide(color: Colors.white),
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(
              _tts.isArabic ? "إلغاء" : "Cancel",
              style: const TextStyle(color: Colors.grey),
            ),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, controller.text),
            child: Text(
              _tts.isArabic ? "تأكيد" : "Confirm",
              style: const TextStyle(color: Colors.white),
            ),
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  VOICE COMMANDS
  // ─────────────────────────────────────────────────────────────────────────
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

    if (text.contains("شوف") || text.contains("ايه") ||
        text.contains("what") || text.contains("object")) {
      await _captureAndProcess((bytes, _) => _yolo.detectImage(bytes));
      return;
    }

    if (text.contains("اقرا") || text.contains("read")) {
      await _captureAndProcess(
              (_, path) => _textDetector.extractText(path, isArabic: _tts.isArabic));
      return;
    }

    if (text.contains("لون") || text.contains("color")) {
      await _captureAndProcess(
            (bytes, _) =>
            _colorDetector.detectDominantColor(bytes, isArabic: _tts.isArabic),
      );
      return;
    }

    if (text.contains("عملة") || text.contains("فلوس") ||
        text.contains("currency") || text.contains("money")) {
      await _captureAndProcess(
            (bytes, _) =>
            _currencyDetector.detectCurrency(bytes, isArabic: _tts.isArabic),
      );
      return;
    }

    // ✅ NEW – أوامر صوتية للوجه
    if (text.contains("من هذا") || text.contains("من ده") ||
        text.contains("who is") || text.contains("recognize")) {
      await _captureAndProcess(
            (bytes, path) =>
            _faceService.recognizeFace(bytes, isArabic: _tts.isArabic),
      );
      return;
    }

    if (text.contains("سجل وجه") || text.contains("أضف شخص") ||
        text.contains("enroll") || text.contains("add face")) {
      await _startEnroll();
      return;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  BUILD
  // ─────────────────────────────────────────────────────────────────────────
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
                behavior: HitTestBehavior.opaque,
                onLongPressStart: (_) {
                  _holding = true;
                  _voice.enable(_onVoiceCommand);
                },
                onLongPressEnd: (_) {
                  _holding = false;
                  _voice.disable();
                },
                onDoubleTap: _switchCamera,
              ),
            ),

            // ── زر تبديل الكاميرا ──────────────────────────────────────
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

            // ── زر اللغة ───────────────────────────────────────────────
            Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: IconButton(
                  icon: const Icon(Icons.language, color: Colors.white, size: 36),
                  onPressed: () async {
                    await _tts.toggleLanguage();
                    await _yolo.loadLabels();
                    setState(() {});
                  },
                ),
              ),
            ),

            // ── أزرار المود ────────────────────────────────────────────
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
                      'assets/icons/money.png',
                      _tts.isArabic ? "عملة" : "currency",
                    ),
                    // ✅ NEW
                    _buildModeButton(
                      DetectionMode.face,
                      'assets/icons/face.png', // ← أضف أيقونة وجه
                      _tts.isArabic ? "وجه" : "face",
                    ),
                  ],
                ),
              ),
            ),

            // ── زر الالتقاط ────────────────────────────────────────────
            Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 160),
                child: _mode == DetectionMode.face // ✅ NEW – وضع الوجه له زرّين
                    ? _buildFaceButtons()
                    : FloatingActionButton(
                  onPressed: _onCapturePressed,
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

            // ✅ NEW – مؤشر التسجيل
            if (_enrolling)
              Positioned(
                top: 80,
                left: 0,
                right: 0,
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      _tts.isArabic ? "جاري التسجيل..." : "Enrolling...",
                      style: const TextStyle(color: Colors.white, fontSize: 18),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  // ✅ NEW – زرّا وضع الوجه (تعرّف + تسجيل)
  Widget _buildFaceButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // زر التعرف
        FloatingActionButton(
          heroTag: "recognize",
          onPressed: () => _captureAndProcess(
                (bytes, path) =>
                _faceService.recognizeFace(bytes, isArabic: _tts.isArabic),
          ),
          backgroundColor: Colors.white,
          child: const Icon(Icons.face_retouching_natural, color: Colors.black, size: 32),
        ),
        const SizedBox(width: 24),
        // زر التسجيل
        FloatingActionButton(
          heroTag: "enroll",
          onPressed: _startEnroll,
          backgroundColor: Colors.blueAccent,
          child: const Icon(Icons.person_add, color: Colors.white, size: 32),
        ),
      ],
    );
  }

  Future<void> _onCapturePressed() async {
    switch (_mode) {
      case DetectionMode.objects:
        await _captureAndProcess((bytes, _) => _yolo.detectImage(bytes));
      case DetectionMode.color:
        await _captureAndProcess(
              (bytes, _) =>
              _colorDetector.detectDominantColor(bytes, isArabic: _tts.isArabic),
        );
      case DetectionMode.text:
        await _captureAndProcess(
              (_, path) => _textDetector.extractText(path, isArabic: _tts.isArabic),
        );
      case DetectionMode.currency:
        await _captureAndProcess(
              (bytes, _) =>
              _currencyDetector.detectCurrency(bytes, isArabic: _tts.isArabic),
        );
      case DetectionMode.face:
        break; // له أزراره الخاصة
    }
  }

  Widget _buildModeButton(DetectionMode mode, String imagePath, String label) {
    final isSelected = _mode == mode;
    return GestureDetector(
      onTap: () async {
        setState(() => _mode = mode);
        await _speak(_tts.isArabic ? "تم تفعيل $label" : "Mode changed to $label");
      },
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            radius: 30, // ✅ قلّلنا الحجم قليلا عشان يتسع للزر الخامس
            backgroundColor: isSelected
                ? Colors.white
                : Colors.white.withValues(alpha: 0.3),
            child: Image.asset(
              imagePath,
              width: 36,
              height: 36,
              color: isSelected ? Colors.black : Colors.white,
            ),
          ),
          const SizedBox(height: 6),
          Text(label, style: const TextStyle(color: Colors.white, fontSize: 13)),
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
    _faceService.dispose(); // ✅ NEW
    super.dispose();
  }
}