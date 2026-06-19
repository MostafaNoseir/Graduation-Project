// lib/screens/camera_screen.dart

import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:graduation_project/services/camera_service.dart';
import 'package:graduation_project/services/color_detector.dart';
import 'package:graduation_project/services/text_detector.dart';
import 'package:graduation_project/services/ttf_service.dart';
import 'package:graduation_project/services/voice_command_service.dart';
import 'package:graduation_project/services/yolo_detector.dart';
import 'package:graduation_project/main.dart';
import 'package:graduation_project/services/currency_detector.dart';
import 'package:graduation_project/services/face_recognition_service.dart';
import 'package:graduation_project/screens/onboarding_screen.dart';

enum DetectionMode { objects, color, text, currency, face }

// ── مفاتيح SharedPreferences ──────────────────────────────────────────────
const _kMode      = 'last_mode';
const _kIsArabic  = 'last_is_arabic';
const _kCamFront  = 'last_cam_front';

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
  late final FaceRecognitionService _faceService;

  DetectionMode _mode = DetectionMode.objects;
  bool _busy = false;
  CameraDescription? _currentCamera;
  bool _holding = false;
  bool _enrolling = false;
  int _enrollId = 0; // ✅ يتغيّر عند كل إلغاء لإيقاف أي تسجيل قديم في الخلفية

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _init();
  }

  // ── تحميل آخر حالة محفوظة ─────────────────────────────────────────────
  Future<Map<String, dynamic>> _loadSavedState() async {
    final prefs = await SharedPreferences.getInstance();
    return {
      'mode':     prefs.getInt(_kMode) ?? 0,
      'isArabic': prefs.getBool(_kIsArabic) ?? true,
      'isFront':  prefs.getBool(_kCamFront) ?? false,
    };
  }

  // ── حفظ الحالة الحالية ────────────────────────────────────────────────
  Future<void> _saveState() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_kMode, _mode.index);
    await prefs.setBool(_kIsArabic, _tts.isArabic);
    await prefs.setBool(
        _kCamFront,
        _currentCamera?.lensDirection == CameraLensDirection.front);
  }

  Future<void> _init() async {
    try {
      // ── استرجاع الحالة المحفوظة أولاً ──────────────────────────────
      final saved = await _loadSavedState();
      final savedIsArabic = saved['isArabic'] as bool;
      final savedIsFront  = saved['isFront']  as bool;
      final savedModeIdx  = saved['mode']     as int;

      // ── TTS ──────────────────────────────────────────────────────────
      _tts = TtsService();
      // نضبط اللغة المحفوظة قبل init حتى يبدأ بها
      await _tts.initWithLanguage(savedIsArabic);

      // ── Voice ─────────────────────────────────────────────────────────
      _voice = VoiceCommandService();
      _voice.isArabic = _tts.isArabic;
      await _voice.init();

      // ── YOLO ──────────────────────────────────────────────────────────
      _yolo = YoloDetector(_tts);
      await _yolo.loadModel();

      // ── Detectors ─────────────────────────────────────────────────────
      _colorDetector    = ColorDetector();
      _textDetector     = TextDetector();
      _currencyDetector = CurrencyDetector();
      await _currencyDetector.loadModel();

      _faceService = FaceRecognitionService();
      await _faceService.init();

      // ── الكاميرا بالاتجاه المحفوظ ─────────────────────────────────
      final targetDir = savedIsFront
          ? CameraLensDirection.front
          : CameraLensDirection.back;
      _currentCamera = cameras.firstWhere(
            (c) => c.lensDirection == targetDir,
        orElse: () => cameras.first,
      );
      await _cameraService.init(_currentCamera!);

      // ── الوضع المحفوظ ─────────────────────────────────────────────
      _mode = DetectionMode.values[
      savedModeIdx.clamp(0, DetectionMode.values.length - 1)];

      setState(() {});
    } catch (e) {
      print("Init failed: $e");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  Future<void> _speak(String text) async {
    if (!_holding) _voice.disable();
    await _tts.speak(text);
    if (_holding) _voice.startListening(_onVoiceCommand);
  }

  Future<void> _speakDuringCommand(String text) async {
    await _tts.speak(text);
    if (_holding) {
      await Future.delayed(const Duration(milliseconds: 400));
      _voice.startListening(_onVoiceCommand);
    }
  }

  Future<void> _toggleLanguage() async {
    if (_enrolling) return; // 🔒 مقفل أثناء التسجيل
    await _tts.toggleLanguage();
    _voice.isArabic = _tts.isArabic;
    await _yolo.loadLabels();
    if (_holding) _voice.restartWithNewLocale(_onVoiceCommand);
    await _saveState();
    setState(() {});
  }

  Future<void> _toggleFlash() async {
    if (_enrolling) return;
    // 🔒 الكشاف غير متاح مع الكاميرا الأمامية
    if (_currentCamera?.lensDirection == CameraLensDirection.front) {
      await _speakDuringCommand(_tts.isArabic
          ? "الكشاف يعمل مع الكاميرا الخلفية فقط"
          : "Flash works with back camera only");
      return;
    }
    await _cameraService.toggleFlash();
    setState(() {});
    await _speakDuringCommand(_tts.isArabic
        ? (_cameraService.isFlashOn ? "تم تشغيل الكشاف" : "تم إيقاف الكشاف")
        : (_cameraService.isFlashOn ? "Flash on" : "Flash off"));
  }

  // ─────────────────────────────────────────────────────────────────────────
  Future<void> _switchCamera({CameraLensDirection? targetDirection}) async {
    if (_enrolling) return; // 🔒 مقفل أثناء التسجيل
    if (cameras.length < 2) {
      await _speakDuringCommand(
          _tts.isArabic ? "لا توجد كاميرا أخرى" : "No other camera");
      return;
    }

    final newDir = targetDirection ??
        (_currentCamera!.lensDirection == CameraLensDirection.front
            ? CameraLensDirection.back
            : CameraLensDirection.front);

    if (_currentCamera!.lensDirection == newDir) {
      await _speakDuringCommand(_tts.isArabic
          ? (newDir == CameraLensDirection.front
          ? "الكاميرا الأمامية مفعّلة"
          : "الكاميرا الخلفية مفعّلة")
          : (newDir == CameraLensDirection.front
          ? "Front camera already active"
          : "Back camera already active"));
      return;
    }

    _currentCamera = cameras.firstWhere(
          (c) => c.lensDirection == newDir,
      orElse: () => cameras.first,
    );
    await _cameraService.init(_currentCamera!);
    await _cameraService.setFlashOff(); // إيقاف الفلاش عند تبديل الكاميرا
    await _saveState();
    setState(() {});

    await _speakDuringCommand(_tts.isArabic
        ? (_currentCamera!.lensDirection == CameraLensDirection.front
        ? "تم التبديل للكاميرا الأمامية"
        : "تم التبديل للكاميرا الخلفية")
        : (_currentCamera!.lensDirection == CameraLensDirection.front
        ? "Switched to front camera"
        : "Switched to back camera"));
  }

  // ─────────────────────────────────────────────────────────────────────────
  Future<void> _captureAndProcess(
      Future<String?> Function(Uint8List bytes, String path) task,
      ) async {
    if (_busy) return;
    _busy = true;
    try {
      final file = await _cameraService.takePicture();
      if (file == null) {
        await _speak(_tts.isArabic ? "فشل التصوير" : "Capture failed");
        _busy = false;
        return;
      }
      final bytes = await file.readAsBytes();
      final result = await task(bytes, file.path);
      if (result != null) await _speak(result);
    } catch (_) {
      await _speak(_tts.isArabic ? "حدث خطأ" : "An error occurred");
    }
    _busy = false;
  }

  Future<void> _captureAndRecognizeFace() async {
    if (_busy || _enrolling) return; // 🔒 مقفل أثناء التسجيل
    _busy = true;
    try {
      final file = await _cameraService.takePicture();
      if (file == null) {
        await _speak(_tts.isArabic ? "فشل التصوير" : "Capture failed");
        _busy = false;
        return;
      }
      final bytes = await file.readAsBytes();
      final result =
      await _faceService.recognizeFace(bytes, isArabic: _tts.isArabic);

      if (!_holding) _voice.disable();
      if (!result.found || result.name.isEmpty) {
        await _tts.speak(result.message);
      } else {
        await _tts.speakWithName(result.prefix, result.name);
      }
      if (_holding) _voice.startListening(_onVoiceCommand);
    } catch (_) {
      await _speak(_tts.isArabic ? "حدث خطأ" : "Error");
    }
    _busy = false;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // وضعيات التسجيل — كل وضعية: نص عربي، نص إنجليزي، عدد الصور
  static const List<(String, String, int)> _enrollPoses = [
    ("ضع وجهك أمام الكاميرا مباشرةً", "Face the camera directly", 6),
    ("حرّك وجهك قليلاً نحو اليمين", "Turn your face slightly to the right", 2),
    ("حرّك وجهك قليلاً نحو اليسار", "Turn your face slightly to the left", 2),
  ];

  Future<void> _startEnroll() async {
    if (_busy || _enrolling) return;

    // ── طلب الاسم ──────────────────────────────────────────────
    await _tts.speak(_tts.isArabic
        ? "أدخل اسم الشخص المراد إضافة وجهه، اضغط في أي مكان فارغ للإلغاء"
        : "Enter the name of the person you want to add, tap any empty area to cancel");

    final name = await _showNameDialog();

    if (name == null || name.trim().isEmpty) {
      await _speak(_tts.isArabic ? "تم إلغاء الإضافة" : "Addition cancelled");
      return;
    }

    if (_faceService.enrolledNames.contains(name.trim())) {
      await _speak(_tts.isArabic
          ? "الاسم \"${name.trim()}\" موجود بالفعل"
          : "Name already exists.");
      return;
    }

    // ── بدء التسجيل — تعطيل جميع التحكمات ─────────────────────
    _enrollId++;                          // ✅ رقم جديد لكل جلسة تسجيل
    final myId = _enrollId;              // ✅ نسخة محلية — لو تغيّر معناه إلغاء
    _enrolling = true;
    _busy = true;
    _voice.disable();
    setState(() {});

    // ── جملة التعريف ────────────────────────────────────────────
    await _tts.speakAndWait(_tts.isArabic
        ? "سيتم التقاط عشر صور في وضعيات مختلفة، يرجى اتباع التعليمات. يمكنك إلغاء التسجيل في أي وقت بالضغط مرتين على الشاشة"
        : "10 photos will be captured in different poses, please follow the instructions. You can cancel at any time by double tapping the screen");
    await Future.delayed(const Duration(seconds: 2));

    final samples = <({Uint8List bytes, String path})>[];
    bool cancelled = false;

    // ✅ الفحص الموحّد: يتأكد من _enrolling وأن الجلسة لم تتغيّر
    bool isActive() => _enrolling && _enrollId == myId;

    // ── التقاط صور لكل وضعية ───────────────────────────────────
    for (final pose in _enrollPoses) {
      if (!isActive()) { cancelled = true; break; }

      final arText = pose.$1;
      final enText = pose.$2;
      final count  = pose.$3;

      await _tts.speakAndWait(_tts.isArabic ? arText : enText);
      if (!isActive()) { cancelled = true; break; }

      await Future.delayed(const Duration(seconds: 0));
      if (!isActive()) { cancelled = true; break; }

      for (int i = 0; i < count; i++) {
        if (!isActive()) { cancelled = true; break; }
        await Future.delayed(const Duration(milliseconds: 300));
        if (!isActive()) { cancelled = true; break; }
        final file = await _cameraService.takePicture();
        if (!isActive()) { cancelled = true; break; }
        if (file == null) continue;
        samples.add((bytes: await file.readAsBytes(), path: file.path));
      }
    }

    // ✅ فحص نهائي قبل الإرسال للموديل — لو الـ id تغيّر = إلغاء
    if (!isActive()) cancelled = true;

    // ── إرسال العينات للموديل ────────────────────────────────────
    if (!cancelled && samples.isNotEmpty) {
      final result = await _faceService.enrollPerson(
          name.trim(), samples,
          isArabic: _tts.isArabic);
      if (result.success && result.name.isNotEmpty) {
        if (result.isUpdate) {
          final ar = _tts.isArabic;
          final prefix  = ar ? "تم تغيير الاسم من" : "Name changed from";
          final middle  = ar ? "إلى" : "to";
          final suffix  = ar ? "بنجاح" : "successfully";
          await _tts.speakNameChange(
              prefix, result.oldName, middle, result.name, suffix);
        } else {
          final prefix = result.message;
          final suffix = _tts.isArabic ? " بنجاح" : " successfully";
          await _tts.speakWithNameAndSuffix(prefix, result.name, suffix);
        }
      } else {
        await _tts.speak(result.message);
      }
    } else if (!cancelled) {
      await _tts.speak(_tts.isArabic
          ? "فشل التسجيل، حاول مرة أخرى"
          : "Enrollment failed, please try again");
    }

    // ✅ نُعيد الحالة دايماً عند انتهاء هذه الجلسة
    _enrolling = false;
    _busy = false;
    setState(() {});
  }

  Future<void> _manageFaces() async {
    if (_busy) return;

    final names = _faceService.enrolledNames;
    if (names.isEmpty) {
      await _speak(_tts.isArabic
          ? "لا يوجد وجوه محفوظة"
          : "No saved faces");
      return;
    }

    await _tts.speak(_tts.isArabic
        ? "اختر الأشخاص التي تريد حذفهم، اضغط في أي مكان فارغ للإلغاء"
        : "Select people to delete, tap outside to cancel");

    final toDelete = await _showDeleteDialog(names);

    if (toDelete == null || toDelete.isEmpty) {
      await _speak(_tts.isArabic ? "تم إلغاء عملية الحذف" : "Deletion cancelled");
      return;
    }

    final count = await _faceService.deletePersons(toDelete);
    setState(() {});

    if (_tts.isArabic) {
      await _speak(count == 1
          ? "تم حذف شخص واحد بنجاح"
          : "تم حذف $count أشخاص بنجاح");
    } else {
      await _speak(count == 1
          ? "1 person deleted successfully"
          : "$count people deleted successfully");
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  Future<List<String>?> _showDeleteDialog(List<String> names) async {
    // ترتيب أبجدي: عربي أولاً ثم إنجليزي
    final sorted = List<String>.from(names)
      ..sort((a, b) {
        final aAr = RegExp(r'[\u0600-\u06FF]').hasMatch(a);
        final bAr = RegExp(r'[\u0600-\u06FF]').hasMatch(b);
        if (aAr && !bAr) return -1;
        if (!aAr && bAr) return 1;
        return a.compareTo(b);
      });

    return showGeneralDialog<List<String>>(
      context: context,
      barrierDismissible: true,
      barrierLabel: "dismiss",
      barrierColor: Colors.transparent,
      pageBuilder: (ctx, _, __) => _DeleteDialogContent(
        names: sorted,
        isArabic: _tts.isArabic,
        onCancel: () => Navigator.pop(ctx),
      ),
    );
  }

  Future<String?> _showNameDialog() async {
    final ctrl = TextEditingController();
    final focusNode = FocusNode();
    return showGeneralDialog<String>(
      context: context,
      barrierDismissible: true,
      barrierLabel: "dismiss",
      barrierColor: Colors.transparent,
      pageBuilder: (ctx, _, __) => _NameDialogContent(
        controller: ctrl,
        focusNode: focusNode,
        isArabic: _tts.isArabic,
        onCancel: () => Navigator.pop(ctx),
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  void _onVoiceCommand(String text) {
    if (_tts.isArabic) {
      _handleArabicCommand(text);
    } else {
      _handleEnglishCommand(text);
    }
  }

  void _handleArabicCommand(String text) async {
    if (text.contains("انجليزي") || text.contains("إنجليزي") ||
        text.contains("انجليزية") || text.contains("إنجليزية") ||
        text.contains("english") || text.contains("انجلش") ||
        text.contains("إنجلش")) {
      await _toggleLanguage(); return;
    }
    if (text.contains("أمامية") || text.contains("امامية") ||
        text.contains("أمامي") || text.contains("امامي") ||
        text.contains("قدام") || text.contains("سيلفي")) {
      await _switchCamera(targetDirection: CameraLensDirection.front); return;
    }
    if (text.contains("خلفية") || text.contains("خلفي") ||
        text.contains("الخلفية") || text.contains("ورا") ||
        text.contains("خلف")) {
      await _switchCamera(targetDirection: CameraLensDirection.back); return;
    }
    if (text.contains("شوف") || text.contains("أشياء") ||
        text.contains("اشياء") || text.contains("ايه") ||
        text.contains("إيه") || text.contains("تعرف") ||
        text.contains("كشف") || text.contains("فيه ايه")) {
      await _setMode(DetectionMode.objects);
      await _captureAndProcess((b, _) => _yolo.detectImage(b)); return;
    }
    if (text.contains("اقرا") || text.contains("اقرأ") ||
        text.contains("قراءة") || text.contains("قرا") ||
        text.contains("نص") || text.contains("نصوص")) {
      await _setMode(DetectionMode.text);
      await _captureAndProcess(
              (_, p) => _textDetector.extractText(p, isArabic: true)); return;
    }
    if (text.contains("لون") || text.contains("الوان") ||
        text.contains("ألوان") || text.contains("لونه") ||
        text.contains("لونها")) {
      await _setMode(DetectionMode.color);
      await _captureAndProcess(
              (b, _) => _colorDetector.detectDominantColor(b, isArabic: true)); return;
    }
    if (text.contains("عملة") || text.contains("فلوس") ||
        text.contains("ورقة") || text.contains("جنيه") ||
        text.contains("مال") || text.contains("نقود") ||
        text.contains("كام")) {
      await _setMode(DetectionMode.currency);
      await _captureAndProcess(
              (b, _) => _currencyDetector.detectCurrency(b, isArabic: true)); return;
    }
    if (text.contains("فلاش") || text.contains("كشاف") ||
        text.contains("ضوء") || text.contains("نور")) {
      await _toggleFlash(); return;
    }
    if (text.contains("تعليمات") || text.contains("تعليمة") ||
        text.contains("شرح") || text.contains("مساعدة")) {
      await _openOnboarding(); return;
    }
    if (text.contains("وجه") || text.contains("وش") ||
        text.contains("مين") || text.contains("من هذا") ||
        text.contains("من ده") || text.contains("من هو")) {
      await _setMode(DetectionMode.face);
      await _captureAndRecognizeFace(); return;
    }
  }

  void _handleEnglishCommand(String text) async {
    if (text.contains("arabic") || text.contains("عربي") ||
        text.contains("عربية")) {
      await _toggleLanguage(); return;
    }
    if (text.contains("front") || text.contains("selfie") ||
        text.contains("forward")) {
      await _switchCamera(targetDirection: CameraLensDirection.front); return;
    }
    if (text.contains("back") || text.contains("rear") ||
        text.contains("backward")) {
      await _switchCamera(targetDirection: CameraLensDirection.back); return;
    }
    if (text.contains("object") || text.contains("detect") ||
        text.contains("scan") || text.contains("see") ||
        text.contains("what") || text.contains("identify")) {
      await _setMode(DetectionMode.objects);
      await _captureAndProcess((b, _) => _yolo.detectImage(b)); return;
    }
    if (text.contains("read") || text.contains("text") ||
        text.contains("ocr") || text.contains("words")) {
      await _setMode(DetectionMode.text);
      await _captureAndProcess(
              (_, p) => _textDetector.extractText(p, isArabic: false)); return;
    }
    if (text.contains("color") || text.contains("colour") ||
        text.contains("colors")) {
      await _setMode(DetectionMode.color);
      await _captureAndProcess(
              (b, _) => _colorDetector.detectDominantColor(b, isArabic: false)); return;
    }
    if (text.contains("money") || text.contains("currency") ||
        text.contains("pound") || text.contains("cash") ||
        text.contains("bill")) {
      await _setMode(DetectionMode.currency);
      await _captureAndProcess(
              (b, _) => _currencyDetector.detectCurrency(b, isArabic: false)); return;
    }
    if (text.contains("flash") || text.contains("torch") ||
        text.contains("light")) {
      await _toggleFlash(); return;
    }
    if (text.contains("instruction") || text.contains("instructions") ||
        text.contains("help") || text.contains("guide")) {
      await _openOnboarding(); return;
    }
    if (text.contains("face") || text.contains("who") ||
        text.contains("recognize") || text.contains("person")) {
      await _setMode(DetectionMode.face);
      await _captureAndRecognizeFace(); return;
    }
  }

  // تغيير الوضع مع حفظ فوري
  Future<void> _setMode(DetectionMode mode) async {
    setState(() => _mode = mode);
    await _saveState();
  }

  // فتح شاشة التعليمات
  Future<void> _openOnboarding() async {
    _voice.disable();
    await _tts.stop();
    if (!mounted) return;
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const OnboardingScreen(fromApp: true)),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  BUILD
  // ─────────────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    if (_cameraService.controller == null ||
        !_cameraService.controller!.value.isInitialized) {
      return const Scaffold(
          body: Center(child: CircularProgressIndicator()));
    }

    final mq = MediaQuery.of(context);
    final sw = mq.size.width;
    final sh = mq.size.height;

    final iconBtnSize  = (sw * 0.085).clamp(26.0, 46.0);
    final modeRadius   = (sw * 0.072).clamp(20.0, 34.0);
    final modeIconSize = modeRadius * 1.15;
    final modeFontSize = (sw * 0.028).clamp(9.0, 13.0);
    final fabSize      = (sw * 0.155).clamp(48.0, 76.0);
    final fabIconSize  = fabSize * 0.52;
    final fabGap       = (sw * 0.04).clamp(8.0, 18.0);

    final modeBarBottom = sh * 0.025;
    final captureBottom = sh * 0.13;

    final previewSize = _cameraService.controller!.value.previewSize!;
    final camW = previewSize.height;
    final camH = previewSize.width;

    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [

            // ── الكاميرا ───────────────────────────────────────────────
            Positioned.fill(
              child: ClipRect(
                child: OverflowBox(
                  alignment: Alignment.center,
                  child: FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: camW,
                      height: camH,
                      child: CameraPreview(_cameraService.controller!),
                    ),
                  ),
                ),
              ),
            ),

            // ── أوامر صوتية ────────────────────────────────────────────
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
                onDoubleTap: () async {
                  if (_enrolling) {
                    // ✅ إلغاء التسجيل — تغيير الـ id يوقف أي جلسة شغّالة في الخلفية
                    _enrollId++;
                    _enrolling = false;
                    _busy = false;
                    setState(() {});
                    await _tts.stop();
                    await Future.delayed(const Duration(milliseconds: 400));
                    await _tts.speak(_tts.isArabic
                        ? "تم إلغاء التسجيل"
                        : "Enrollment cancelled");
                  } else {
                    _switchCamera();
                  }
                },
              ),
            ),

            // ── زر تبديل الكاميرا ──────────────────────────────────────
            Align(
              alignment: Alignment.topLeft,
              child: Padding(
                padding: const EdgeInsets.all(10),
                child: IconButton(
                  icon: Icon(
                    _currentCamera?.lensDirection == CameraLensDirection.front
                        ? Icons.camera_front
                        : Icons.camera_rear,
                    color: Colors.white,
                    size: iconBtnSize,
                  ),
                  onPressed: _switchCamera,
                ),
              ),
            ),

            // ── زر الفلاش ──────────────────────────────────────────────
            Align(
              alignment: Alignment.topCenter,
              child: Padding(
                padding: const EdgeInsets.all(10),
                child: IconButton(
                  icon: Icon(
                    _cameraService.isFlashOn
                        ? Icons.flashlight_on
                        : Icons.flashlight_off,
                    color: _currentCamera?.lensDirection == CameraLensDirection.front
                        ? Colors.white24
                        : (_cameraService.isFlashOn
                        ? Colors.yellow
                        : Colors.white),
                    size: iconBtnSize,
                  ),
                  onPressed: _enrolling ? null : _toggleFlash,
                ),
              ),
            ),

            // ── زر اللغة ───────────────────────────────────────────────
            Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.all(10),
                child: IconButton(
                  icon: Icon(Icons.language,
                      color: Colors.white, size: iconBtnSize),
                  onPressed: _toggleLanguage,
                ),
              ),
            ),

            // ── أزرار الأوضاع ──────────────────────────────────────────
            Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: EdgeInsets.only(bottom: modeBarBottom),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _modeBtn(DetectionMode.objects,
                        'assets/icons/recognition.png',
                        _tts.isArabic ? "أشياء" : "Objects",
                        modeRadius, modeIconSize, modeFontSize),
                    _modeBtn(DetectionMode.color,
                        'assets/icons/color.png',
                        _tts.isArabic ? "ألوان" : "Colors",
                        modeRadius, modeIconSize, modeFontSize),
                    _modeBtn(DetectionMode.text,
                        'assets/icons/text.png',
                        _tts.isArabic ? "نص" : "Text",
                        modeRadius, modeIconSize, modeFontSize),
                    _modeBtn(DetectionMode.currency,
                        'assets/icons/money.png',
                        _tts.isArabic ? "عملة" : "Money",
                        modeRadius, modeIconSize, modeFontSize),
                    _modeBtn(DetectionMode.face,
                        'assets/icons/face.png',
                        _tts.isArabic ? "وجه" : "Face",
                        modeRadius, modeIconSize, modeFontSize),
                  ],
                ),
              ),
            ),

            // ── زر الالتقاط / أزرار الوجه ─────────────────────────────
            Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: EdgeInsets.only(bottom: captureBottom),
                child: _mode == DetectionMode.face
                    ? _buildFaceButtons(fabSize, fabIconSize, fabGap)
                    : _buildCaptureButton(fabSize, fabIconSize),
              ),
            ),

            // ── مؤشر التسجيل ───────────────────────────────────────────
            if (_enrolling)
              Positioned(
                top: sh * 0.08,
                left: 0,
                right: 0,
                child: Center(
                  child: Container(
                    padding: EdgeInsets.symmetric(
                        horizontal: sw * 0.05, vertical: sh * 0.012),
                    decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(20)),
                    child: Text(
                      _tts.isArabic ? "جاري التسجيل..." : "Enrolling...",
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: (sw * 0.042).clamp(14.0, 20.0)),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  Widget _buildCaptureButton(double size, double iconSize) {
    return SizedBox(
      width: size,
      height: size,
      child: FloatingActionButton(
        onPressed: _onCapturePressed,
        backgroundColor: Colors.white,
        child: Image.asset('assets/icons/eye.png',
            width: iconSize, height: iconSize, color: Colors.black),
      ),
    );
  }

  Widget _buildFaceButtons(double size, double iconSize, double gap) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _faceFab("recognize", Icons.face_retouching_natural,
            Colors.white, Colors.black, size, iconSize, _captureAndRecognizeFace),
        SizedBox(width: gap),
        _faceFab("enroll", Icons.person_add,
            Colors.blueAccent, Colors.white, size, iconSize, _startEnroll),
        SizedBox(width: gap),
        _faceFab("reset", Icons.delete_forever,
            Colors.redAccent, Colors.white, size, iconSize, _manageFaces),
      ],
    );
  }

  Widget _faceFab(String tag, IconData icon, Color bg, Color iconColor,
      double size, double iconSize, VoidCallback onPressed) {
    return SizedBox(
      width: size,
      height: size,
      child: FloatingActionButton(
        heroTag: tag,
        onPressed: onPressed,
        backgroundColor: bg,
        child: Icon(icon, color: iconColor, size: iconSize),
      ),
    );
  }

  Widget _modeBtn(DetectionMode mode, String img, String label,
      double radius, double iconSize, double fontSize) {
    final sel = _mode == mode;
    return GestureDetector(
      onTap: () async {
        if (_enrolling) return; // 🔒 مقفل أثناء التسجيل
        await _setMode(mode);
        await _speak(_tts.isArabic ? "تم تفعيل $label" : "Mode: $label");
      },
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            radius: radius,
            backgroundColor:
            sel ? Colors.white : Colors.white.withValues(alpha: 0.3),
            child: Image.asset(img,
                width: iconSize,
                height: iconSize,
                color: sel ? Colors.black : Colors.white),
          ),
          SizedBox(height: radius * 0.15),
          Text(label,
              style: TextStyle(color: Colors.white, fontSize: fontSize)),
        ],
      ),
    );
  }

  Future<void> _onCapturePressed() async {
    if (_enrolling) return; // 🔒 مقفل أثناء التسجيل
    switch (_mode) {
      case DetectionMode.objects:
        await _captureAndProcess((b, _) => _yolo.detectImage(b));
      case DetectionMode.color:
        await _captureAndProcess((b, _) =>
            _colorDetector.detectDominantColor(b, isArabic: _tts.isArabic));
      case DetectionMode.text:
        await _captureAndProcess(
                (_, p) => _textDetector.extractText(p, isArabic: _tts.isArabic));
      case DetectionMode.currency:
        await _captureAndProcess((b, _) =>
            _currencyDetector.detectCurrency(b, isArabic: _tts.isArabic));
      case DetectionMode.face:
        break;
    }
  }

  @override
  void dispose() {
    _cameraService.dispose();
    _voice.dispose();
    _tts.dispose();
    _yolo.dispose();
    TextDetector.dispose();
    _currencyDetector.dispose();
    _faceService.dispose();
    super.dispose();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dialogs
// ═══════════════════════════════════════════════════════════════════════════

class _NameDialogContent extends StatelessWidget {
  final TextEditingController controller;
  final FocusNode focusNode;
  final bool isArabic;
  final VoidCallback onCancel;

  const _NameDialogContent({
    required this.controller,
    required this.focusNode,
    required this.isArabic,
    required this.onCancel,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(
          child: GestureDetector(
            onTap: onCancel,
            child: Container(color: Colors.black.withOpacity(0.3)),
          ),
        ),
        Center(
          child: Material(
            color: Colors.transparent,
            child: GestureDetector(
              onTap: () {},
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 32),
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: Colors.grey[900],
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      isArabic ? "أدخل اسم الشخص" : "Enter person's name",
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 16),
                    TextField(
                      controller: controller,
                      focusNode: focusNode,
                      autofocus: true,
                      style: const TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        hintText: isArabic ? "مثال: أحمد" : "e.g. Ahmed",
                        hintStyle: const TextStyle(color: Colors.grey),
                        enabledBorder: const UnderlineInputBorder(
                            borderSide: BorderSide(color: Colors.white54)),
                        focusedBorder: const UnderlineInputBorder(
                            borderSide: BorderSide(color: Colors.white)),
                      ),
                    ),
                    const SizedBox(height: 24),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        TextButton(
                          onPressed: onCancel,
                          child: Text(isArabic ? "إلغاء" : "Cancel",
                              style: const TextStyle(color: Colors.grey)),
                        ),
                        const SizedBox(width: 8),
                        TextButton(
                          onPressed: () =>
                              Navigator.pop(context, controller.text),
                          child: Text(isArabic ? "تأكيد" : "Confirm",
                              style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold)),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _DeleteDialogContent extends StatefulWidget {
  final List<String> names;
  final bool isArabic;
  final VoidCallback onCancel;

  const _DeleteDialogContent({
    required this.names,
    required this.isArabic,
    required this.onCancel,
  });

  @override
  State<_DeleteDialogContent> createState() => _DeleteDialogContentState();
}

class _DeleteDialogContentState extends State<_DeleteDialogContent> {
  final Set<String> _selected = {};

  @override
  Widget build(BuildContext context) {
    final allSelected = _selected.length == widget.names.length;

    return Stack(
      children: [
        // الضغط خارج → إلغاء
        Positioned.fill(
          child: GestureDetector(
            onTap: widget.onCancel,
            child: Container(color: Colors.black.withOpacity(0.5)),
          ),
        ),
        Center(
          child: Material(
            color: Colors.transparent,
            child: GestureDetector(
              onTap: () {}, // منع إغلاق الـ dialog عند الضغط داخله
              child: Container(
                margin: const EdgeInsets.symmetric(horizontal: 24),
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.grey[900],
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // ── العنوان ──────────────────────────────────────────
                    Text(
                      widget.isArabic ? "اختر الأشخاص للحذف" : "Select people to delete",
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 17,
                          fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 12),

                    // ── تحديد الكل / إلغاء الكل ──────────────────────
                    GestureDetector(
                      onTap: () {
                        setState(() {
                          if (allSelected) {
                            _selected.clear();
                          } else {
                            _selected.addAll(widget.names);
                          }
                        });
                      },
                      child: Row(
                        children: [
                          Icon(
                            allSelected
                                ? Icons.check_box
                                : Icons.check_box_outline_blank,
                            color: Colors.white70,
                            size: 22,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            widget.isArabic
                                ? (allSelected ? "إلغاء تحديد الكل" : "تحديد الكل")
                                : (allSelected ? "Deselect all" : "Select all"),
                            style: const TextStyle(
                                color: Colors.white70, fontSize: 14),
                          ),
                        ],
                      ),
                    ),
                    const Divider(color: Colors.white24, height: 20),

                    // ── قائمة الأسماء ────────────────────────────────
                    ConstrainedBox(
                      constraints: BoxConstraints(
                        maxHeight: MediaQuery.of(context).size.height * 0.4,
                      ),
                      child: ListView.builder(
                        shrinkWrap: true,
                        itemCount: widget.names.length,
                        itemBuilder: (_, i) {
                          final name = widget.names[i];
                          final checked = _selected.contains(name);
                          return GestureDetector(
                            onTap: () {
                              setState(() {
                                if (checked) {
                                  _selected.remove(name);
                                } else {
                                  _selected.add(name);
                                }
                              });
                            },
                            child: Padding(
                              padding: const EdgeInsets.symmetric(vertical: 8),
                              child: Row(
                                children: [
                                  Icon(
                                    checked
                                        ? Icons.check_box
                                        : Icons.check_box_outline_blank,
                                    color: checked
                                        ? Colors.redAccent
                                        : Colors.white54,
                                    size: 22,
                                  ),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: Text(
                                      name,
                                      style: TextStyle(
                                        color: checked
                                            ? Colors.white
                                            : Colors.white70,
                                        fontSize: 16,
                                        fontWeight: checked
                                            ? FontWeight.bold
                                            : FontWeight.normal,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
                    ),

                    const Divider(color: Colors.white24, height: 20),

                    // ── أزرار الإجراء ─────────────────────────────────
                    Row(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        TextButton(
                          onPressed: widget.onCancel,
                          child: Text(
                            widget.isArabic ? "إلغاء" : "Cancel",
                            style: const TextStyle(color: Colors.grey),
                          ),
                        ),
                        const SizedBox(width: 8),
                        TextButton(
                          onPressed: _selected.isEmpty
                              ? null
                              : () => Navigator.pop(
                              context, _selected.toList()),
                          child: Text(
                            widget.isArabic ? "حذف" : "Delete",
                            style: TextStyle(
                              color: _selected.isEmpty
                                  ? Colors.grey
                                  : Colors.redAccent,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}