import 'dart:async';
import 'package:flutter_tts/flutter_tts.dart';

class TtsService {
  final FlutterTts _tts = FlutterTts();

  bool _arabic = true;
  bool get isArabic => _arabic;

  // init عادي — يبدأ بالعربية
  Future<void> init() async {
    _arabic = true;
    await _setup();
  }

  // ✅ init مع لغة محفوظة مسبقاً
  Future<void> initWithLanguage(bool isArabic) async {
    _arabic = isArabic;
    await _setup();
  }

  Future<void> _setup() async {
    await _tts.setSpeechRate(0.45);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
    await _setLanguage();
  }

  Future<void> _setLanguage() async {
    await _tts.setLanguage(_arabic ? "ar-EG" : "en-US");
  }

  Future<void> speak(String text) async {
    await _tts.stop();
    await _tts.speak(text);
  }

  Future<void> speakWithName(String prefix, String name) async {
    await _tts.stop();

    final nameIsArabic = _containsArabic(name);

    if (nameIsArabic == _arabic) {
      await _tts.speak("$prefix $name");
    } else {
      final completer = Completer<void>();
      _tts.setCompletionHandler(() {
        if (!completer.isCompleted) completer.complete();
      });
      await _tts.speak(prefix);
      await completer.future
          .timeout(const Duration(seconds: 5), onTimeout: () {});
      _tts.setCompletionHandler(() {});

      await _tts.setLanguage(nameIsArabic ? "ar-EG" : "en-US");
      await _tts.speak(name);

      final completer2 = Completer<void>();
      _tts.setCompletionHandler(() {
        if (!completer2.isCompleted) completer2.complete();
      });
      await completer2.future
          .timeout(const Duration(seconds: 5), onTimeout: () {});
      _tts.setCompletionHandler(() {});

      await _setLanguage();
    }
  }

  Future<void> toggleLanguage() async {
    _arabic = !_arabic;
    await _setLanguage();
    await speak(_arabic ? "تم تغيير اللغة للعربية" : "Language changed to English");
  }

  bool _containsArabic(String text) =>
      RegExp(r'[\u0600-\u06FF]').hasMatch(text);

  // إيقاف النطق فوراً بدون إغلاق الـ service
  Future<void> stop() async {
    await _tts.stop();
  }

  void dispose() {
    _tts.stop();
  }
}