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

  /// ✅ نطق وانتظار حتى الانتهاء الفعلي من النطق
  Future<void> speakAndWait(String text) async {
    await _tts.stop();
    final completer = Completer<void>();
    _tts.setCompletionHandler(() {
      if (!completer.isCompleted) completer.complete();
    });
    await _tts.speak(text);
    await completer.future.timeout(
      const Duration(seconds: 30),
      onTimeout: () {},
    );
    _tts.setCompletionHandler(() {});
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

  // ✅ نطق: prefix (بلغة الـ UI) + name (بلغته) + suffix (بلغة الـ UI)
  // مثال: "تم تسجيل" + "Ahmed" + " بنجاح"
  Future<void> speakWithNameAndSuffix(
      String prefix, String name, String suffix) async {
    await _tts.stop();

    final nameIsArabic = _containsArabic(name);

    if (nameIsArabic == _arabic) {
      // نفس اللغة — انطق كل شيء دفعة واحدة
      await _tts.speak("$prefix $name$suffix");
    } else {
      // لغتان مختلفتان — انطق prefix ثم name ثم suffix
      Future<void> waitCompletion() async {
        final c = Completer<void>();
        _tts.setCompletionHandler(() { if (!c.isCompleted) c.complete(); });
        await c.future.timeout(const Duration(seconds: 8), onTimeout: () {});
        _tts.setCompletionHandler(() {});
      }

      // prefix بلغة الـ UI
      await _tts.speak(prefix);
      await waitCompletion();

      // name بلغته الصحيحة
      await _tts.setLanguage(nameIsArabic ? "ar-EG" : "en-US");
      await _tts.speak(name);
      await waitCompletion();

      // suffix بلغة الـ UI
      await _setLanguage();
      await _tts.speak(suffix.trim());
      await waitCompletion();

      // أعد اللغة الأصلية
      await _setLanguage();
    }
  }

  // ✅ نطق تغيير الاسم: prefix + oldName + middle + newName + suffix
  // كل اسم يُنطق بلغته الصحيحة
  Future<void> speakNameChange(String prefix, String oldName,
      String middle, String newName, String suffix) async {
    await _tts.stop();

    Future<void> waitDone() async {
      final c = Completer<void>();
      _tts.setCompletionHandler(() { if (!c.isCompleted) c.complete(); });
      await c.future.timeout(const Duration(seconds: 8), onTimeout: () {});
      _tts.setCompletionHandler(() {});
    }

    final oldIsAr = _containsArabic(oldName);
    final newIsAr = _containsArabic(newName);

    // prefix بلغة الـ UI
    await _tts.speak(prefix);
    await waitDone();

    // oldName بلغته
    await _tts.setLanguage(oldIsAr ? "ar-EG" : "en-US");
    await _tts.speak(oldName);
    await waitDone();

    // middle بلغة الـ UI
    await _setLanguage();
    await _tts.speak(middle);
    await waitDone();

    // newName بلغته
    await _tts.setLanguage(newIsAr ? "ar-EG" : "en-US");
    await _tts.speak(newName);
    await waitDone();

    // suffix بلغة الـ UI
    await _setLanguage();
    await _tts.speak(suffix);
    await waitDone();
  }

  void dispose() {
    _tts.stop();
  }
}