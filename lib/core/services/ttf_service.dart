import 'package:flutter_tts/flutter_tts.dart';

class TtsService {
  final FlutterTts _tts = FlutterTts();
  bool _arabic = false;

  // getter علشان يستخدم في كلاس تاني
  bool get isArabic => _arabic;

  Future<void> init() async {
    await _tts.setSpeechRate(0.45);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
    await _setLanguage();
  }

  Future<void> _setLanguage() async {
    if (_arabic) {
      await _tts.setLanguage("ar-EG");
    } else {
      await _tts.setLanguage("en-US");
    }
  }

  Future<void> speak(String text) async {
    await _tts.stop();
    await _tts.speak(text);
  }

  Future<void> toggleLanguage() async {
    _arabic = !_arabic;
    await _setLanguage();
    await speak(_arabic ? "تم تغيير اللغة للعربية" : "Language changed to English");
  }

  void dispose() {
    _tts.stop();
  }
}
