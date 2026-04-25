import 'dart:async';
import 'package:speech_to_text/speech_to_text.dart';

class VoiceCommandService {
  final SpeechToText _speech = SpeechToText();

  bool _listening = false;
  bool _enabled = false;
  Timer? _silenceTimer;

  // ✅ يتحكم في localeId — يُحدَّث من camera_screen عند تغيير اللغة
  bool isArabic = true;

  String get _localeId => isArabic ? "ar_EG" : "en_US";

  Future<bool> init() async {
    try {
      return await _speech.initialize();
    } catch (_) {
      return false;
    }
  }

  void startListening(Function(String) onResult) async {
    if (_listening || !_enabled) return;
    _listening = true;
    _startSilenceTimer(onResult);

    await _speech.listen(
      localeId: _localeId, // ✅ يقرأ القيمة الحالية في كل مرة
      listenMode: ListenMode.dictation,
      onResult: (result) {
        if (!_enabled) return;
        _resetSilenceTimer(onResult);
        if (result.finalResult) {
          final words = result.recognizedWords.trim().toLowerCase();
          if (words.isNotEmpty) {
            print("🎤 [$_localeId] $words");
            onResult(words);
          }
          _restartListening(onResult);
        }
      },
    );
  }

  void _startSilenceTimer(Function(String) onResult) {
    _silenceTimer?.cancel();
    _silenceTimer = Timer(const Duration(seconds: 4), () {
      if (_enabled) _restartListening(onResult);
    });
  }

  void _resetSilenceTimer(Function(String) onResult) {
    _silenceTimer?.cancel();
    _startSilenceTimer(onResult);
  }

  void _restartListening(Function(String) onResult) {
    if (!_enabled) return;
    stopInternal();
    Future.delayed(const Duration(milliseconds: 300), () {
      if (_enabled) startListening(onResult);
    });
  }

  void stopInternal() {
    _silenceTimer?.cancel();
    _speech.stop();
    _listening = false;
  }

  void enable(Function(String) onResult) {
    _enabled = true;
    startListening(onResult);
  }

  void disable() {
    _enabled = false;
    stopInternal();
  }

  // ✅ يُستدعى من camera_screen بعد تغيير اللغة
  // يوقف الجلسة الحالية ويعيد تشغيلها بالـ locale الجديد
  void restartWithNewLocale(Function(String) onResult) {
    if (!_enabled) return;
    stopInternal();
    Future.delayed(const Duration(milliseconds: 300), () {
      if (_enabled) startListening(onResult);
    });
  }

  void dispose() => disable();
}
