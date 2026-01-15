import 'dart:async';
import 'package:speech_to_text/speech_to_text.dart';

class VoiceCommandService {
  final SpeechToText _speech = SpeechToText();

  bool _listening = false;
  bool _enabled = false; // ✅ هل مسموح نسمع؟
  Timer? _silenceTimer;

  Future<bool> init() async {
    try {
      return await _speech.initialize();
    } catch (_) {
      return false;
    }
  }

  // 🔊 تشغيل الاستماع (من الهولد فقط)
  void startListening(Function(String text) onResult) async {
    if (_listening || !_enabled) return;

    _listening = true;

    _startSilenceTimer(onResult);

    await _speech.listen(
      localeId: "ar_EG",
      listenMode: ListenMode.confirmation,
      onResult: (result) {
        if (!_enabled) return;

        _resetSilenceTimer(onResult);

        if (result.finalResult) {
          onResult(result.recognizedWords.toLowerCase());
          _restartListening(onResult);
        }
      },
    );
  }

  // ===================== SILENCE TIMER =====================
  void _startSilenceTimer(Function(String) onResult) {
    _silenceTimer?.cancel();
    _silenceTimer = Timer(const Duration(seconds: 3), () {
      if (_enabled) {
        _restartListening(onResult);
      }
    });
  }

  void _resetSilenceTimer(Function(String) onResult) {
    _silenceTimer?.cancel();
    _startSilenceTimer(onResult);
  }
  // ========================================================

  void _restartListening(Function(String) onResult) {
    if (!_enabled) return;

    stopInternal();
    Future.delayed(const Duration(milliseconds: 300), () {
      if (_enabled) {
        startListening(onResult);
      }
    });
  }

  // ❌ إيقاف داخلي (من غير ما يقفل enable)
  void stopInternal() {
    _silenceTimer?.cancel();
    _speech.stop();
    _listening = false;
  }

  // ✅ تستخدمها مع الهولد
  void enable(Function(String) onResult) {
    _enabled = true;
    startListening(onResult);
  }

  void disable() {
    _enabled = false;
    stopInternal();
  }

  void dispose() {
    disable();
  }
}
