import 'dart:async';
import 'package:speech_to_text/speech_to_text.dart';

class VoiceCommandService {
  final SpeechToText _speech = SpeechToText();

  bool _listening = false;
  Timer? _silenceTimer;

  Future<bool> init() async {
    try {
      return await _speech.initialize();
    } catch (_) {
      return false;
    }
  }

  void startListening(Function(String text) onResult) async {
    if (_listening) return;
    _listening = true;

    // ⏱️ تايمر الصمت (10 ثواني)
    _startSilenceTimer(onResult);

    await _speech.listen(
      localeId: "ar_EG",
      listenMode: ListenMode.confirmation,
      onResult: (result) {
        // لو في كلام → نلغي التايمر
        _resetSilenceTimer(onResult);

        if (result.finalResult) {
          onResult(result.recognizedWords.toLowerCase());
          // ميوقفش، يشتغل تاني فورًا
          _restartListening(onResult);
        }
      },
    );
  }

  // ===================== SILENCE TIMER =====================
  void _startSilenceTimer(Function(String) onResult) {
    _silenceTimer?.cancel();
    _silenceTimer = Timer(const Duration(seconds: 10), () {
      // ⛔ مفيش كلام → نعيد الاستماع مباشرة
      _restartListening(onResult);
    });
  }

  void _resetSilenceTimer(Function(String) onResult) {
    _silenceTimer?.cancel();
    _startSilenceTimer(onResult);
  }
  // ========================================================

  void _restartListening(Function(String) onResult) {
    stop();
    Future.delayed(const Duration(milliseconds: 500), () {
      startListening(onResult);
    });
  }

  void stop() {
    _silenceTimer?.cancel();
    _speech.stop();
    _listening = false;
  }

  void dispose() {
    stop();
  }
}

