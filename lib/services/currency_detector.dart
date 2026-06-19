// lib/services/currency_detector.dart
//
// ✅ نسخة محسّنة — Preprocessing مطابق لكود البايثون 100%
//
// السبب الجذري للمشكلة في النسخة القديمة:
// كانت تستخدم letterbox (الحفاظ على النسبة + padding رمادي 114,114,114)
// بينما الموديل تم تدريبه/اختباره في بايثون باستخدام:
//     load_img(path, target_size=(224, 224))
// وهذه الدالة تعمل "stretch" مباشر للصورة لتصبح 224×224
// بدون الحفاظ على النسبة وبدون أي padding.
//
// النتيجة: كانت الشبكة "ترى" صورة مختلفة شكلاً عن التي تدربت عليها،
// وهذا يفسّر ضعف الدقة. تم تصحيح ذلك هنا بعمل resize مباشر مطابق تمامًا.

import 'dart:typed_data';
import 'package:image/image.dart' as imgLib;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

class CurrencyDetector {
  late tfl.Interpreter _interpreter;
  bool _isLoaded = false;

  static const List<String> _classNames = [
    '1',
    '10',
    '10 (new)',
    '100',
    '20',
    '20 (new)',
    '200',
    '5',
    '50',
  ];

  // ✅ نفس قيمة الـ threshold المستخدمة في كود البايثون
  static const double _threshold = 0.7;

  static const int inputSize = 224;

  Future<void> loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset(
        'assets/models/egypt_currency_model.tflite',
      );
      _isLoaded = true;
      print('Currency model loaded successfully');
    } catch (e) {
      _isLoaded = false;
      print('Failed to load currency model: $e');
    }
  }

  Future<String?> detectCurrency(
      Uint8List bytes, {
        required bool isArabic,
      }) async {
    try {
      if (!_isLoaded) {
        return isArabic
            ? "الموديل غير مُحمّل بعد"
            : "Model not loaded yet";
      }

      final input = _preprocess(bytes);
      if (input == null) {
        return isArabic ? "فشل في قراءة الصورة" : "Failed to read image";
      }

      // output shape: [1, 9]
      final output = List.filled(1 * _classNames.length, 0.0)
          .reshape([1, _classNames.length]);

      _interpreter.run(input.buffer.asUint8List(), output);

      // ── أعلى احتمالية (مطابق لـ np.argmax / np.max في البايثون) ──
      int maxIndex = 0;
      double maxProb = output[0][0];
      for (int i = 1; i < output[0].length; i++) {
        if (output[0][i] > maxProb) {
          maxProb = output[0][i];
          maxIndex = i;
        }
      }

      // ── Threshold check ──
      if (maxProb < _threshold) {
        return isArabic
            ? "قرّب الكاميرا أكثر من العملة وحاول مرة أخرى"
            : "Move the camera closer to the note and try again";
      }

      final label = _classNames[maxIndex];
      final displayName = _displayName(label, isArabic);
      final confidencePercent = (maxProb * 100).toStringAsFixed(0);

      print(
          "Currency prediction: $label  |  confidence: $confidencePercent%");

      final prefix = isArabic ? "هذه ورقة " : "This is a ";
      final suffix = isArabic ? " مصرية" : " Egyptian Pound note";

      return "$prefix$displayName$suffix";
    } catch (e) {
      print("Currency detection error: $e");
      return isArabic
          ? "حدث خطأ أثناء التعرف على العملة"
          : "Error detecting currency";
    }
  }

  String _displayName(String label, bool isArabic) {
    switch (label) {
      case '1':
        return isArabic ? "جنيه واحد" : "One Pound";
      case '5':
        return isArabic ? "خمسة جنيهات" : "Five Pounds";
      case '10':
        return isArabic ? "عشرة جنيهات" : "Ten Pounds";
      case '10 (new)':
        return isArabic ? "عشرة جنيهات جديدة" : "New Ten Pounds";
      case '20':
        return isArabic ? "عشرين جنيها" : "Twenty Pounds";
      case '20 (new)':
        return isArabic ? "عشرين جنيها جديدة" : "New Twenty Pounds";
      case '50':
        return isArabic ? "خمسين جنيها" : "Fifty Pounds";
      case '100':
        return isArabic ? "مئة جنيه" : "One Hundred Pounds";
      case '200':
        return isArabic ? "مئتي جنيه" : "Two Hundred Pounds";
      default:
        return label;
    }
  }

  // ══════════════════════════════════════════════════════════════
  //  Preprocessing مطابق تمامًا لـ:
  //    img = load_img(path, target_size=(224, 224))
  //    img_array = img_to_array(img) / 255.0
  //
  //  أي: resize مباشر (stretch) لـ 224×224 بدون الحفاظ على النسبة
  //  وبدون أي padding، ثم تطبيع القيم بقسمتها على 255.
  // ══════════════════════════════════════════════════════════════
  Float32List? _preprocess(Uint8List bytes) {
    final img = imgLib.decodeImage(bytes);
    if (img == null) return null;

    // ✅ resize مباشر إلى 224×224 (stretch)، نفس سلوك load_img + target_size
    final resized = imgLib.copyResize(
      img,
      width: inputSize,
      height: inputSize,
      interpolation: imgLib.Interpolation.linear,
    );

    final input = Float32List(inputSize * inputSize * 3);
    int idx = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[idx++] = pixel.r / 255.0;
        input[idx++] = pixel.g / 255.0;
        input[idx++] = pixel.b / 255.0;
      }
    }

    return input;
  }

  void dispose() {
    if (_isLoaded) {
      _interpreter.close();
    }
  }
}
