import 'dart:typed_data';
import 'package:image/image.dart' as imgLib;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

class CurrencyDetector {
  late tfl.Interpreter _interpreter;

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

  static const int inputSize = 224;

  Future<void> loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset(
        'assets/models/egypt_currency_model.tflite',
      );
      print('Currency model loaded successfully');
    } catch (e) {
      print('Failed to load currency model: $e');
    }
  }

  Future<String?> detectCurrency(
      Uint8List bytes, {
        required bool isArabic,
      }) async {
    try {
      final input = _preprocess(bytes);
      if (input == null) {
        return isArabic ? "فشل في تحميل الصورة" : "Failed to load image";
      }

      // output shape: [1, 9]
      var output = List.filled(1 * _classNames.length, 0.0).reshape([1, _classNames.length]);

      _interpreter.run(input.buffer.asUint8List(), output);

      // ابحث عن أعلى قيمة
      int maxIndex = 0;
      double maxProb = output[0][0];

      for (int i = 1; i < output[0].length; i++) {
        if (output[0][i] > maxProb) {
          maxProb = output[0][i];
          maxIndex = i;
        }
      }

      if (maxProb < 0.50) {  // يمكن تعديل الثRESHOLD حسب اختبارك
        return isArabic
            ? "غير واضح، حاول تصوير الورقة بشكل أفضل"
            : "Not clear enough, try better angle / lighting";
      }

      final label = _classNames[maxIndex];

      String displayName;
      switch (label) {
        case '1':
          displayName = isArabic ? "جنيه واحد" : "One Pound";
          break;
        case '5':
          displayName = isArabic ? "خمسة جنيهات" : "Five Pounds";
          break;
        case '10':
          displayName = isArabic ? "عشرة جنيهات" : "Ten Pounds (old design)";
          break;
        case '10 (new)':
          displayName = isArabic ? "عشرة جنيهات جديدة" : "Ten Pounds";
          break;
        case '20':
          displayName = isArabic ? "عشرين جنيها" : "Twenty Pounds (old design)";
          break;
        case '20 (new)':
          displayName = isArabic ? "عشرين جنيها جديدة" : "Twenty Pounds";
          break;
        case '50':
          displayName = isArabic ? "خمسين جنيها" : "Fifty Pounds";
          break;
        case '100':
          displayName = isArabic ? "مئة جنيه" : "One Hundred Pounds";
          break;
        case '200':
          displayName = isArabic ? "مئتي جنيه" : "Two Hundred Pounds";
          break;
        default:
          displayName = label;
      }

      final prefix = isArabic ? "هذه ورقة " : "This is a ";
      final suffix  = isArabic ? " مصري" : " Egyptian Pound note";

      return "$prefix$displayName$suffix";
    } catch (e) {
      print("Currency detection error: $e");
      return isArabic
          ? "حدث خطأ أثناء التعرف على العملة"
          : "Error detecting currency";
    }
  }

  Float32List? _preprocess(Uint8List bytes) {
    final img = imgLib.decodeImage(bytes);
    if (img == null) return null;

    // Resize مع الحفاظ على النسبة + padding (letterbox)
    final scale = inputSize / img.width > inputSize / img.height
        ? inputSize / img.width
        : inputSize / img.height;

    final newWidth  = (img.width  * scale).round();
    final newHeight = (img.height * scale).round();

    var resized = imgLib.copyResize(
      img,
      width: newWidth,
      height: newHeight,
      interpolation: imgLib.Interpolation.cubic,
    );

    final canvas = imgLib.Image(width: inputSize, height: inputSize);
    imgLib.fill(canvas, color: imgLib.ColorRgb8(114, 114, 114)); // gray background مثل YOLO

    final dx = (inputSize - resized.width) ~/ 2;
    final dy = (inputSize - resized.height) ~/ 2;

    imgLib.compositeImage(canvas, resized, dstX: dx, dstY: dy);

    final input = Float32List(inputSize * inputSize * 3);
    int idx = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = canvas.getPixel(x, y);
        input[idx++] = pixel.r / 255.0;
        input[idx++] = pixel.g / 255.0;
        input[idx++] = pixel.b / 255.0;
      }
    }

    return input;
  }

  void dispose() {
    _interpreter.close();
  }
}