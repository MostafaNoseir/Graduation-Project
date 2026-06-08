// lib/text_detector.dart
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

class TextDetector {
  static final _textRecognizer = TextRecognizer();

  Future<String?> extractText(
      String imagePath, {
        required bool isArabic,
      }) async {
    try {
      final inputImage = InputImage.fromFilePath(imagePath);
      final recognizedText = await _textRecognizer.processImage(inputImage);

      // لا يوجد نص
      if (recognizedText.blocks.isEmpty) {
        return isArabic
            ? "لم يتم اكتشاف أي نص في الصورة"
            : "No text was detected in the image";
      }

      // ترتيب الكتل من أعلى لأسفل
      final sortedBlocks = recognizedText.blocks.toList()
        ..sort((a, b) => a.boundingBox.top.compareTo(b.boundingBox.top));

      final List<String> lines = [];

      for (var block in sortedBlocks) {
        // ترتيب السطور داخل الكتلة
        final sortedLines = block.lines.toList()
          ..sort((a, b) => a.boundingBox.top.compareTo(b.boundingBox.top));

        for (var line in sortedLines) {
          String lineText = line.text.trim();
          lineText = lineText.replaceAll(RegExp(r'\s+'), ' ');

          if (lineText.isNotEmpty) {
            lines.add(lineText);
          }
        }

        // سطر فاضي بين الفقرات
        if (lines.isNotEmpty) {
          lines.add("");
        }
      }

      String fullText = lines.join('\n').trim();

      fullText = fullText
          .replaceAll(RegExp(r'\n{3,}'), '\n\n')
          .trim();

      if (fullText.isEmpty) {
        return isArabic
            ? "تم اكتشاف نص لكن غير قابل للقراءة"
            : "Text was detected but could not be read";
      }

      // نص مناسب للـ TTS
      return isArabic
          ? "النص المكتشف في الصورة:\n\n$fullText"
          : "Detected text in the image:\n\n$fullText";

    } catch (e) {
      print("Text recognition error: $e");
      return isArabic
          ? "حدث خطأ أثناء قراءة النص"
          : "An error occurred while reading the text";
    }
  }

  static void dispose() {
    _textRecognizer.close();
  }
}