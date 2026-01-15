// lib/text_detector.dart
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

class TextDetector {
  static final _textRecognizer = TextRecognizer();

  Future<String?> extractText(String imagePath) async {
    try {
      final inputImage = InputImage.fromFilePath(imagePath);
      final recognizedText = await _textRecognizer.processImage(inputImage);

      // لو مفيش نص
      if (recognizedText.blocks.isEmpty) {
        return "لم يتم اكتشاف أي نص في الصورة";
      }

      // ترتيب الكتل (blocks) حسب الموقع العلوي (من أعلى لأسفل)
      final sortedBlocks = recognizedText.blocks.toList()
        ..sort((a, b) => a.boundingBox.top.compareTo(b.boundingBox.top));

      final List<String> lines = [];

      for (var block in sortedBlocks) {
        // ترتيب السطور داخل الكتلة (من أعلى لأسفل)
        final sortedLines = block.lines.toList()
          ..sort((a, b) => a.boundingBox.top.compareTo(b.boundingBox.top));

        for (var line in sortedLines) {
          String lineText = line.text.trim();

          // تنظيف بسيط: إزالة مسافات زايدة
          lineText = lineText.replaceAll(RegExp(r'\s+'), ' ');

          if (lineText.isNotEmpty) {
            lines.add(lineText);
          }
        }

        // فاصل بين الكتل (فقرات)
        if (lines.isNotEmpty) {
          lines.add(""); // سطر فارغ بين الفقرات
        }
      }

      // جمع النص كله
      String fullText = lines.join('\n').trim();

      // تنظيف نهائي
      fullText = fullText
          .replaceAll(RegExp(r'\n{3,}'), '\n\n') // مش أكتر من سطرين فارغين
          .trim();

      if (fullText.isEmpty) {
        return "تم اكتشاف نص لكن غير قابل للقراءة";
      }

      // كلام أحسن للـ TTS
      return "النص المكتشف في الصورة:\n\n$fullText";

    } catch (e) {
      print("خطأ في قراءة النص: $e");
      return "حدث خطأ أثناء قراءة النص";
    }
  }

  static void dispose() {
    _textRecognizer.close();
  }
}