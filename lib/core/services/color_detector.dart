// lib/color_detector.dart
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:image/image.dart' as imgLib;
import 'package:palette_generator/palette_generator.dart';
import 'package:flutter/material.dart';

class ColorDetector {
  Future<String?> detectDominantColor(Uint8List bytes) async {
    try {
      final image = imgLib.decodeImage(bytes);
      if (image == null) return "فشل في قراءة الصورة";

      final paletteGenerator = await PaletteGenerator.fromImageProvider(
        MemoryImage(bytes),
        maximumColorCount: 10,
      );

      final dominantColor = paletteGenerator.dominantColor?.color;
      if (dominantColor == null) return "لم يتم اكتشاف لون رئيسي";

      final r = dominantColor.red;
      final g = dominantColor.green;
      final b = dominantColor.blue;

      // قائمة ألوان عربي + قابلة للتوسيع
      final Map<(int, int, int), String> colorMap = {
        (255, 0, 0): "أحمر",
        (255, 100, 100): "أحمر فاتح",
        (139, 0, 0): "أحمر غامق",
        (0, 255, 0): "أخضر",
        (0, 128, 0): "أخضر غامق",
        (144, 238, 144): "أخضر فاتح",
        (0, 0, 255): "أزرق",
        (0, 0, 139): "أزرق غامق",
        (173, 216, 230): "أزرق فاتح",
        (255, 255, 0): "أصفر",
        (255, 165, 0): "برتقالي",
        (255, 192, 203): "وردي",
        (255, 105, 180): "وردي غامق",
        (128, 0, 128): "بنفسجي",
        (0, 0, 0): "أسود",
        (255, 255, 255): "أبيض",
        (128, 128, 128): "رمادي",
        (165, 42, 42): "بني",
        (255, 255, 224): "بيج",
        (0, 255, 255): "سماوي",
      };

      String closestName = "لون غير معروف";
      double minDistance = double.infinity;

      colorMap.forEach((rgb, name) {
        double distance = math.sqrt(
          math.pow(r - rgb.$1, 2) +
              math.pow(g - rgb.$2, 2) +
              math.pow(b - rgb.$3, 2),
        );
        if (distance < minDistance) {
          minDistance = distance;
          closestName = name;
        }
      });

      return "اللون الرئيسي في الصورة هو $closestName";
    } catch (e) {
      print("خطأ في كشف اللون: $e");
      return "حدث خطأ أثناء كشف اللون";
    }
  }
}