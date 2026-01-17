// lib/color_detector.dart
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:image/image.dart' as imgLib;
import 'package:palette_generator/palette_generator.dart';
import 'package:flutter/material.dart';

class ColorDetector {
  Future<String?> detectDominantColor(
      Uint8List bytes, {
        bool isArabic = true,
      }) async {
    try {
      final image = imgLib.decodeImage(bytes);
      if (image == null) {
        return isArabic
            ? "فشل في قراءة الصورة"
            : "Failed to read image";
      }

      final paletteGenerator = await PaletteGenerator.fromImageProvider(
        MemoryImage(bytes),
        maximumColorCount: 10,
      );

      final dominantColor = paletteGenerator.dominantColor?.color;
      if (dominantColor == null) {
        return isArabic
            ? "لم يتم اكتشاف لون رئيسي"
            : "No dominant color detected";
      }

      final r = dominantColor.red;
      final g = dominantColor.green;
      final b = dominantColor.blue;

      /// RGB : { ar , en }
      final Map<(int, int, int), Map<String, String>> colorMap = {
        (255, 0, 0): {"ar": "أحمر", "en": "Red"},
        (255, 100, 100): {"ar": "أحمر فاتح", "en": "Light Red"},
        (139, 0, 0): {"ar": "أحمر غامق", "en": "Dark Red"},
        (0, 255, 0): {"ar": "أخضر", "en": "Green"},
        (0, 128, 0): {"ar": "أخضر غامق", "en": "Dark Green"},
        (144, 238, 144): {"ar": "أخضر فاتح", "en": "Light Green"},
        (0, 0, 255): {"ar": "أزرق", "en": "Blue"},
        (0, 0, 139): {"ar": "أزرق غامق", "en": "Dark Blue"},
        (173, 216, 230): {"ar": "أزرق فاتح", "en": "Light Blue"},
        (255, 255, 0): {"ar": "أصفر", "en": "Yellow"},
        (255, 165, 0): {"ar": "برتقالي", "en": "Orange"},
        (255, 192, 203): {"ar": "وردي", "en": "Pink"},
        (255, 105, 180): {"ar": "وردي غامق", "en": "Hot Pink"},
        (128, 0, 128): {"ar": "بنفسجي", "en": "Purple"},
        (0, 0, 0): {"ar": "أسود", "en": "Black"},
        (255, 255, 255): {"ar": "أبيض", "en": "White"},
        (128, 128, 128): {"ar": "رمادي", "en": "Gray"},
        (165, 42, 42): {"ar": "بني", "en": "Brown"},
        (255, 255, 224): {"ar": "بيج", "en": "Beige"},
        (0, 255, 255): {"ar": "سماوي", "en": "Cyan"},
      };

      String closestName =
      isArabic ? "لون غير معروف" : "Unknown color";
      double minDistance = double.infinity;

      colorMap.forEach((rgb, names) {
        final distance = math.sqrt(
          math.pow(r - rgb.$1, 2) +
              math.pow(g - rgb.$2, 2) +
              math.pow(b - rgb.$3, 2),
        );

        if (distance < minDistance) {
          minDistance = distance;
          closestName = isArabic ? names["ar"]! : names["en"]!;
        }
      });

      return isArabic
          ? "اللون الرئيسي في الصورة هو $closestName"
          : "The dominant color in the image is $closestName";
    } catch (e) {
      return isArabic
          ? "حدث خطأ أثناء كشف اللون"
          : "An error occurred while detecting color";
    }
  }
}