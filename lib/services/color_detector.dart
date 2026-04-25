// lib/color_detector.dart
//
// ✅ النسخة المحسّنة — إصلاح مشكلة الأسود والأبيض
//
//  السبب الجذري للمشكلة:
//  كاميرا الهاتف تضيف ضوضاء (noise) وتعديلات تلقائية على الصورة
//  فالأسود الحقيقي يُصبح L ≈ 0.20~0.35 بدلاً من 0.0
//  والأبيض الحقيقي يُصبح S ≈ 0.08~0.20 بدلاً من 0.0
//  لذلك وسّعنا حدودهما بشكل كبير
//
//  المنطق الجديد:
//  1. نقيس متوسط السطوع للصورة كلها أولاً (avgL)
//  2. نستخدم عتبات ديناميكية بناءً على avgL
//  3. إذا كانت الصورة كلها داكنة → نُخفّض عتبة الأسود
//  4. إذا كانت الصورة كلها فاتحة → نُخفّض عتبة التشبع للأبيض

import 'dart:typed_data';
import 'package:image/image.dart' as imgLib;

class ColorDetector {
  static const int _sampleStep = 6;

  Future<String?> detectDominantColor(
      Uint8List bytes, {
        bool isArabic = true,
      }) async {
    try {
      final image = imgLib.decodeImage(bytes);
      if (image == null) {
        return isArabic ? "فشل في قراءة الصورة" : "Failed to read image";
      }

      final small = imgLib.copyResize(image, width: 200);
      final w = small.width;
      final h = small.height;

      // ── المرحلة 1: حساب متوسط السطوع والتشبع للصورة كلها ──────────
      double sumL = 0;
      double sumS = 0;
      int count = 0;

      for (int y = 0; y < h; y += _sampleStep) {
        for (int x = 0; x < w; x += _sampleStep) {
          final pixel = small.getPixel(x, y);
          final hsl = _toHSL(pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt());
          sumL += hsl[2];
          sumS += hsl[1];
          count++;
        }
      }

      final avgL = count > 0 ? sumL / count : 0.5;
      final avgS = count > 0 ? sumS / count : 0.5;

      // ── المرحلة 2: تصنيف كل بيكسل بعتبات ديناميكية ────────────────
      final colorCounts = <String, int>{};
      int totalSamples = 0;

      for (int y = 0; y < h; y += _sampleStep) {
        for (int x = 0; x < w; x += _sampleStep) {
          final pixel = small.getPixel(x, y);
          final r = pixel.r.toInt();
          final g = pixel.g.toInt();
          final b = pixel.b.toInt();

          final name = _classifyPixel(r, g, b, isArabic, avgL, avgS);
          colorCounts[name] = (colorCounts[name] ?? 0) + 1;
          totalSamples++;
        }
      }

      if (totalSamples == 0 || colorCounts.isEmpty) {
        return isArabic ? "لم يتم اكتشاف لون رئيسي" : "No dominant color detected";
      }

      final sorted = colorCounts.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));

      final dominantName = sorted.first.key;

      return isArabic
          ? "اللون الرئيسي في الصورة هو $dominantName"
          : "The dominant color in the image is $dominantName";
    } catch (e) {
      print("Color detection error: $e");
      return isArabic
          ? "حدث خطأ أثناء كشف اللون"
          : "An error occurred while detecting color";
    }
  }

  // ══════════════════════════════════════════════════════════════
  //  تصنيف بيكسل واحد مع مراعاة متوسط الصورة
  // ══════════════════════════════════════════════════════════════
  String _classifyPixel(
      int r, int g, int b, bool isArabic, double avgL, double avgS) {

    final hsl = _toHSL(r, g, b);
    final h = hsl[0];
    final s = hsl[1];
    final l = hsl[2];

    // ── أسود ──────────────────────────────────────────────────────
    // عتبة ديناميكية: إذا الصورة كلها داكنة → نعتبر L < 0.45 أسوداً
    // إذا الصورة عادية → L < 0.28 أسود
    final blackThreshold = avgL < 0.35 ? 0.45 : 0.28;
    if (l <= blackThreshold && s <= 0.35) {
      return isArabic ? "أسود" : "Black";
    }

    // ── أبيض ──────────────────────────────────────────────────────
    // عتبة ديناميكية: إذا الصورة كلها فاتحة → نقبل تشبعاً أعلى قليلاً
    final whiteLThreshold = avgL > 0.65 ? 0.68 : 0.75;
    final whiteSThreshold = avgS < 0.15 ? 0.25 : 0.18;

    if (l >= whiteLThreshold && s <= whiteSThreshold) {
      return isArabic ? "أبيض" : "White";
    }

    // أبيض بيج فاتح جداً (مثل ورق الطباعة أو جدار أبيض في إضاءة دافئة)
    if (l >= 0.72 && s <= 0.22 && (h < 60 || h > 300)) {
      return isArabic ? "أبيض" : "White";
    }

    // ── رمادي ─────────────────────────────────────────────────────
    if (s <= 0.12 && l > blackThreshold && l < whiteLThreshold) {
      return isArabic ? "رمادي" : "Gray";
    }

    // ── بني ───────────────────────────────────────────────────────
    if (h >= 5 && h < 50 && l < 0.50 && s > 0.18) {
      return isArabic ? "بني" : "Brown";
    }

    // ── بيج ───────────────────────────────────────────────────────
    if (h >= 20 && h < 70 && l >= 0.55 && s >= 0.05 && s <= 0.38) {
      return isArabic ? "بيج" : "Beige";
    }

    // ── الألوان الكروماتية ─────────────────────────────────────────
    if (h < 15 || h >= 345) return isArabic ? "أحمر" : "Red";
    if (h < 40)              return isArabic ? "برتقالي" : "Orange";
    if (h < 70)              return isArabic ? "أصفر" : "Yellow";
    if (h < 155)             return isArabic ? "أخضر" : "Green";
    if (h < 195)             return isArabic ? "سماوي" : "Cyan";
    if (h < 255)             return isArabic ? "أزرق" : "Blue";
    if (h < 290)             return isArabic ? "بنفسجي" : "Purple";
    if (h < 345)             return isArabic ? "وردي" : "Pink";

    return isArabic ? "رمادي" : "Gray";
  }

  // ══════════════════════════════════════════════════════════════
  //  تحويل RGB → [H (0-360), S (0-1), L (0-1)]
  // ══════════════════════════════════════════════════════════════
  List<double> _toHSL(int r, int g, int b) {
    final rf = r / 255.0;
    final gf = g / 255.0;
    final bf = b / 255.0;

    final maxC = rf > gf ? (rf > bf ? rf : bf) : (gf > bf ? gf : bf);
    final minC = rf < gf ? (rf < bf ? rf : bf) : (gf < bf ? gf : bf);
    final delta = maxC - minC;

    final l = (maxC + minC) / 2.0;

    double s = 0.0;
    if (delta > 0) {
      final denom = 1.0 - (2.0 * l - 1.0).abs();
      s = denom <= 0 ? 0.0 : (delta / denom).clamp(0.0, 1.0);
    }

    double h = 0.0;
    if (delta > 0) {
      if (maxC == rf) {
        h = ((gf - bf) / delta) % 6.0;
      } else if (maxC == gf) {
        h = (bf - rf) / delta + 2.0;
      } else {
        h = (rf - gf) / delta + 4.0;
      }
      h = (h / 6.0) * 360.0;
      if (h < 0) h += 360.0;
    }

    return [h, s, l];
  }
}