import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;

class FaceRecognitionService {
  tfl.Interpreter? _interpreter;

  bool _isInitialized = false;
  bool get isInitialized => _isInitialized && _interpreter != null;

  final Map<String, List<double>> _db = {};
  static const String _storageKey = 'face_db';
  static const double _sameFaceThreshold = 0.70;
  static const double _recognizeThreshold = 0.60;

  Future<void> init() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset(
        'assets/models/mobilefacenet.tflite',
      );
      _interpreter!.allocateTensors();
      await _loadDb();
      _isInitialized = true;
      print("FaceRecognitionService READY — ${_db.length} person(s)");
    } catch (e, s) {
      print("❌ Face init error: $e\n$s");
      _isInitialized = false;
      _interpreter = null;
    }
  }

  Future<void> _loadDb() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final json = prefs.getString(_storageKey);
      if (json == null || json.isEmpty) return;
      final Map<String, dynamic> decoded = jsonDecode(json);
      _db.clear();
      decoded.forEach((k, v) => _db[k] = List<double>.from(v as List));
    } catch (e) {
      print("❌ loadDb: $e");
    }
  }

  Future<void> _saveDb() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_storageKey, jsonEncode(Map.from(_db)));
    } catch (e) {
      print("❌ saveDb: $e");
    }
  }

  // ── تطبيع الاسم: يوحّد الهمزة والألف ويحوّل الإنجليزي لصغير ──────────
  static String _normalizeName(String name) {
    // توحيد كل أشكال الألف والهمزة → ا
    String n = name.trim();
    n = n.replaceAll(RegExp(r'[أإآ]'), 'ا');
    // توحيد الياء المقصورة والياء العادية → ي
    n = n.replaceAll('ى', 'ي');
    // توحيد التاء المربوطة → ة  (اختياري — أبقيناها كما هي)
    // تصغير الحروف الإنجليزية
    n = n.toLowerCase();
    return n;
  }

  Future<EnrollResult> enrollPerson(
      String name,
      List samples, {
        bool isArabic = false,
      }) async {
    if (!isInitialized) {
      return EnrollResult(
          success: false,
          message: isArabic ? "النظام غير جاهز" : "System not ready");
    }

    final trimmed  = name.trim();           // الاسم الأصلي للعرض
    final normalized = _normalizeName(trimmed); // المفتاح الموحَّد في الـ DB

    // تحقق إن كان هناك اسم مطابق بعد التطبيع
    final existingKey = _db.keys.firstWhere(
          (k) => _normalizeName(k) == normalized,
      orElse: () => '',
    );

    if (existingKey.isNotEmpty) {
      return EnrollResult(
        success: false,
        message: isArabic
            ? "الاسم \"$existingKey\" موجود بالفعل، اختر اسماً مختلفاً"
            : "Name \"$existingKey\" already exists. Choose a different name.",
      );
    }

    try {
      const int embSize = 192;
      final sum = List.filled(embSize, 0.0);
      int valid = 0;

      for (final s in samples) {
        try {
          final emb = _getEmbedding(s.bytes as Uint8List);
          for (int i = 0; i < embSize; i++) sum[i] += emb[i];
          valid++;
        } catch (_) {}
      }

      if (valid == 0) {
        return EnrollResult(
            success: false,
            message: isArabic ? "فشل في قراءة الصور" : "Failed to process images");
      }

      final avg = sum.map((e) => e / valid).toList();

      String? existingName;
      double bestScore = 0.0;
      _db.forEach((n, emb) {
        final s = _cosineSimilarity(avg, emb);
        if (s > bestScore) {
          bestScore = s;
          existingName = n;
        }
      });

      if (existingName != null && bestScore >= _sameFaceThreshold) {
        final old = existingName!;
        _db.remove(old);
        _db[trimmed] = avg;
        await _saveDb();
        return EnrollResult(
          success: true,
          message: isArabic ? "تم تغيير الاسم من" : "Name changed from",
          name: trimmed,
          oldName: old,
          isUpdate: true,
        );
      }

      _db[trimmed] = avg;
      await _saveDb();
      return EnrollResult(
        success: true,
        message: isArabic ? "تم تسجيل" : "Enrolled",
        name: trimmed,
      );
    } catch (e) {
      return EnrollResult(
          success: false,
          message: isArabic ? "خطأ أثناء التسجيل" : "Enrollment error");
    }
  }

  Future<RecognizeResult> recognizeFace(
      Uint8List imageBytes, {
        bool isArabic = false,
      }) async {
    if (!isInitialized) {
      return RecognizeResult(
          found: false,
          message: isArabic ? "النظام غير جاهز" : "System not ready");
    }
    if (_db.isEmpty) {
      return RecognizeResult(
          found: false,
          message: isArabic ? "لا يوجد وجوه محفوظة" : "No faces enrolled yet");
    }

    try {
      final emb = _getEmbedding(imageBytes);
      String bestMatch = "";
      double bestScore = 0.0;
      _db.forEach((n, e) {
        final s = _cosineSimilarity(emb, e);
        if (s > bestScore) {
          bestScore = s;
          bestMatch = n;
        }
      });

      if (bestScore < _recognizeThreshold || bestMatch.isEmpty) {
        return RecognizeResult(
            found: false,
            message: isArabic ? "غير معروف" : "Unknown");
      }

      return RecognizeResult(
          found: true,
          prefix: isArabic ? "هذا هو" : "This is",
          name: bestMatch);
    } catch (e) {
      return RecognizeResult(
          found: false,
          message: isArabic ? "خطأ في التعرف" : "Recognition error");
    }
  }

  List<String> get enrolledNames => _db.keys.toList();

  Future<bool> deletePerson(String name) async {
    // ابحث بالاسم المطابق بعد التطبيع أيضاً
    final key = _db.keys.firstWhere(
          (k) => k == name || _normalizeName(k) == _normalizeName(name),
      orElse: () => '',
    );
    if (key.isEmpty) return false;
    _db.remove(key);
    await _saveDb();
    return true;
  }

  /// حذف قائمة من الأشخاص دفعةً واحدة
  Future<int> deletePersons(List<String> names) async {
    int count = 0;
    for (final name in names) {
      final key = _db.keys.firstWhere(
            (k) => k == name || _normalizeName(k) == _normalizeName(name),
        orElse: () => '',
      );
      if (key.isNotEmpty) {
        _db.remove(key);
        count++;
      }
    }
    if (count > 0) await _saveDb();
    return count;
  }

  Future<void> resetAll() async {
    _db.clear();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_storageKey);
  }

  List<double> _getEmbedding(Uint8List bytes) {
    final input = _preprocess(bytes);
    final output = List.generate(1, (_) => List.filled(192, 0.0));
    _interpreter!.run(input, output);
    return List<double>.from(output[0]);
  }

  List<List<List<List<double>>>> _preprocess(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception("Cannot decode image");
    final resized = img.copyResize(image, width: 112, height: 112);
    return [
      List.generate(
        112,
            (y) => List.generate(112, (x) {
          final p = resized.getPixel(x, y);
          return [
            (p.r - 127.5) / 127.5,
            (p.g - 127.5) / 127.5,
            (p.b - 127.5) / 127.5,
          ];
        }),
      )
    ];
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    double dot = 0, nA = 0, nB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      nA += a[i] * a[i];
      nB += b[i] * b[i];
    }
    return dot / (sqrt(nA) * sqrt(nB));
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}

class EnrollResult {
  final bool success;
  final String message;   // prefix الجملة (بدون اسم)
  final String name;      // الاسم الجديد
  final String oldName;   // الاسم القديم (في حالة التحديث)
  final bool isUpdate;    // هل هو تحديث لاسم موجود؟
  const EnrollResult({
    required this.success,
    required this.message,
    this.name = "",
    this.oldName = "",
    this.isUpdate = false,
  });
}

class RecognizeResult {
  final bool found;
  final String message;
  final String prefix;
  final String name;

  const RecognizeResult({
    required this.found,
    this.message = "",
    this.prefix = "",
    this.name = "",
  });
}