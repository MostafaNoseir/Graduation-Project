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

  // ✅ قاعدة البيانات في الذاكرة
  final Map<String, List<double>> _db = {};

  // ✅ المفتاح المستخدم في SharedPreferences
  static const String _storageKey = 'face_db';

  // =========================
  // INIT
  // =========================
  Future<void> init() async {
    try {
      print("Loading Face model...");

      _interpreter = await tfl.Interpreter.fromAsset(
        'assets/models/mobilefacenet.tflite',
      );

      _interpreter!.allocateTensors();

      print("Input shape:  ${_interpreter!.getInputTensor(0).shape}");
      print("Output shape: ${_interpreter!.getOutputTensor(0).shape}");

      // ✅ تحميل الوجوه المحفوظة
      await _loadDb();

      _isInitialized = true;
      print("FaceRecognitionService READY — ${_db.length} person(s) in DB");
    } catch (e, stack) {
      print("❌ Face init error: $e");
      print(stack);
      _isInitialized = false;
      _interpreter = null;
    }
  }

  // =========================
  // LOAD DB من SharedPreferences
  // =========================
  Future<void> _loadDb() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final jsonString = prefs.getString(_storageKey);

      if (jsonString == null || jsonString.isEmpty) {
        print("No saved faces found.");
        return;
      }

      // JSON: { "name": [0.1, 0.2, ...], ... }
      final Map<String, dynamic> decoded = jsonDecode(jsonString);

      _db.clear();
      decoded.forEach((name, value) {
        _db[name] = List<double>.from(value as List);
      });

      print("✅ Loaded ${_db.length} face(s) from local storage.");
    } catch (e) {
      print("❌ Failed to load face DB: $e");
    }
  }

  // =========================
  // SAVE DB في SharedPreferences
  // =========================
  Future<void> _saveDb() async {
    try {
      final prefs = await SharedPreferences.getInstance();

      // حوّل Map<String, List<double>> → JSON string
      final Map<String, List<double>> toEncode = Map.from(_db);
      final jsonString = jsonEncode(toEncode);

      await prefs.setString(_storageKey, jsonString);
      print("✅ Face DB saved — ${_db.length} person(s).");
    } catch (e) {
      print("❌ Failed to save face DB: $e");
    }
  }

  // =========================
  // ENROLL PERSON
  // =========================
  Future<bool> enrollPerson(String name, List samples) async {
    if (!isInitialized) return false;

    try {
      const int embSize = 192;
      List<double> sum = List.filled(embSize, 0.0);

      for (final s in samples) {
        final emb = _getEmbedding(s.bytes as Uint8List);
        for (int i = 0; i < embSize; i++) {
          sum[i] += emb[i];
        }
      }

      // متوسط الـ embeddings لتمثيل أفضل
      final avg = sum.map((e) => e / samples.length).toList();
      _db[name] = avg;

      // ✅ حفظ فوري بعد كل إضافة
      await _saveDb();

      print("✅ Enrolled & saved: $name");
      return true;
    } catch (e) {
      print("❌ Enroll error: $e");
      return false;
    }
  }

  // =========================
  // DELETE PERSON
  // =========================
  Future<bool> deletePerson(String name) async {
    if (!_db.containsKey(name)) return false;

    _db.remove(name);
    await _saveDb();

    print("🗑️ Deleted: $name");
    return true;
  }

  // =========================
  // LIST ENROLLED NAMES
  // =========================
  List<String> get enrolledNames => _db.keys.toList();

  // =========================
  // CLEAR ALL
  // =========================
  Future<void> clearAll() async {
    _db.clear();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_storageKey);
    print("🗑️ All faces cleared.");
  }

  // =========================
  // RECOGNIZE
  // =========================
  Future<String> recognizeFace(
      Uint8List imageBytes, {
        bool isArabic = false,
      }) async {
    if (!isInitialized) {
      return isArabic ? "النظام غير جاهز" : "System not ready";
    }

    if (_db.isEmpty) {
      return isArabic
          ? "لا يوجد وجوه محفوظة"
          : "No faces enrolled yet";
    }

    try {
      final embedding = _getEmbedding(imageBytes);

      String bestMatch = "Unknown";
      double bestScore = 0.0;

      _db.forEach((name, dbEmbedding) {
        final score = _cosineSimilarity(embedding, dbEmbedding);
        if (score > bestScore) {
          bestScore = score;
          bestMatch = name;
        }
      });

      if (bestScore < 0.60) {
        return isArabic ? "غير معروف" : "Unknown";
      }

      return isArabic ? "هذا هو $bestMatch" : "This is $bestMatch";
    } catch (e) {
      print("Recognize error: $e");
      return isArabic ? "خطأ في التعرف" : "Recognition error";
    }
  }

  // =========================
  // EMBEDDING PIPELINE
  // =========================
  List<double> _getEmbedding(Uint8List bytes) {
    final input = _preprocess(bytes);
    final output = List.generate(1, (_) => List.filled(192, 0.0));
    _interpreter!.run(input, output);
    return List<double>.from(output[0]);
  }

  // =========================
  // PREPROCESS IMAGE → 4D TENSOR
  // =========================
  List<List<List<List<double>>>> _preprocess(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception("Cannot decode image");

    final resized = img.copyResize(image, width: 112, height: 112);

    return [
      List.generate(112, (y) {
        return List.generate(112, (x) {
          final pixel = resized.getPixel(x, y);
          return [
            (pixel.r - 127.5) / 127.5,
            (pixel.g - 127.5) / 127.5,
            (pixel.b - 127.5) / 127.5,
          ];
        });
      })
    ];
  }

  // =========================
  // COSINE SIMILARITY
  // =========================
  double _cosineSimilarity(List<double> a, List<double> b) {
    double dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (sqrt(normA) * sqrt(normB));
  }

  // =========================
  // DISPOSE
  // =========================
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}