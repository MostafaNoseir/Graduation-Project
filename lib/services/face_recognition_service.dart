import 'dart:math';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;

class FaceRecognitionService {
  tfl.Interpreter? _interpreter;

  bool _isInitialized = false;
  bool get isInitialized => _isInitialized && _interpreter != null;

  final Map<String, List<double>> _db = {};

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

      print("Input shape: ${_interpreter!.getInputTensor(0).shape}");
      print("Output shape: ${_interpreter!.getOutputTensor(0).shape}");

      await _loadDb();

      _isInitialized = true;
      print("FaceRecognitionService READY");
    } catch (e, stack) {
      print("❌ Face init error: $e");
      print(stack);

      _isInitialized = false;
      _interpreter = null;
    }
  }

  Future<void> _loadDb() async {
    // TODO: load embeddings from storage (Hive / Firestore)
  }

  // =========================
  // ENROLL PERSON
  // =========================
  Future<bool> enrollPerson(String name, List samples) async {
    if (!isInitialized) return false;

    try {
      const int size = 192;

      List<double> sum = List.filled(size, 0.0);

      for (final s in samples) {
        final emb = _getEmbedding(s.bytes);

        for (int i = 0; i < size; i++) {
          sum[i] += emb[i];
        }
      }

      final avg = sum.map((e) => e / samples.length).toList();
      _db[name] = avg;

      print("Enrolled: $name");
      return true;
    } catch (e) {
      print("Enroll error: $e");
      return false;
    }
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

      return bestMatch;
    } catch (e) {
      print("Recognize error: $e");
      return isArabic ? "خطأ في التعرف" : "Recognition error";
    }
  }

  // =========================
  // EMBEDDING PIPELINE (IMPORTANT FIX)
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
    if (image == null) {
      throw Exception("Cannot decode image");
    }

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