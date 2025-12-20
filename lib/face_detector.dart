// lib/face_detector.dart
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as imgLib;
import 'package:hive_flutter/hive_flutter.dart';

class FaceDetector {
  late tfl.Interpreter _interpreter;
  late Box<List<double>> _db;

  static const int INPUT_SIZE = 112;

  Future<void> init() async {
    _interpreter = await tfl.Interpreter.fromAsset('assets/models/MobileFaceNet.tflite');
    await Hive.initFlutter();
    _db = await Hive.openBox<List<double>>('face_db');
  }

  Future<Float32List?> getEmbedding(Uint8List bytes) async {
    try {
      final img = imgLib.decodeImage(bytes);
      if (img == null) return null;

      // نريزايز الصورة كلها لـ 112x112 (بدون crop، بس كفاية لو الوجه في الوسط)
      final resized = imgLib.copyResize(img, width: INPUT_SIZE, height: INPUT_SIZE);

      final input = Float32List(1 * 3 * INPUT_SIZE * INPUT_SIZE);
      int idx = 0;
      for (int y = 0; y < INPUT_SIZE; y++) {
        for (int x = 0; x < INPUT_SIZE; x++) {
          final pixel = resized.getPixel(x, y);
          input[idx++] = (pixel.r - 127.5) / 128.0;
          input[idx++] = (pixel.g - 127.5) / 128.0;
          input[idx++] = (pixel.b - 127.5) / 128.0;
        }
      }

      final output = List.filled(1 * 192, 0.0).reshape([1, 192]);
      _interpreter.run(input.buffer.asUint8List(), output);

      return Float32List.fromList(output[0]);
    } catch (e) {
      print("خطأ في استخراج embedding: $e");
      return null;
    }
  }

  double cosineSimilarity(Float32List a, Float32List b) {
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (math.sqrt(normA) * math.sqrt(normB));
  }

  Future<void> enroll(String name, List<Float32List> embeddings) async {
    final meanEmb = Float32List(192);
    for (var emb in embeddings) {
      for (int i = 0; i < 192; i++) meanEmb[i] += emb[i];
    }
    for (int i = 0; i < 192; i++) meanEmb[i] /= embeddings.length;

    await _db.put(name, meanEmb.toList());
  }

  Future<String> recognize(Float32List emb) async {
    if (_db.isEmpty) return "غير معروف";

    String bestName = "غير معروف";
    double bestScore = 0.0;

    for (var key in _db.keys) {
      final storedList = _db.get(key)!;
      final stored = Float32List.fromList(storedList.cast<double>());
      final score = cosineSimilarity(emb, stored);
      if (score > bestScore) {
        bestScore = score;
        bestName = key.toString();
      }
    }

    return bestScore > 0.75 ? bestName : "غير معروف";
  }

  void dispose() {
    _interpreter.close();
  }
}