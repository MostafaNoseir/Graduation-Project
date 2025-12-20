// lib/core/services/face_embedder.dart

import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img_lib;
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';

class FaceEmbedder {
  late Interpreter _interpreter;
  final String modelPath = "assets/models/MobileFaceNet.tflite";

  List<String> knownNames = [];
  List<List<double>> knownEmbeddings = [];

  static const double THRESHOLD = 0.75;
  static const String DB_KEY = "face_database";

  Future<void> init() async {
    _interpreter = await Interpreter.fromAsset(modelPath);
    await _loadDatabase();

    print("Input shape: ${_interpreter.getInputTensor(0).shape}");
    print("Input type: ${_interpreter.getInputTensor(0).type}");
    print("Output shape: ${_interpreter.getOutputTensor(0).shape}");
    print("Output type: ${_interpreter.getOutputTensor(0).type}");
  }

  Future<void> _loadDatabase() async {
    final prefs = await SharedPreferences.getInstance();
    final String? jsonString = prefs.getString(DB_KEY);

    if (jsonString == null || jsonString.isEmpty) {
      knownNames = [];
      knownEmbeddings = [];
      return;
    }

    try {
      final Map<String, dynamic> data = json.decode(jsonString);
      knownNames = List<String>.from(data['names'] as List<dynamic>);
      knownEmbeddings = List<List<double>>.from(
          (data['embeddings'] as List<dynamic>).map((dynamic embedding) {
            return List<double>.from(
                (embedding as List<dynamic>).map((dynamic value) => (value as num).toDouble()));
          }));
      print("تم تحميل ${knownNames.length} شخص من القاعدة المحلية.");
    } catch (e) {
      print("خطأ في قراءة قاعدة البيانات: $e");
      knownNames = [];
      knownEmbeddings = [];
    }
  }

  Future<void> _saveDatabase() async {
    final prefs = await SharedPreferences.getInstance();
    final data = {
      "names": knownNames,
      "embeddings": knownEmbeddings,
    };
    await prefs.setString(DB_KEY, json.encode(data));
    print("تم حفظ قاعدة البيانات (${knownNames.length} شخص).");
  }

  Future<bool> enrollPerson(String name, List<XFile> images) async {
    print("بدء تسجيل: $name مع ${images.length} صورة");

    if (name.isEmpty) {
      print("الاسم فارغ");
      return false;
    }

    if (images.length < 3) {
      print("ينصح باستخدام 3 صور على الأقل");
      return false;
    }

    List<List<double>> embeddings = [];

    for (int i = 0; i < images.length; i++) {
      print("معالجة الصورة ${i + 1}/${images.length}");

      final Uint8List bytes = await images[i].readAsBytes();
      img_lib.Image? original = img_lib.decodeImage(bytes);

      if (original == null) {
        print("فشل في فك تشفير الصورة ${i + 1}");
        continue;
      }

      // كشف وجه بسيط (لو مفيش كشف وجه نستخدم الصورة كاملة)
      img_lib.Image faceImage = original;

      final Float32List input = _preprocess(faceImage);
      final Float32List output = Float32List(512);

      try {
        _interpreter.run(input, output); // بدون reshape
        embeddings.add(output.toList());
        print("تم استخراج embedding للصورة ${i + 1}");
      } catch (e) {
        print("خطأ في تشغيل الموديل على الصورة ${i + 1}: $e");
        continue;
      }
    }

    if (embeddings.isEmpty) {
      print("لم يتم استخراج أي embeddings.");
      return false;
    }

    final List<double> meanEmbedding = _calculateMeanEmbedding(embeddings);

    int index = knownNames.indexOf(name);
    if (index != -1) {
      knownEmbeddings[index] = meanEmbedding;
      print("تم تحديث embeddings للشخص '$name' مع ${embeddings.length} صور.");
    } else {
      knownNames.add(name);
      knownEmbeddings.add(meanEmbedding);
      print("تم تسجيل الشخص '$name' بنجاح مع ${embeddings.length} صور.");
    }

    await _saveDatabase();
    return true;
  }

  List<double> _calculateMeanEmbedding(List<List<double>> embeddings) {
    final int length = embeddings.length;
    final List<double> result = List<double>.filled(512, 0.0);

    for (var emb in embeddings) {
      for (int i = 0; i < 512; i++) {
        result[i] += emb[i] / length;
      }
    }
    return result;
  }

  double cosineSimilarity(List<double> a, List<double> b) {
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA == 0 || normB == 0) return 0.0;
    return dot / (math.sqrt(normA) * math.sqrt(normB));
  }

  img_lib.Image _convertCameraImage(CameraImage cameraImage) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    final Uint8List bytes = cameraImage.planes[0].bytes;

    final img_lib.Image image = img_lib.Image(width: width, height: height);
    int pixelIndex = 0;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int b = bytes[pixelIndex++];
        final int g = bytes[pixelIndex++];
        final int r = bytes[pixelIndex++];
        pixelIndex++; // alpha
        image.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return image;
  }

  Float32List _preprocess(img_lib.Image src) {
    final img_lib.Image resized = img_lib.copyResize(src, width: 112, height: 112);

    final Float32List input = Float32List(1 * 112 * 112 * 3);
    int index = 0;

    for (int y = 0; y < 112; y++) {
      for (int x = 0; x < 112; x++) {
        final pixel = resized.getPixel(x, y);
        input[index++] = (pixel.r.toDouble() - 127.5) / 128.0;
        input[index++] = (pixel.g.toDouble() - 127.5) / 128.0;
        input[index++] = (pixel.b.toDouble() - 127.5) / 128.0;
      }
    }

    return input;
  }

  Future<String?> recognizeFrame(CameraImage cameraImage) async {
    if (knownNames.isEmpty) {
      return null;
    }

    final img_lib.Image image = _convertCameraImage(cameraImage);
    final Float32List input = _preprocess(image);
    final Float32List output = Float32List(512);

    try {
      _interpreter.run(input, output);
    } catch (e) {
      print("خطأ في التعرف: $e");
      return null;
    }

    final List<double> currentEmbedding = output.toList();

    String? bestName;
    double bestScore = -1.0;

    for (int i = 0; i < knownEmbeddings.length; i++) {
      final double score = cosineSimilarity(currentEmbedding, knownEmbeddings[i]);
      if (score > bestScore) {
        bestScore = score;
        bestName = knownNames[i];
      }
    }

    if (bestScore >= THRESHOLD) {
      print("تم التعرف على الشخص: $bestName (score=${bestScore.toStringAsFixed(2)})");
      return bestName;
    } else {
      print("الشخص غير معروف (أفضل تطابق: $bestName, score=${bestScore.toStringAsFixed(2)})");
      return null;
    }
  }

}
